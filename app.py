from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import os
import numpy as np
import cv2
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_extra_results,
    empty_cuda_cache
)
import json
from typing import List
import editdistance
from path import Path
from SimpleHTR.src.dataloader_iam import DataLoaderIAM, Batch
from SimpleHTR.src.model import Model, DecoderType
from SimpleHTR.src.preprocessor import Preprocessor
import shutil

app = FastAPI()

class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = 'SimpleHTR/model/charList.txt'
    fn_summary = 'SimpleHTR/model/summary.json'
    fn_corpus = 'SimpleHTR/data/corpus.txt'

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(size, Image.LANCZOS)
    resized_image.save(output_image_path)

def cluster_boxes(boxes, y_threshold=10):
    sorted_boxes = sorted(boxes, key=lambda box: min(box[:, 1]))
    lines = []
    current_line = []
    current_y = min(sorted_boxes[0][:, 1])
    for box in sorted_boxes:
        box_y = min(box[:, 1])
        if abs(box_y - current_y) > y_threshold:
            lines.append(current_line)
            current_line = []
            current_y = box_y
        current_line.append(box)
    lines.append(current_line)
    return lines

def merge_boxes(lines):
    merged_boxes = []
    for line in lines:
        x_min = min([min(box[:, 0]) for box in line])
        x_max = max([max(box[:, 0]) for box in line])
        y_min = min([min(box[:, 1]) for box in line])
        y_max = max([max(box[:, 1]) for box in line])
        merged_boxes.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
    return merged_boxes

def get_img_height() -> int:
    return 32

def get_img_size(line_mode: bool = False) :
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())

def infer(model: Model, img_dir: Path) -> str:
    recognized_lines = []
    for img_file in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None
        preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
        img = preprocessor.process_img(img)
        batch = Batch([img], None, 1)
        recognized, probability = model.infer_batch(batch, True)
        recognized_lines.append(recognized[0])
    final_output = ' '.join(recognized_lines)
    return final_output

def run_craft(image_path, output_dir):
    image = read_image(image_path)
    refine_net = load_refinenet_model(cuda=False)
    craft_net = load_craftnet_model(cuda=False)
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.2,
        link_threshold=0.1,
        low_text=0.5,
        cuda=False,
        long_size=255
    )
    lines = cluster_boxes(prediction_result["boxes"])
    merged_boxes = merge_boxes(lines)
    line_crops_dir = os.path.join(output_dir, 'line_crops')
    os.makedirs(line_crops_dir, exist_ok=True)
    for idx, box in enumerate(merged_boxes):
        x_min = int(min(box[:, 0]))
        x_max = int(max(box[:, 0]))
        y_min = int(min(box[:, 1]))
        y_max = int(max(box[:, 1]))
        cropped_img = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(os.path.join(line_crops_dir, f'line_{idx}.png'), cropped_img)
    export_extra_results(
        image=image,
        regions=merged_boxes,
        heatmaps=prediction_result["heatmaps"],
        output_dir=output_dir
    )
    empty_cuda_cache()
    return line_crops_dir

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, and JPEG files are allowed.")

    contents = await file.read()
    input_image_path = "input_image.png"
    with open(input_image_path, "wb") as f:
        f.write(contents)

    resize_image(input_image_path, "SimpleHTR/craft/outputs/resizedimg.png", (500, 500))

    output_dir = 'SimpleHTR/outputs/'
    line_crops_dir = run_craft("SimpleHTR/craft/outputs/resizedimg.png", output_dir)

    decoder_type = DecoderType.WordBeamSearch
    model = Model(char_list_from_file(), decoder_type, must_restore=True)
    final_output = infer(model, Path(line_crops_dir))

    # Clean up files
    if os.path.exists(input_image_path):
        os.remove(input_image_path)
    if os.path.exists("SimpleHTR/craft/outputs/resizedimg.png"):
        os.remove("SimpleHTR/craft/outputs/resizedimg.png")
    shutil.rmtree(output_dir)

    return JSONResponse(content={"recognized_text": final_output})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.10.18.253" ,port=8000)