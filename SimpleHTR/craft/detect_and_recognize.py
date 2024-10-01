from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_extra_results,
    empty_cuda_cache
)
import numpy as np
import cv2
import os

# Function to cluster boxes into lines
def cluster_boxes(boxes, y_threshold=10):
    # Sort boxes by their vertical position
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
    
    lines.append(current_line)  # Add the last line
    return lines

# Function to merge boxes horizontally into a single bounding box for each line
def merge_boxes(lines):
    merged_boxes = []
    for line in lines:
        x_min = min([min(box[:, 0]) for box in line])
        x_max = max([max(box[:, 0]) for box in line])
        y_min = min([min(box[:, 1]) for box in line])
        y_max = max([max(box[:, 1]) for box in line])
        merged_boxes.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
    return merged_boxes

# Set image path and export folder directory
image_path = 'SimpleHTR/craft/img.png'  # can be filepath, PIL image or numpy array
output_dir = 'SimpleHTR/outputs/'
line_crops_dir = os.path.join(output_dir, 'line_crops')

# Create output directory if it doesn't exist
os.makedirs(line_crops_dir, exist_ok=True)

# Read image
image = read_image(image_path)

# Load models
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)

# Perform prediction
prediction_result = get_prediction(
    image=image,
    craft_net=craft_net,
    refine_net=refine_net,
    text_threshold=0.2, #0.4
    link_threshold=0.1, #0.4
    low_text=0.5,       #0.2
    cuda=False,
    long_size=255       #4000
)

# Cluster and merge boxes
lines = cluster_boxes(prediction_result["boxes"])
merged_boxes = merge_boxes(lines)

# Export detected text regions for merged boxes
for idx, box in enumerate(merged_boxes):
    x_min = int(min(box[:, 0]))
    x_max = int(max(box[:, 0]))
    y_min = int(min(box[:, 1]))
    y_max = int(max(box[:, 1]))
    
    cropped_img = image[y_min:y_max, x_min:x_max]
    cv2.imwrite(os.path.join(line_crops_dir, f'line_{idx}.png'), cropped_img)

# Export heatmap, detection points, box visualization
export_extra_results(
    image=image,
    regions=merged_boxes,
    heatmaps=prediction_result["heatmaps"],
    output_dir=output_dir
)

# Unload models from GPU
empty_cuda_cache()