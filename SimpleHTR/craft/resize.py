from PIL import Image

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(size, Image.LANCZOS)  # Use Lanczos filter for high-quality resizing
    resized_image.save(output_image_path)  # Save the resized image

# Replace 'original.png' and 'resized.png' with your image paths
resize_image('SimpleHTR/craft/img.png', 'SimpleHTR/craft/outputs/resizedimg.png', (500, 500))  #(271, 186)