from PIL import Image
import os
from tqdm import tqdm

def resize_images(source_dir, dest_dir, target_size):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Get a list of all files in the source directory
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        # Open the image
        image_path = os.path.join(source_dir, image_file)
        img = Image.open(image_path)

        # Resize the image
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Save the resized image to the destination directory
        dest_path = os.path.join(dest_dir, image_file)
        img.save(dest_path)

if __name__ == "__main__":

    for res in tqdm([32, 64, 128]):
        for cur_obj in tqdm(["drums", "lego" "hotdog", "chair", , "materials", "mic", "ship", "ficus"]):
            source_directory = f"data/blender/{cur_obj}/test_800"
            destination_directory = f"data/blender/{cur_obj}/test_{res}"

            # Set the target size for resizing
            target_size = (res, res)

            # Resize images
            resize_images(source_directory, destination_directory, target_size)
