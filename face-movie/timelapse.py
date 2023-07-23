from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path

def add_text_to_image(image, text):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Set up font for the text
    font_path = str(Path(cv2.__path__[0]) / 'qt' / 'fonts' / 'DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=64)
    
    # Add the text at the bottom of the image
    text_width, text_height = draw.textsize(text, font=font)
    text_x = (image.shape[1] - text_width) // 2
    text_y = image.shape[0] - text_height - 10
    draw.text((text_x, text_y), text, fill="white", font=font)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def create_timelapse(images_folder, output_path, fps=30):
    # Get the list of image file paths in the folder
    image_paths = sorted(Path(images_folder).glob('*.jpg'))
    # Read the first image to get the dimensions
    first_image = cv2.imread(str(image_paths[0]))
    height, width, _ = first_image.shape
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or use 'XVID' for AVI format
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Write each image to the video writer
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        video_writer.write(image)
    # Release the video writer
    video_writer.release()

# Example usage
images_folder = "./aligned"
output_path = "./video.mp4"
fps = 5
create_timelapse(images_folder, output_path, fps)
