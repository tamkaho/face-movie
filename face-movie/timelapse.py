from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import numpy as np


def add_text_to_image(image: np.ndarray, text: str) -> np.ndarray:
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Set up font for the text
    font_path = str(Path(cv2.__path__[0]) / "qt" / "fonts" / "DejaVuSans.ttf")
    font = ImageFont.truetype(font_path, size=64)

    # Add the text at the bottom of the image
    text_width, text_height = draw.textsize(text, font=font)
    text_x = (image.shape[1] - text_width) // 2
    text_y = image.shape[0] - text_height - 10
    draw.text((text_x, text_y), text, fill="white", font=font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def add_thumbnails_to_timelapse(
    image: np.ndarray,
    i: int,
    image_paths: list[Path],
    n_thumb: int = 8,
    x_crop: float = 0.5,
):
    """
    Add thumbnails of the images in the timelapse to the bottom of the image.
    image: The image to add the thumbnails to
    i: The index of the current image in the timelapse
    image_paths: The list of image paths in the timelapse
    n_thumb: The number of thumbnails to add
    x_crop: The left/right of the thumbs can be cropped. Choose 1 to keep the whole image.
    """
    h, w = image.shape[:2]
    thumb_w = w // n_thumb
    curr_fraction = i / len(image_paths)
    show_n_thumbs = int(np.ceil(curr_fraction * n_thumb))

    thumbs = []
    for n in range(show_n_thumbs):
        r = 1 / (n_thumb - n + 1)
        f = n / n_thumb + r * (curr_fraction - n / n_thumb)
        thumb = cv2.imread(str(image_paths[int(f * len(image_paths))]))
        x_start = int(thumb.shape[1] * (1 - x_crop) / 2)
        x_end = x_start + int(thumb.shape[1] * x_crop)
        thumb = thumb[:, x_start:x_end]
        scale = thumb_w / thumb.shape[1]
        thumb = cv2.resize(thumb, None, fx=scale, fy=scale)
        thumbs.append(thumb)

    if len(thumbs) > 0:
        # crop_last = 1 - (curr_fraction - show_n_thumbs / n_thumb) * n_thumb
        thumbs = np.hstack(thumbs)
        thumbs = thumbs[:, : int(w * curr_fraction)]
        image[h - thumbs.shape[0] :, : thumbs.shape[1]] = thumbs

    return image


def create_timelapse(images_folder: Path, output_path: Path, fps=30):
    # Get the list of image file paths in the folder
    image_paths = sorted(images_folder.glob("*.jpg"))
    # Read the first image to get the dimensions
    first_image = cv2.imread(str(image_paths[0]))
    height, width, _ = first_image.shape
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # or use 'XVID' for AVI format
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    # Write each image to the video writer
    image = None
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        image = add_thumbnails_to_timelapse(image, i, image_paths)
        video_writer.write(image)
    # Add some pause frames to the end of the video
    for _ in range(end_pause_frames):
        video_writer.write(image)
    # Release the video writer
    video_writer.release()


# Example usage
images_folder = Path("./running_avg_5")
output_path = Path("./running_avg_5.mp4")
fps = 7
end_pause_frames = 10  # Number of pause frames to add to the end of the video
create_timelapse(images_folder, output_path, fps)
