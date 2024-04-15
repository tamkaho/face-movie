import argparse
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


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
    n_thumb: int = 11,
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
    r = 1 / (n_thumb + 1)
    curr_fraction = min(
        1, max(0, (i - running_avg) / (len(image_paths) - 2 * running_avg))
    )
    show_n_thumbs = int(np.ceil(curr_fraction * n_thumb))

    thumbs = []
    for n in range(show_n_thumbs):
        f = n / n_thumb + r * (curr_fraction - n / n_thumb)
        thumb = cv2.imread(
            str(
                image_paths[int(f * (len(image_paths) - 2 * running_avg) + running_avg)]
            )
        )
        x_start = int(thumb.shape[1] * (1 - x_crop) / 2)
        x_end = x_start + int(thumb.shape[1] * x_crop)
        thumb = thumb[:, x_start:x_end]
        thumbs.append(thumb)

    if len(thumbs) > 0:
        thumbs = np.hstack(thumbs)
        scale = w * show_n_thumbs / (thumbs.shape[1] * n_thumb)
        thumbs = cv2.resize(thumbs, None, fx=scale, fy=scale)
        thumbs = thumbs[:, : int(w * curr_fraction)]
        image[h - thumbs.shape[0] :, : thumbs.shape[1]] = thumbs

    return image

def read_and_process_image(image_path: Path, index: int, image_paths: List[Path]) -> Tuple[int, np.ndarray]:
    image = cv2.imread(str(image_path))
    image = add_thumbnails_to_timelapse(image, index, image_paths)
    return (index, image)

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
    with ThreadPoolExecutor() as executor:
        # Process images in parallel
        futures = [executor.submit(read_and_process_image, image_path, i, image_paths) for i, image_path in enumerate(image_paths)]
        # Ensure images are written in the correct order
        for future in sorted(futures, key=lambda x: x.result()[0]):
            _, image = future.result()
            video_writer.write(image)
    # Add some pause frames to the end of the video
    for _ in range(end_pause_frames):
        video_writer.write(image)
    # Release the video writer
    video_writer.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-images_folder", type=str, required=True)
    ap.add_argument("-output_path", type=str, required=True)
    ap.add_argument("-fps", type=int, default=10)
    ap.add_argument(
        "-end_pause_frames",
        type=int,
        default=10,
        help="Number of pause frames to add to the end of the video",
    )
    ap.add_argument("-running_avg", type=int, default=0)
    args = vars(ap.parse_args())

    images_folder = Path(args["images_folder"])
    output_path = Path(args["output_path"])
    fps = args["fps"]
    end_pause_frames = args["end_pause_frames"]
    running_avg = args["running_avg"]
    create_timelapse(images_folder, output_path, fps)
