# USAGE: python face-movie/main.py (-morph | -average) -images IMAGES [-tf TF] [-pf PF] [-fps FPS] -out OUT

from scipy.spatial import Delaunay
from PIL import Image, ImageFont, ImageDraw
from face_morph import morph_seq, warp_im
from subprocess import Popen, PIPE
import argparse
import numpy as np
import face_alignment
from pathlib import Path
import cv2
import time
import json

########################################
# FACIAL LANDMARK DETECTION CODE
########################################

RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))

PREDICTOR = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    face_detector="blazeface",
    device="cpu",
)
EYE_FILE_NAME = Path("manual_eye_coords.json")
global align_eye_coords


def get_boundary_points(shape: np.ndarray) -> np.ndarray:
    h, w = shape[:2]
    boundary_pts = [
        (1, 1),
        (w - 1, 1),
        (1, h - 1),
        (w - 1, h - 1),
        ((w - 1) // 2, 1),
        (1, (h - 1) // 2),
        ((w - 1) // 2, h - 1),
        ((w - 1) // 2, (h - 1) // 2),
    ]
    return np.array(boundary_pts)


def get_landmarks(fname: Path) -> np.ndarray | None:
    im = cv2.resize(
        cv2.imread(str(fname), cv2.IMREAD_COLOR),
        None,
        fx=RESIZE_FACTOR,
        fy=RESIZE_FACTOR,
    )
    preds = PREDICTOR.get_landmarks(im)
    if preds is None:
        raise Exception("No Faces Found")
    if len(preds) > 1:
        if str(TARGET.name) in align_eye_coords:
            # The eye should already be aligned to the target image
            manual_eye_coords = align_eye_coords[str(TARGET.name)]
            ratio = im.shape[1] / manual_eye_coords[2]
            manual_eye_coords = [manual_eye_coords[0], manual_eye_coords[1]]
            tolerance = 1.5  # no. of standard deviation for acceptable eye positions
            for pred in preds:
                if (
                    np.linalg.norm(
                        pred[LEFT_EYE_POINTS].mean(axis=0) - manual_eye_coords[0]
                    )
                    < np.linalg.norm(pred[LEFT_EYE_POINTS].std(axis=0)) * tolerance
                    and np.linalg.norm(
                        pred[RIGHT_EYE_POINTS].mean(axis=0) - manual_eye_coords[1]
                    )
                    < np.linalg.norm(pred[RIGHT_EYE_POINTS].std(axis=0)) * tolerance
                ) or (
                    np.linalg.norm(
                        pred[LEFT_EYE_POINTS].mean(axis=0) - manual_eye_coords[1]
                    )
                    < np.linalg.norm(pred[LEFT_EYE_POINTS].std(axis=0)) * tolerance
                    and np.linalg.norm(
                        pred[RIGHT_EYE_POINTS].mean(axis=0) - manual_eye_coords[0]
                    )
                    < np.linalg.norm(pred[RIGHT_EYE_POINTS].std(axis=0)) * tolerance
                ):
                    return np.append(pred, get_boundary_points(im.shape), axis=0)
        return None  # Face Selection Not Impelemented
    return np.append(preds[0], get_boundary_points(im.shape), axis=0)


########################################
# VISUALIZATION CODE FOR DEBUGGING
########################################


def draw_triangulation(
    im: np.ndarray, landmarks: np.ndarray, triangulation: np.ndarray
) -> None:
    import matplotlib.pyplot as plt

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.triplot(
        landmarks[:, 0], landmarks[:, 1], triangulation, color="blue", linewidth=1
    )
    plt.axis("off")
    plt.show()


def annotate_landmarks(im: np.ndarray, landmarks: np.ndarray) -> None:
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.putText(
            im,
            str(idx + 1),
            pos,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=0.4,
            color=(255, 255, 255),
        )
        cv2.circle(im, pos, 3, color=(255, 0, 0))
    cv2.imwrite("landmarks.jpg", im)


########################################
# MAIN DRIVER FUNCTIONS
########################################


def average_images(out_name: str):
    # avg_landmarks = sum(LANDMARK_LIST) / len(LANDMARK_LIST)
    # triangulation = Delaunay(avg_landmarks).simplices

    # warped_ims = [
    #     warp_im(
    #         np.float32(
    #             cv2.resize(
    #                 cv2.imread(str(IM_FILES[i]), cv2.IMREAD_COLOR),
    #                 None,
    #                 fx=RESIZE_FACTOR,
    #                 fy=RESIZE_FACTOR,
    #             )
    #         ),
    #         LANDMARK_LIST[i],
    #         avg_landmarks,
    #         triangulation,
    #     )
    #     for i in range(len(LANDMARK_LIST))
    # ]

    # average = (1.0 / len(LANDMARK_LIST)) * sum(warped_ims)
    # average = np.uint8(average)

    # cv2.imwrite(out_name, average)
    pass


def morph_images(total_frames: int, fps: int, pause_frames: int, out_name: str) -> None:
    first_im = cv2.cvtColor(
        cv2.resize(
            cv2.imread(str(IM_FILES[0]), cv2.IMREAD_COLOR),
            None,
            fx=RESIZE_FACTOR,
            fy=RESIZE_FACTOR,
        ),
        cv2.COLOR_BGR2RGB,
    )
    if TXT_PREFIX:
        first_im = add_text_to_frame(first_im, 0)
    h = max(first_im.shape[:2])
    w = min(first_im.shape[:2])

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "image2pipe",
        "-r",
        str(fps),
        "-s",
        str(h) + "x" + str(w),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        out_name,
    ]

    p = Popen(command, stdin=PIPE)

    fill_frames(Image.fromarray(first_im), pause_frames, p)

    for i in range(len(IM_FILES) - 1):
        print("Morphing {} to {}".format(IM_FILES[i], IM_FILES[i + 1]))
        last_frame = morph_pair(i, i + 1, total_frames, fps, out_name, p)
        fill_frames(last_frame, pause_frames, p)

    p.stdin.close()
    p.wait()


def morph_pair(
    idx1: int,
    idx2: int,
    total_frames: int,
    fps: int,
    out_name: str,
    stream: Popen[bytes],
) -> Image:
    """
    For a pair of images, produce a morph sequence with the given duration
    and fps to be written to the provided output stream.
    """
    im1 = cv2.resize(
        cv2.imread(str(IM_FILES[idx1]), cv2.IMREAD_COLOR),
        None,
        fx=RESIZE_FACTOR,
        fy=RESIZE_FACTOR,
    )
    im2 = cv2.resize(
        cv2.imread(str(IM_FILES[idx2]), cv2.IMREAD_COLOR),
        None,
        fx=RESIZE_FACTOR,
        fy=RESIZE_FACTOR,
    )

    im1_landmarks = get_landmarks(IM_FILES[idx1])
    im2_landmarks = get_landmarks(IM_FILES[idx2])

    if im1_landmarks is None or im2_landmarks is None:
        print(
            "No faces found, performing cross-dissolve between {} and {}".format(
                IM_FILES[idx1], IM_FILES[idx2]
            )
        )
        cross_dissolve(
            total_frames,
            im1,
            im2,
            stream,
            lambda arr: add_text_to_frame(arr, idx1) if TXT_PREFIX else arr,
        )

    else:
        average_landmarks = (im1_landmarks + im2_landmarks) / 2

        triangulation = Delaunay(average_landmarks).simplices
        # draw_triangulation(im2, average_landmarks, triangulation)

        h, w = im1.shape[:2]
        morph_seq(
            total_frames,
            im1,
            im2,
            im1_landmarks,
            im2_landmarks,
            triangulation.tolist(),
            (w, h),
            out_name,
            stream,
            lambda arr: add_text_to_frame(arr, idx1) if TXT_PREFIX else arr,
        )
    return Image.fromarray(
        add_text_to_frame(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB), idx2)
        if TXT_PREFIX
        else im2
    )


# TODO: less janky way of filling frames?
def fill_frames(im: Image, num, p: Popen[bytes]) -> None:
    for _ in range(num):
        im.save(p.stdin, "JPEG")


def cross_dissolve(
    total_frames: int, im1: np.ndarray, im2: np.ndarray, p, add_text
) -> None:
    for j in range(total_frames):
        alpha = j / (total_frames - 1)
        blended = (1.0 - alpha) * im1 + alpha * im2
        blended = cv2.cvtColor(np.uint8(blended), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(add_text(blended))
        im.save(p.stdin, "JPEG")


def running_avg_morph() -> None:  # Todo: running average morph
    # first_im = cv2.cvtColor(cv2.resize(cv2.imread(str(IM_FILES[0]), cv2.IMREAD_COLOR), None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR), cv2.COLOR_BGR2RGB)
    # h = max(first_im.shape[:2])
    # w = min(first_im.shape[:2])

    # outdir = Path(Path(OUTPUT_NAME).name)
    # outdir.mkdir(parents=True, exist_ok=True)

    # opened_images = [] # A list of images to cache
    # opened_landmarks = [] # A list of landmarks to cache
    # for i, imname in enumerate(IM_FILES):
    #     opened_images.append()
    #     pass
    pass


def add_text_to_frame(img: np.ndarray, idx: int) -> np.ndarray:
    # Create an ImageDraw object
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    # Load the font
    font_paths = [
        Path("C:\\Windows\\Fonts\\DejaVuSans.ttf"),
        Path(cv2.__path__[0]) / "qt" / "fonts" / "DejaVuSans.ttf",
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]

    font_path = next((path for path in font_paths if path.exists()), None)
    font = ImageFont.truetype(str(font_path), size=int(min(img.size[:2]) * 0.084))

    # Add the text at the bottom of the image
    text = f"{TXT_PREFIX} {idx}"
    text_bbox = draw.textbbox((0, 0), text, font=font)

    # Calculate text width and height
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Draw the text on the image
    x = (img.width - text_width) // 2
    y = img.height - text_height - 50
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return np.array(img)


if __name__ == "__main__":
    start_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("-morph", help="Create morph sequence", action="store_true")
    ap.add_argument("-running_avg", type=int, default=0)
    ap.add_argument("-images", help="Directory of input images", required=True)
    ap.add_argument("-tf", type=int, help="Total frames for each image", default=2)
    ap.add_argument("-pf", type=int, help="Pause frames", default=1)
    ap.add_argument("-fps", type=int, help="Frames per second", default=25)
    ap.add_argument("-out", help="Output file name", required=True)
    ap.add_argument(
        "-rs", help="Resize factor", required=False, default=1.0, type=float
    )
    ap.add_argument(
        "-text_prefix", help="Text prefix. e.g. Day X", type=str, default=""
    )
    ap.add_argument(
        "-target",
        help="Path to target image to which all others will be aligned",
        required=True,
    )
    args = vars(ap.parse_args())

    MORPH = args["morph"]
    IM_DIR = Path(args["images"])
    FRAME_RATE = args["fps"]
    TOTAL_FRAMES = args["tf"]
    PAUSE_FRAMES = args["pf"]
    OUTPUT_NAME = args["out"]
    RESIZE_FACTOR = args["rs"]
    RUNNING_AVG = args["running_avg"]
    TXT_PREFIX = args["text_prefix"]
    TARGET = Path(args["target"])

    valid_formats = [".jpg", ".jpeg", ".png"]

    # Constraints on input images (for morphing):
    # - Must all have same dimension
    # - Must have clear frontal view of a face (there may be multiple)
    # - Filenames must be in lexicographic order of the order in which they are to appear

    if EYE_FILE_NAME.exists():
        with open(EYE_FILE_NAME, "r") as file:
            align_eye_coords = json.load(file)
    else:
        align_eye_coords = dict()

    IM_FILES = [f for f in IM_DIR.iterdir() if f.suffix in valid_formats]
    IM_FILES = sorted(IM_FILES, key=lambda x: x.name)

    assert len(IM_FILES) > 0, "No valid images found in {}".format(IM_DIR)

    if MORPH and RUNNING_AVG == 0:
        morph_images(TOTAL_FRAMES, FRAME_RATE, PAUSE_FRAMES, OUTPUT_NAME)
    elif MORPH:
        running_avg_morph()
    else:
        average_images(OUTPUT_NAME)

    elapsed_time = time.time() - start_time
    print(
        "Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    )
