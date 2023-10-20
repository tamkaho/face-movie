# USAGE: python face-movie/main.py (-morph | -average) -images IMAGES [-tf TF] [-pf PF] [-fps FPS] -out OUT

from datetime import datetime, timedelta
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
import mediapipe as mp
from mediapipe.tasks import python as mpPython
from mediapipe.tasks.python import vision as mpVision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

########################################
# FACIAL LANDMARK DETECTION CODE
########################################

RIGHT_EYE_POINTS = [
    33,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
    133,
    155,
    154,
    153,
    145,
    144,
    163,
    7,
]
LEFT_EYE_POINTS = [
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
]

PREDICTOR = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    face_detector="blazeface",
    device="cpu",
)
EYE_FILE_NAME = Path("manual_eye_coords.json")
global align_eye_coords

########################################
# Mediapipe
########################################
# !wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# !curl -o face_landmarker_v2_with_blendshapes.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
base_options = mpPython.BaseOptions(
    model_asset_path="face_landmarker_v2_with_blendshapes.task"
)
options = mpVision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=3,
)
detector = mpVision.FaceLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


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

    image = mp.Image.create_from_file(str(fname))
    detection_result = detector.detect(image)
    if detection_result.face_landmarks == []:
        return None
    elif len(detection_result.face_landmarks) > 1:
        if str(TARGET.name) in align_eye_coords:
            # The eye should already be aligned to the target image
            manual_eye_coords = align_eye_coords[str(TARGET.name)]
            ratio = im.shape[1] / manual_eye_coords[2]
            manual_eye_coords = [
                np.array(manual_eye_coords[0]) * ratio,
                np.array(manual_eye_coords[1]) * ratio,
            ]
            tolerance = 1.5  # no. of standard deviation for acceptable eye positions
            for pred in detection_result.face_landmarks:
                landmarks = np.array(
                    [[l.x * im.shape[1], l.y * im.shape[0]] for l in pred]
                )
                if (
                    np.linalg.norm(
                        landmarks[LEFT_EYE_POINTS].mean(axis=0) - manual_eye_coords[0]
                    )
                    < np.linalg.norm(landmarks[LEFT_EYE_POINTS].std(axis=0)) * tolerance
                    and np.linalg.norm(
                        landmarks[RIGHT_EYE_POINTS].mean(axis=0) - manual_eye_coords[1]
                    )
                    < np.linalg.norm(landmarks[RIGHT_EYE_POINTS].std(axis=0))
                    * tolerance
                ) or (
                    np.linalg.norm(
                        landmarks[LEFT_EYE_POINTS].mean(axis=0) - manual_eye_coords[1]
                    )
                    < np.linalg.norm(landmarks[LEFT_EYE_POINTS].std(axis=0)) * tolerance
                    and np.linalg.norm(
                        landmarks[RIGHT_EYE_POINTS].mean(axis=0) - manual_eye_coords[0]
                    )
                    < np.linalg.norm(landmarks[RIGHT_EYE_POINTS].std(axis=0))
                    * tolerance
                ):
                    draw_mediapipe_landmarks(fname, image, detection_result)
                    return np.append(landmarks, get_boundary_points(im.shape), axis=0)
            return None  # Face Selection Not Impelemented
        else:
            return None
    else:
        landmarks = [
            [l.x * im.shape[1], l.y * im.shape[0]]
            for l in detection_result.face_landmarks[0]
        ]

    draw_mediapipe_landmarks(fname, image, detection_result)

    return np.append(landmarks, get_boundary_points(im.shape), axis=0)


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


def annotate_landmarks2(
    im: np.ndarray, landmarks1: np.ndarray, landmarks2: np.ndarray
) -> np.ndarray:
    im = im.copy()
    for idx, point in enumerate(landmarks1):
        pos = (int(point[0]), int(point[1]))
        cv2.putText(
            im,
            str(idx + 1),
            pos,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=0.4,
            color=(255, 255, 255),
        )
        cv2.circle(im, pos, 3, color=(255, 0, 0))

    for idx, point in enumerate(landmarks2):
        pos = (int(point[0]), int(point[1]))
        cv2.putText(
            im,
            str(idx + 1),
            pos,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=0.4,
            color=(255, 255, 255),
        )
        cv2.circle(im, pos, 3, color=(0, 255, 0))
    return im


def draw_mediapipe_landmarks(fname: Path, image: np.ndarray, detection_result):
    if not (Path("./mediapipe/") / fname.name).exists():
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imwrite(
            str(Path("./mediapipe/") / fname.name),
            cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        )


########################################
# MAIN DRIVER FUNCTIONS
########################################


def average_images(out_name: str):
    LANDMARK_LIST_ALL = [get_landmarks(impath) for impath in IM_FILES]
    LANDMARK_LIST = [x for x in LANDMARK_LIST_ALL if x is not None]
    avg_landmarks = sum(LANDMARK_LIST) / len(LANDMARK_LIST)
    triangulation = Delaunay(avg_landmarks).simplices

    warped_ims = [
        warp_im(
            np.float32(
                cv2.resize(
                    cv2.imread(str(IM_FILES[i]), cv2.IMREAD_COLOR),
                    None,
                    fx=RESIZE_FACTOR,
                    fy=RESIZE_FACTOR,
                )
            ),
            LANDMARK_LIST_ALL[i],
            avg_landmarks,
            triangulation,
        )
        for i in range(len(LANDMARK_LIST_ALL))
        if LANDMARK_LIST_ALL[i] is not None
    ]

    average = (1.0 / len(LANDMARK_LIST)) * sum(warped_ims)
    average = np.uint8(average)

    cv2.imwrite(out_name, average)


def morph_images(
    total_frames: int, fps: int, end_pause_frames: int, out_name: str
) -> None:
    first_im = cv2.resize(
        cv2.imread(str(IM_FILES[0]), cv2.IMREAD_COLOR),
        None,
        fx=RESIZE_FACTOR,
        fy=RESIZE_FACTOR,
    )
    if TXT_PREFIX:
        first_im = add_text_to_frame(first_im, 0)
    h = first_im.shape[0]
    w = first_im.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # or use 'XVID' for AVI format
    video_writer = cv2.VideoWriter(out_name, fourcc, fps, (w, h))
    video_writer.write(first_im)

    for i in range(len(IM_FILES) - 1):
        print("Morphing {} to {}".format(IM_FILES[i], IM_FILES[i + 1]))
        last_frame = morph_pair(i, i + 1, total_frames, out_name, video_writer)
        video_writer.write(last_frame)

    for _ in range(end_pause_frames):
        video_writer.write(last_frame)

    video_writer.release()


def morph_pair(
    idx1: int,
    idx2: int,
    total_frames: int,
    out_name: str,
    video_writer: cv2.VideoWriter,
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
            video_writer,
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
            video_writer,
            lambda arr: add_text_to_frame(arr, idx1) if TXT_PREFIX else arr,
        )
    return add_text_to_frame(im2, idx2) if TXT_PREFIX else im2


def cross_dissolve(
    total_frames: int,
    im1: np.ndarray,
    im2: np.ndarray,
    video_writer: cv2.VideoWriter,
    add_text,
) -> None:
    for j in range(total_frames):
        alpha = j / (total_frames - 1)
        blended = (1.0 - alpha) * im1 + alpha * im2
        blended = np.uint8(blended)
        im = add_text(blended)
        video_writer.write(im)


def running_avg_morph(day_step: int) -> None:
    outdir = Path(Path(OUTPUT_NAME).name)
    outdir.mkdir(parents=True, exist_ok=True)

    opened_images = []  # A list of images to cache
    opened_landmarks = []  # A list of landmarks to cache
    opened_dates = []

    first_date = time.strptime(IM_FILES[0].name[:8], "%Y%m%d")
    first_date = datetime(first_date.tm_year, first_date.tm_mon, first_date.tm_mday)
    curr_date = first_date - timedelta(days=RUNNING_AVG)
    last_date = time.strptime(IM_FILES[-1].name[:8], "%Y%m%d")
    last_date = datetime(last_date.tm_year, last_date.tm_mon, last_date.tm_mday)

    skipped = 0
    file_idx = 0
    next_file_date = first_date
    while curr_date < last_date + timedelta(days=RUNNING_AVG) or len(opened_images) > 1:
        # If all the output files already exist, skip
        # Because we are using a sliding window we need to check the future dates as well and start loading the images
        # if the future dates output images are not all present
        sliding_window_end_date = curr_date + timedelta(days=RUNNING_AVG * 2)
        sliding_window_start_date = curr_date - timedelta(days=RUNNING_AVG)
        skip = True
        while sliding_window_end_date > sliding_window_start_date:
            sliding_window_start_date += timedelta(days=day_step)
            if all(
                (
                    outdir
                    / "{}.jpg".format(
                        (curr_date + timedelta(days=n * day_step)).strftime(
                            "%Y%m%d_%H%M%S"
                        )
                    )
                ).exists()
                for n in range(int(3 * np.ceil(RUNNING_AVG / day_step) + 1))
            ):
                curr_date += timedelta(days=day_step)
                file_idx += day_step
                skipped = min(skipped + 1, 2 * (RUNNING_AVG / day_step))
                break
        else:
            skip = False

        if skip:
            continue

        while (
            file_idx < len(IM_FILES)
            and curr_date < last_date
            and next_file_date <= curr_date + timedelta(days=RUNNING_AVG)
        ):
            # Get date from the filename. Filename must start with YYYYMMDD
            impath = IM_FILES[int(file_idx)]
            prefix = impath.name[:8]
            next_file_date = time.strptime(prefix, "%Y%m%d")
            next_file_date = datetime(
                next_file_date.tm_year, next_file_date.tm_mon, next_file_date.tm_mday
            )

            if (next_file_date - curr_date).days <= RUNNING_AVG:
                latest_im = cv2.cvtColor(
                    cv2.resize(
                        cv2.imread(str(impath), cv2.IMREAD_COLOR),
                        None,
                        fx=RESIZE_FACTOR,
                        fy=RESIZE_FACTOR,
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                latest_landmarks = get_landmarks(impath)

                opened_images.append(latest_im)
                opened_landmarks.append(latest_landmarks)
                opened_dates.append(next_file_date)
                file_idx += 1

        # Remove images outside the running average window
        idx_to_remove = []
        for idx_date, date in enumerate(opened_dates):
            if (curr_date - date).days > RUNNING_AVG:
                idx_to_remove.append(idx_date)
            else:
                break
        idx_to_remove.sort(reverse=True)
        for idx in idx_to_remove:
            del opened_images[idx]
            del opened_landmarks[idx]
            del opened_dates[idx]

        if skipped > 0:
            skipped -= 1
            curr_date += timedelta(days=day_step)
            continue

        weights = np.array(
            [
                max(
                    0,
                    RUNNING_AVG
                    + 1
                    - abs((d - curr_date).total_seconds() / (24 * 3600)),
                )
                for i, d in enumerate(opened_dates)
                if opened_landmarks[i] is not None
            ]
        )
        weights = weights / weights.sum()

        valid_opened_landmarks = [x for x in opened_landmarks if x is not None]
        if len(valid_opened_landmarks) == 0:
            curr_date += timedelta(days=day_step)
            continue
        avg_landmarks = (
            np.array(valid_opened_landmarks)
            * weights.reshape((len(valid_opened_landmarks), 1, 1))
        ).sum(axis=0)
        triangulation = Delaunay(avg_landmarks).simplices
        warped_ims = [
            warp_im(
                np.float32(opened_images[i]),
                opened_landmarks[i],
                avg_landmarks,
                triangulation,
            )
            for i, landmark in enumerate(opened_landmarks)
            if landmark is not None
        ]

        # Debug
        if False:
            debug_dir = Path("running_avg_debug") / "{}".format(
                curr_date.strftime("%Y%m%d")
            )
            debug_dir.mkdir(exist_ok=True, parents=True)
            for j, warped_im in enumerate(warped_ims):
                cv2.imwrite(
                    str(debug_dir / "{}.jpg".format(j)),
                    cv2.cvtColor(warped_im.astype(np.uint8), cv2.COLOR_RGB2BGR),
                )
            for j, opened_image in enumerate(opened_images):
                if opened_landmarks[j] is not None:
                    cv2.imwrite(
                        str(debug_dir / "{}_landmarks.jpg".format(j)),
                        cv2.cvtColor(
                            annotate_landmarks2(
                                opened_image, opened_landmarks[j], avg_landmarks
                            ),
                            cv2.COLOR_RGB2BGR,
                        ),
                    )

        average = (
            np.array(warped_ims) * weights.reshape(len(warped_ims), 1, 1, 1)
        ).sum(axis=0)
        # average = (1.0 / len(opened_landmarks)) * sum(warped_ims)
        average = np.uint8(average)

        average = add_text_to_frame(
            average,
            min(
                (opened_dates[-1] - first_date).days,
                max(0, (curr_date - first_date).days),
            ),
        )
        cv2.imwrite(
            str(outdir / "{}.jpg".format(curr_date.strftime("%Y%m%d_%H%M%S"))),
            cv2.cvtColor(average, cv2.COLOR_RGB2BGR),
        )

        curr_date += timedelta(days=day_step)


def add_text_to_frame(img: np.ndarray, idx: int) -> np.ndarray:
    # Create an ImageDraw object
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    # Load the font
    font_paths = [
        Path("C:\\Windows\\Fonts\\DejaVuSans.ttf"),
        Path(cv2.__path__[0]) / "qt" / "fonts" / "DejaVuSans.ttf"
        if hasattr(cv2, "__path__")
        else None,
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
    y = img.height - text_height - TXT_DISTANCE_FROM_BOTTOM
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return np.array(img)


if __name__ == "__main__":
    start_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("-morph", help="Create morph sequence", action="store_true")
    ap.add_argument(
        "-running_avg",
        help="Number of days to average over in each direction. e.g. 5 means 5 days before and 5 days after",
        type=int,
        default=0,
    )
    ap.add_argument(
        "-day_step",
        help="Number of days step over for sliding window running average. Can be any positive float. Smaller=smoother",
        type=float,
        default=1.0,
    )
    ap.add_argument("-images", help="Directory of input images", required=True)
    ap.add_argument("-tf", type=int, help="Total frames for each image", default=2)
    ap.add_argument("-epf", type=int, help="End Pause frames", default=10)
    ap.add_argument("-fps", type=int, help="Frames per second", default=25)
    ap.add_argument("-out", help="Output file name", required=True)
    ap.add_argument(
        "-rs", help="Resize factor", required=False, default=1.0, type=float
    )
    ap.add_argument(
        "-text_prefix", help="Text prefix. e.g. Day X", type=str, default=""
    )
    ap.add_argument(
        "-txt_dist_bottom",
        help="Text distance from bottom of image",
        type=int,
        default=50,
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
    END_PAUSE_FRAMES = args["epf"]
    OUTPUT_NAME = args["out"]
    RESIZE_FACTOR = args["rs"]
    RUNNING_AVG = args["running_avg"]
    TXT_PREFIX = args["text_prefix"]
    TARGET = Path(args["target"])
    TXT_DISTANCE_FROM_BOTTOM = args["txt_dist_bottom"]
    DAY_STEP = args["day_step"]

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
        morph_images(TOTAL_FRAMES, FRAME_RATE, END_PAUSE_FRAMES, OUTPUT_NAME)
    elif MORPH:
        running_avg_morph(DAY_STEP)
    else:
        average_images(OUTPUT_NAME)

    elapsed_time = time.time() - start_time
    print(
        "Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    )
