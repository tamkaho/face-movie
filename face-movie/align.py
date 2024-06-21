# USAGE: python face-movie/align.py -images IMAGES -target TARGET [-overlay] [-border BORDER] -outdir OUTDIR
# e.g. python face-movie/align.py -images ../KaiHengTam_Timelapse/ -target ../KaiHengTam_Timelapse/20230629_184147.jpg -overlay -outdir ./aligned -manual

import cv2
import numpy as np
import argparse
from pathlib import Path
import json
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks import python as mpPython
from mediapipe.tasks.python import vision as mpVision
from deepface import DeepFace


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

ALIGN_EYES = [*LEFT_EYE_POINTS, *RIGHT_EYE_POINTS]

########################################
# Mediapipe
########################################
# !wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# !curl -o face_landmarker_v2_with_blendshapes.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
base_options = mpPython.BaseOptions(
    model_asset_path="face_landmarker_v2_with_blendshapes.task"
)
defaultFaceLandmarkerOptions = {
    "base_options": base_options,
    "output_face_blendshapes": True,
    "output_facial_transformation_matrixes": True,
    "num_faces": 3,
    "min_face_detection_confidence": 0.3,  # Default confidence
}
options = mpVision.FaceLandmarkerOptions(**defaultFaceLandmarkerOptions)
detector = mpVision.FaceLandmarker.create_from_options(options)


EYE_FILE_NAME = Path("manual_eye_coords.json")
Path("landmark").mkdir(exist_ok=True, parents=True)

cache = dict()
global align_eye_coords


def crop_face_with_deepface(fname: Path, im: np.ndarray) -> np.matrix | None:
    # If face is too small or there are multiple faces in the image, try using deepface to recognize the face.
    # Find the image closest to the current date in the aligned image folder
    curr_date = int(fname.stem[:6])
    db_image_path = sorted(
        Path(OUTPUT_DIR).glob("*.jpg"),
        key=lambda x: abs(int(x.stem[:6]) - curr_date),
        reverse=True,
    )[-1]

    try:
        result = DeepFace.verify(db_image_path, fname, model_name="Facenet")[
            "facial_areas"
        ]["img2"]
    except Exception as e:
        print("Error in DeepFace: ", fname.stem, e)
        return None

    # we get sth like {'x': 463, 'y': 189, 'w': 295, 'h': 295, 'left_eye': (599, 410), 'right_eye': (559, 301)} but eye is often None
    # So we make a tighter crop around the face and call mediapipe to get the landmarks

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(
            im[
                result["y"] : result["y"] + result["h"],
                result["x"] : result["x"] + result["w"],
            ],
            cv2.COLOR_BGR2RGB,
        ),
    )
    detection_result = detector.detect(image)

    if (
        detection_result.face_landmarks == []
        or len(detection_result.face_landmarks) > 1
    ):
        return None

    landmarks = [
        [l.x * result["w"] + result["x"], l.y * result["h"] + result["y"]]
        for l in detection_result.face_landmarks[0]
    ]

    return np.matrix(landmarks)


def get_landmarks(im: np.ndarray, fname: Path) -> np.matrix | None:
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    )
    detection_result = detector.detect(image)
    if detection_result.face_landmarks == []:
        print(
            "No face detected",
        )
        return crop_face_with_deepface(fname, im)
    elif len(detection_result.face_landmarks) > 1:
        if str(fname.name) in align_eye_coords:
            for pred in detection_result.face_landmarks:
                manual_eye_coords = align_eye_coords[str(fname.name)]
                landmarks = np.array(
                    [[l.x * im.shape[1], l.y * im.shape[0]] for l in pred]
                )
                if (
                    np.linalg.norm(
                        landmarks[LEFT_EYE_POINTS].mean(axis=0) - manual_eye_coords[0]
                    )
                    < np.linalg.norm(landmarks[LEFT_EYE_POINTS].std(axis=0))
                    and np.linalg.norm(
                        landmarks[RIGHT_EYE_POINTS].mean(axis=0) - manual_eye_coords[1]
                    )
                    < np.linalg.norm(landmarks[RIGHT_EYE_POINTS].std(axis=0))
                ) or (
                    np.linalg.norm(
                        landmarks[LEFT_EYE_POINTS].mean(axis=0) - manual_eye_coords[1]
                    )
                    < np.linalg.norm(landmarks[LEFT_EYE_POINTS].std(axis=0))
                    and np.linalg.norm(
                        landmarks[RIGHT_EYE_POINTS].mean(axis=0) - manual_eye_coords[0]
                    )
                    < np.linalg.norm(landmarks[RIGHT_EYE_POINTS].std(axis=0))
                ):
                    return np.matrix(landmarks)
            return None  # Face Selection Not Impelemented
        else:
            print(f"Detected {len(detection_result.face_landmarks)} faces")
            return crop_face_with_deepface(fname, im)
    else:
        landmarks = [
            [l.x * im.shape[1], l.y * im.shape[0]]
            for l in detection_result.face_landmarks[0]
        ]

    return np.matrix(landmarks)


def annotate_landmarks(im: np.array, landmarks: np.ndarray, fname: Path):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (int(point[0, 0]), int(point[0, 1]))
        cv2.putText(
            im,
            str(idx + 1),
            pos,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=1.5,
            color=(255, 255, 255),
        )
        cv2.circle(im, pos, 5, color=(255, 0, 0))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    im.thumbnail((900, 900), Image.LANCZOS)
    im.save(Path("landmark") / fname.name)


def magnify_area(
    im: np.ndarray, x: int, y: int, size: int = 50, zoom: int = 3
) -> np.ndarray:
    h, w = im.shape[:2]
    x1, y1 = max(0, x - size), max(0, y - size)
    x2, y2 = min(w, x + size), min(h, y + size)
    if x1 < x2 and y1 < y2:  # Ensure the slice is not empty
        magnified = cv2.resize(
            im[y1:y2, x1:x2], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR
        )
        mh, mw = magnified.shape[:2]
        # Draw a red crosshair at the center
        cv2.drawMarker(
            magnified,
            (mw // 2, mh // 2),
            color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )
        return magnified
    return np.zeros((size * 2 * zoom, size * 2 * zoom, 3), dtype=im.dtype)


def get_eye_coordinates(impath: Path) -> np.matrix:
    eye_coordinates = []
    im = read_im(impath)

    if str(impath.name) in align_eye_coords:
        eye_coordinates = align_eye_coords[str(impath.name)]
    else:
        mouse_position = [0, 0]

        def mouse_callback(event: int, x: int, y: int, flags: int, param):
            mouse_position[0], mouse_position[1] = x, y
            if event == cv2.EVENT_LBUTTONDOWN:
                print(
                    "Eye {}: ({}, {}) selected".format(len(eye_coordinates) + 1, x, y)
                )
                eye_coordinates.append((x, y))
                if len(eye_coordinates) == 2:
                    eye_coordinates.extend([im.shape[1], im.shape[0]])
                    cv2.setMouseCallback(
                        "Select Eyes", lambda *args: None
                    )  # Disable further callbacks
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)

        def draw_magnified_area(
            im: np.ndarray, x: int, y: int, size: int = 50, zoom: int = 3
        ):
            magnified = magnify_area(im, x, y, size, zoom)
            mh, mw = magnified.shape[:2]
            overlay = im.copy()
            x_offset, y_offset = min(x, im.shape[1] - mw), min(y, im.shape[0] - mh)
            overlay[y_offset : y_offset + mh, x_offset : x_offset + mw] = magnified
            return overlay

        print("Select eyes for " + str(impath.name))
        cv2.namedWindow("Select Eyes", cv2.WINDOW_NORMAL)
        cv2.imshow("Select Eyes", im)
        cv2.setMouseCallback("Select Eyes", mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 and len(eye_coordinates) == 4:  # Exit on ESC key
                break
            x, y = mouse_position
            if (
                0 <= x < im.shape[1] and 0 <= y < im.shape[0]
            ):  # Check if the cursor is within the image bounds
                overlay = draw_magnified_area(im, int(x), int(y))
                cv2.imshow("Select Eyes", overlay)

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        align_eye_coords[str(impath.name)] = eye_coordinates
        with open(EYE_FILE_NAME, "w", encoding="utf8") as outfile:
            json.dump(
                align_eye_coords,
                outfile,
                indent=4,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
            )

    resized_eye_coordinates = (
        np.matrix(eye_coordinates[:2]) * im.shape[1] / eye_coordinates[2]
    )
    return resized_eye_coordinates


def transformation_from_points(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    u, s, vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) whereas our solution requires the matrix to be on the
    # left (with column vectors).
    r = (u * vt).T
    if r[0, 0] < 0:
        r = np.dot(r, np.array([[-1, 0], [0, -1]]))
    if r[1, 1] < 0:
        r = np.dot(r, np.array([[1, 0], [0, -1]]))

    return np.vstack(
        [
            np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)),
            np.matrix([0.0, 0.0, 1.0]),
        ]
    )


def read_im_and_landmarks(fname: Path, landmark: bool) -> tuple[np.ndarray]:
    if fname in cache:
        return cache[fname]
    im = read_im(fname)
    s = get_landmarks(im, fname)
    if landmark and s.sum() > 0:
        annotate_landmarks(im, s, fname)

    cache[fname] = (im, s)
    return im, s


def warp_im(
    im: np.ndarray, m: np.ndarray, dshape: tuple, prev: np.ndarray = None
) -> np.ndarray:
    output_im = cv2.warpAffine(
        im,
        m,
        (dshape[1], dshape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101 if prev is not None else cv2.BORDER_CONSTANT,
    )

    if prev is not None:
        # overlay the image on the previous images
        mask = cv2.warpAffine(
            np.ones_like(im, dtype="float32"),
            m,
            (dshape[1], dshape[0]),
            flags=cv2.INTER_CUBIC,
        )
        output_im = mask * output_im + (1 - mask) * prev

    return output_im


def read_im(impath: Path) -> np.ndarray:
    im = Image.open(impath)
    im = ImageOps.exif_transpose(im)
    # noinspection PyTypeChecker
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    im = np.ascontiguousarray(im)
    return im


def align_images(
    impath1: Path,
    impath2: Path,
    border: int,
    manual_eye: bool,
    landmark: bool,
    prev=None,
) -> np.ndarray:
    filename = impath2.name
    outfile = OUTPUT_DIR / filename
    if not outfile.exists():
        manual_align_eye = manual_eye or (str(impath2.name) in align_eye_coords)
        if not manual_align_eye:  # If manual eye coords exist, use it.
            im1, landmarks1 = read_im_and_landmarks(impath1, landmark)
            try:
                im2, landmarks2 = read_im_and_landmarks(impath2, landmark)
                t = transformation_from_points(landmarks1, landmarks2)
            except:
                manual_align_eye = True

        if manual_align_eye:
            eye_coords1 = get_eye_coordinates(impath1)
            eye_coords2 = get_eye_coordinates(impath2)
            im1 = read_im(impath1)
            im2 = read_im(impath2)
            t = transformation_from_points(eye_coords1, eye_coords2)
        m = cv2.invertAffineTransform(t[:2])

        if border is not None:
            im2 = cv2.copyMakeBorder(
                im2,
                border,
                border,
                border,
                border,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

        warped_im2 = warp_im(im2, m, im1.shape, prev)

        # Apply CLAHE to aligned image
        # lab_img = cv2.cvtColor(warped_im2.astype(np.uint8), cv2.COLOR_RGB2LAB)
        # l_channel, a_channel, b_channel = cv2.split(lab_img)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # cl_l = clahe.apply(l_channel)
        # clab_img = cv2.merge((cl_l, a_channel, b_channel))
        # warped_im2 = cv2.cvtColor(clab_img, cv2.COLOR_LAB2RGB)

        cv2.imwrite(str(outfile), warped_im2)
        print("Aligned {}".format(filename))
        return warped_im2
    else:
        print("Skipping", outfile)
        return read_im(outfile)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-images", help="Directory of images to be aligned", required=True)
    ap.add_argument(
        "-target",
        help="Path to target image to which all others will be aligned",
        required=True,
    )
    ap.add_argument(
        "-overlay",
        help="Flag to overlay images on top of each other",
        action="store_true",
    )
    ap.add_argument(
        "-border", type=int, help="Border size (in pixels) to be added to images"
    )
    ap.add_argument("-outdir", help="Output directory name", required=True)
    ap.add_argument("-manual_eye", help="Manual Eye Coordinates", action="store_true")
    ap.add_argument("-landmark", help="Save landmarks", action="store_true")
    args = vars(ap.parse_args())
    im_dir = Path(args["images"])
    target = Path(args["target"])
    overlay = args["overlay"]
    use_border = args["border"]
    use_manual_eye = args["manual_eye"]
    use_landmark = args["landmark"]
    OUTPUT_DIR = Path(args["outdir"])

    valid_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    prevImg = None

    if EYE_FILE_NAME.exists():
        with open(EYE_FILE_NAME, "r") as file:
            align_eye_coords = json.load(file)
    else:
        align_eye_coords = dict()

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Constraints on input images (for aligning):
    # - Must have clear frontal view of a face (there may be multiple)
    # - Filenames must be in lexicographic order of the order in which they are to appear

    im_files = [f for f in im_dir.iterdir() if f.suffix in valid_formats]
    im_files = sorted(im_files, key=lambda x: x.name)

    # If the current and the next output already exist, skip it
    im_files = [
        f
        for idx, f in enumerate(im_files)
        if not (
            (OUTPUT_DIR / f.name).exists()
            and (
                idx == len(im_files) - 1
                or (OUTPUT_DIR / im_files[idx + 1].name).exists()
            )
        )
    ]

    for im_file in im_files:
        if overlay:
            prevImg = align_images(
                target, im_file, use_border, use_manual_eye, use_landmark, prevImg
            )
        else:
            align_images(target, im_file, use_border, use_landmark, use_manual_eye)
