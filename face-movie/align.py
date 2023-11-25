# USAGE: python face-movie/align.py -images IMAGES -target TARGET [-overlay] [-border BORDER] -outdir OUTDIR
# e.g. python face-movie/align.py -images ../KaiHengTam_Timelapse/ -target ../KaiHengTam_Timelapse/20230629_184147.jpg -overlay -outdir ./aligned -manual

import cv2
import face_alignment
import numpy as np
import argparse
from pathlib import Path
import json
from PIL import Image, ImageOps

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (
    LEFT_BROW_POINTS
    + RIGHT_EYE_POINTS
    + LEFT_EYE_POINTS
    + RIGHT_BROW_POINTS
    + NOSE_POINTS
    + MOUTH_POINTS
)
ALIGN_EYES = [LEFT_EYE_POINTS[0]] + [RIGHT_EYE_POINTS[0]]

# PREDICTOR = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, face_detector="dlib", device="cpu")
# PREDICTOR = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, face_detector="sfd", device="cpu")
PREDICTOR = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    face_detector="blazeface",
    device="cpu",
)
# PREDICTOR = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, face_detector="blazeface", face_detector_kwargs={'back_model': True}, device="cpu")


EYE_FILE_NAME = Path("manual_eye_coords.json")
Path("landmark").mkdir(exist_ok=True, parents=True)

cache = dict()
global align_eye_coords


def get_landmarks(im: np.ndarray, fname: Path) -> np.matrix:
    preds = PREDICTOR.get_landmarks(im)
    if preds is None:
        raise Exception("No Faces Found")
    if len(preds) > 1:
        if str(fname.name) in align_eye_coords:
            for pred in preds:
                manual_eye_coords = align_eye_coords[str(fname.name)]
                if (
                    np.linalg.norm(
                        pred[LEFT_EYE_POINTS].mean(axis=0) - manual_eye_coords[0]
                    )
                    < np.linalg.norm(pred[LEFT_EYE_POINTS].std(axis=0))
                    and np.linalg.norm(
                        pred[RIGHT_EYE_POINTS].mean(axis=0) - manual_eye_coords[1]
                    )
                    < np.linalg.norm(pred[RIGHT_EYE_POINTS].std(axis=0))
                ) or (
                    np.linalg.norm(
                        pred[LEFT_EYE_POINTS].mean(axis=0) - manual_eye_coords[1]
                    )
                    < np.linalg.norm(pred[LEFT_EYE_POINTS].std(axis=0))
                    and np.linalg.norm(
                        pred[RIGHT_EYE_POINTS].mean(axis=0) - manual_eye_coords[0]
                    )
                    < np.linalg.norm(pred[RIGHT_EYE_POINTS].std(axis=0))
                ):
                    return np.matrix(pred)
        raise Exception("Face Selection Not Impelemented")
    return np.matrix(preds[0])


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


def get_eye_coordinates(impath: Path) -> np.matrix:
    eye_coordinates = []
    im = read_im(impath)

    if str(impath.name) in align_eye_coords:
        eye_coordinates = align_eye_coords[str(impath.name)]
    else:

        def mouse_callback(event: int, x: int, y: int, flags: int, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(
                    "Eye {}: ({}, {}) selected".format(len(eye_coordinates) + 1, x, y)
                )
                eye_coordinates.append(
                    (x, y)
                )  # Recording the h and w in case we resize the image at some point.
                if len(eye_coordinates) == 2:
                    eye_coordinates.extend([im.shape[1], im.shape[0]])
                    cv2.destroyAllWindows()

        print("Select eyes for " + impath.name)
        cv2.namedWindow("Select Eyes", cv2.WINDOW_NORMAL)
        cv2.imshow("Select Eyes", im)
        cv2.setMouseCallback("Select Eyes", mouse_callback)
        cv2.waitKey(0)
        align_eye_coords[str(impath.name)] = eye_coordinates
        with open(EYE_FILE_NAME, "w", encoding="utf8") as outfile:
            str_ = json.dumps(
                align_eye_coords,
                indent=4,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
            )
            outfile.write(str(str_))
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


def read_im_and_landmarks(fname: Path, landmark: bool) -> (np.ndarray, np.ndarray):
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
                t = transformation_from_points(
                    landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS]
                )
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
