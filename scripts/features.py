# it's all about the features
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import imageio.v3 as iio
from scipy import ndimage
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import face_recognition
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ != "__main__":
    from scripts import utils as ut
else:
    import utils as ut


# Feature Selector for sk-learn-pipelines -------------------------------------
def feature_selector(df, features_to_keep):
    return df[features_to_keep]


# --- here are the scan_path features calculated for a given file -------------
def calculate_sp_features(sp_file: str) -> pd.DataFrame:
    """calculate SCAN_PATH features for *.txt file

    Args:
        sp_file (str): path to scanpath_*.txt

    Returns:
        DatFrame: pd.DatFrame containing calculated features
    """
    # instantiate df
    df = None

    # get size of image
    img_file = ut.get_img_of_sp(sp_file)
    image_size = iio.imread(img_file).shape

    # loop scanpaths
    sps = ut.load_scanpath(sp_file)
    for sp_i, sp in enumerate(sps):
        # id
        id = ut.get_sp_id(sp_file, sp_i)
        df_sp = pd.DataFrame(pd.Series(id), columns=["id"])

        # instance identifiers
        df_sp["asd"] = 1 if "ASD" in sp_file.split("/")[-2] else 0
        df_sp["img"] = int(sp_file.split("_")[-1].split(".")[0])
        df_sp["sp_idx"] = sp_i

        # fix count
        df_sp["sp_fix_count"] = len(sp)

        # total, mean and variance of duration
        df_sp["sp_fix_duration_ms_total"] = np.sum(sp["duration"])
        df_sp["sp_fix_duration_ms_mean"] = np.mean(sp["duration"])
        df_sp["sp_fix_duration_ms_var"] = np.var(sp["duration"])

        # total, mean and variance of saccade amplitude
        differences = np.diff(sp[["x", "y"]].values, axis=0)
        amplitudes = np.linalg.norm(differences, axis=1)

        df_sp["sp_len_px_total"] = np.sum(amplitudes)
        df_sp["sp_saccade_amplitude_px_mean"] = (
            np.mean(amplitudes) if len(amplitudes) else 0.0
        )
        df_sp["sp_saccade_amplitude_px_var"] = (
            np.var(amplitudes) if len(amplitudes) else 0.0
        )

        # mean and variance of fixation distance to centre
        dist_2_centre = np.sqrt(
            (sp["x"] - image_size[1] / 2) ** 2 + (sp["y"] - image_size[0] / 2) ** 2
        )

        df_sp["sp_distance_to_centre_px_mean"] = np.mean(dist_2_centre)
        df_sp["sp_distance_to_centre_px_var"] = np.var(dist_2_centre)

        # mean and median fixation distance to mean fixation
        mean_x, mean_y = sp["x"].mean(), sp["y"].mean()
        dist_to_mean = np.sqrt((sp["x"] - mean_x) ** 2 + (sp["y"] - mean_y) ** 2)

        df_sp["sp_distance_to_sp_mean_px_mean"] = np.mean(dist_to_mean)
        df_sp["sp_distance_to_sp_mean_px_var"] = np.var(dist_to_mean)

        # concat to df
        df = pd.concat([df, df_sp], ignore_index=True)

    return df


# --- here are the SALIENCY features calculated for a given sp-file -----------
def calculate_saliency_features(sp_file: str, mdl: str = "sam_resnet") -> pd.DataFrame:
    """calculate SALIENCY features for *.txt file

    Args:
        sp_file (str): path to scanpath_*.txt

    Returns:
        DatFrame: pd.DatFrame containing calculated features
    """
    # instantiate df
    df = None

    # load SALIENCY PREDICTION map
    loaded_sal_map = ut.load_saliency_map(sp_file, mdl)
    image_size = loaded_sal_map.shape

    # loop scanpaths
    sps = ut.load_scanpath(sp_file)
    for sp_i, sp in enumerate(sps):
        # id
        id = ut.get_sp_id(sp_file, sp_i)
        df_sal = pd.DataFrame(pd.Series(id), columns=["id"])

        # scanpath -> empirical saliency_map
        empirical_fix_map = np.zeros(image_size)
        empirical_fix_map[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = 1
        ndimage.gaussian_filter(empirical_fix_map, sigma=40, output=empirical_fix_map)

        # copy loaded salience map
        sal_map = loaded_sal_map.copy()
        sal_values = sal_map[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)]

        # features: fixation
        df_sal["sal_first_fixation"] = sal_values[0]

        max_val = sal_map.max()
        for max_share in [0.75, 0.9]:
            try:
                first_reaching = np.where(sal_values >= max_val * max_share)[0][0] + 1
            except IndexError:
                first_reaching = max(20, len(sal_values) + 1)
            df_sal["sal_first_above_{}*max_rank".format(max_share)] = first_reaching

        # features: saliency
        df_sal["sal_mean"] = np.mean(sal_values)
        df_sal["sal_sum"] = np.sum(sal_values)
        df_sal["sal_max"] = np.max(sal_values)

        # features: duration-by-saliency
        df_sal["sal_weighted_duration_sum"] = np.sum(sp["duration"] * sal_values)
        df_sal["sal_weighted_duration_mean"] = np.mean(sp["duration"] * sal_values)

        # features: normalisation for KLD
        empirical_fix_map -= empirical_fix_map.min()
        empirical_fix_map /= empirical_fix_map.sum()
        sal_map -= sal_map.min()
        if sal_map.sum() == 0:
            sal_map = np.ones_like(sal_map)
        sal_map /= sal_map.sum()

        eps = np.finfo(sal_map.dtype).eps
        df_sal["sal_KLD"] = (
            empirical_fix_map * np.log(empirical_fix_map / (sal_map + eps) + eps)
        ).sum()

        # features: normalisation for NSS
        sal_map -= sal_map.mean()
        if sal_map.std() != 0:
            sal_map /= sal_map.std()

        df_sal["sal_NSS"] = np.mean(
            sal_map[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)]
        )

        # concat to df
        df = pd.concat([df, df_sal], ignore_index=True)

    return df


# --- object detection features based on SCANPATH_*.txt files and images ------
def get_object_detector_object():
    curdir = os.path.dirname(__file__)
    mdl_pth = os.path.join(
        curdir, "..", "models", "mediapipe", "efficientdet_lite0.tflite"
    )

    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=mdl_pth),
        score_threshold=0.2,
        max_results=10,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.ObjectDetector.create_from_options(options)
    return detector


def intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calculate intersection coordinates
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Check if there is an intersection
    if x_right >= x_left and y_bottom >= y_top:
        return True
    else:
        return False


def is_animate(x):
    animate = [
        "person",
        "deer",
        "cat",
        "alligator",
        "bear",
        "dog",
        "elk",
        "bird",
        "rodent",
        "horse",
        "tiger",
        "fish",
        "sealion",
        "lion",
        "insect",
        "giraffe",
        "squirrel",
        "ape",
        "leopard",
        "butterfly",
        "gorilla",
    ]
    return True if x in animate else False


def process_detections(detections, img_file, output, path_obj_recog):
    ignore = {
        4: [1, 2, 3, 4, 6, 7],
        10: [3, 4, 5, 6],
        11: [0, 1],
        13: [1],
        16: [1, 2, 3],
        17: [6],
        19: [2, 3],
        23: list(range(3, 10)),
        24: [1],
        27: [3],
        28: [2, 3],
        33: [2, 3, 4, 5, 6, 7, 8],
        34: [1],
        36: [1],
        38: [1],
        39: list(range(1, 5)),
        42: [4],
        44: [8],
        46: [7, 8, 9],
        51: [2],
        52: [0],
        53: [3],
        54: [2],
        55: [0, 1],
        63: [7],
        64: [1, 3],
        65: [2, 3],
        66: [0, 1],
        67: [2, 3],
        68: [2, 4, 9],
        70: [1, 2],
        72: [7],
        73: [0],
        74: [5, 7],
        76: [1, 2, 3],
        77: [0],
        79: [0],
        81: [0],
        83: [2, 3],
        84: [0, 3, 4, 5, 7, 8],
        87: [1],
        89: [2, 3],
        98: [1],
        102: [4],
        108: [2, 3],
        111: [2],
        112: [0],
        114: [1],
        115: [6],
        116: [4, 7],
        119: [1],
        123: [2],
        125: [0, 1, 2, 3, 4],
        127: [7],
        130: [6],
        131: [0],
        132: [1],
        134: [3, 6],
        135: [1],
        136: [1],
        137: [2],
        138: [4],
        139: [4, 6],
        141: [3, 6],
        143: [0],
        148: [1, 2],
        149: [6, 7],
        150: [4, 5],
        151: [2, 4, 5, 6],
        153: [0],
        154: [2],
        160: [5, 6, 7, 9],
        161: [2],
        162: [1],
        163: [5],
        167: [9],
        169: [2, 3],
        171: [2],
        172: [7, 8, 9],
        173: [2],
        175: [0],
        177: [2],
        183: [0, 4],
        186: [1],
        188: list(range(2, 16)),
        192: [7],
        198: [0],
        200: [7],
        201: [0],
        202: [3, 4, 5, 6],
        205: [4],
        206: [1],
        207: [3],
        208: [1],
        209: [4, 5, 7, 8, 9],
        210: [8],
        211: [0, 2],
        214: [7],
        215: [4, 9],
        217: [1],
        220: [1],
        224: [1],
        225: list(range(1, 6)),
        226: [1],
        228: [4],
        229: [1],
        231: [7, 9],
        232: [1],
        234: [5, 6, 7, 8, 9],
        237: [4],
        239: [1],
        241: [1, 2, 3, 4],
        242: [3],
        246: [1],
        247: [0, 1, 2],
        250: [0, 3, 4, 5],
        252: [1, 2, 3],
        254: [3],
        255: [1],
        256: [1],
        257: [5, 6],
        258: [0],
        259: [0, 5],
        261: [1],
        265: [8],
        266: [1, 2, 3],
        271: [1, 2, 3],
        273: [1, 3],
        274: [6],
        281: [4],
        282: [1, 2],
        284: [2, 3, 4, 6, 7, 8],
        285: [1, 4, 5, 6, 8],
        287: [1, 4],
        290: [3, 5],
        293: list(range(7)),
        295: [2],
        297: [6],
    }
    switch = {
        2: {1: "guitar"},
        3: {1: "guitar"},
        4: {0: "icecream"},
        8: {0: "leopard"},
        13: {0: "rodent"},
        15: {0: "tiger"},
        17: {4: "sealion"},
        18: {0: "lion"},
        22: {0: "deer"},
        23: {2: "fan"},
        26: {0: "cat"},
        27: {4: "orange"},
        29: {5: "handbag"},
        36: {0: "statue"},
        40: {0: "deer", 1: "deer", 2: "deer"},
        44: {7: "bicycle"},
        53: {2: "guitar"},
        67: {1: "airplane"},
        74: {8: "bag"},
        77: {2: "meat", 3: "meat", 5: "bread"},
        80: {0: "machine"},
        84: {1: "cat", 2: "cat", 6: "cat"},
        86: {0: "lion", 1: "lion"},
        91: {0: "dummy"},
        93: {2: "book"},
        94: {0: "butterfly"},
        96: {
            0: "gorilla",
            1: "gorilla",
            2: "gorilla",
            3: "plant",
            4: "gorilla",
            5: "gorilla",
        },
        97: {0: "squirrel"},
        98: {0: "deer"},
        100: {0: "deer"},
        101: {3: "statue"},
        106: {0: "toy"},
        109: {1: "trophy"},
        110: {0: "alligator"},
        111: {1: "book"},
        116: {3: "table", 5: "paper", 6: "couch"},
        117: {2: "toy", 3: "person", 4: "toy"},
        118: {1: "toy"},
        119: {3: "person"},
        124: {0: "elk"},
        135: {0: "rodent"},
        137: {1: "statue"},
        138: {3: "can"},
        146: {7: "art"},
        153: {1: "lock"},
        155: {0: "alligator"},
        156: {0: "fish", 1: "fish", 2: "fish", 3: "fish"},
        159: {0: "bird", 5: "bird"},
        167: {4: "cup"},
        169: {0: "balloon"},
        170: {0: "plant"},
        172: {6: "bag"},
        179: {0: "food"},
        181: {0: "insect"},
        188: {1: "sign"},
        202: {1: "glass"},
        205: {3: "sign"},
        208: {0: "car"},
        210: {9: "instrument"},
        212: {1: "cup"},
        219: {0: "fish"},
        220: {0: "fish"},
        223: {2: "hat", 3: "bottle"},
        242: {4: "bowl"},
        249: {0: "squirrel"},
        252: {4: "book"},
        259: {6: "bag"},
        263: {0: "ape"},
        267: {8: "plate"},
        272: {3: "person", 7: "person"},
        273: {2: "table"},
        287: {3: "suitcase", 6: "box "},
        288: {0: "statue"},
        290: {4: "hat"},
        294: {4: "tie"},
    }

    # current image number
    img_nr = int(img_file.split("/")[-1].split(".")[0])

    # handle RENAMINGS of detections
    if img_nr in switch.keys():
        for i in switch[img_nr].keys():
            detections.detections[i].categories[0].category_name = switch[img_nr][i]

    # handle DELETIONS of detections
    if img_nr in ignore.keys():
        to_delete = ignore[img_nr]
        detections.detections = [
            detections.detections[i]
            for i in range(len(detections.detections))
            if i not in to_delete
        ]

    if output:
        path_scores = os.path.join(path_obj_recog, "scores")
        if not os.path.exists(path_scores):
            os.makedirs(path_scores)

        # file to write
        ftw = os.path.join(path_scores, f"{img_nr}_scores.txt")

        # delete previous file
        if os.path.exists(ftw):
            os.remove(ftw)

        # write scores of detected objects
        with open(ftw, "a+") as file:
            file.write("\nobj_id obj_name obj_score")
        for obj_id, detection in enumerate(detections.detections):
            obj_name = detection.categories[0].category_name
            obj_score = detection.categories[0].score
            with open(ftw, "a+") as file:
                file.write(f"\n{obj_id} {obj_name} {round(obj_score*100, 3)}")

    return detections


def draw_objects_faces(fn, ax, face_locations, detection_result, labels: bool = True):
    plt.figure(fn)
    # add faces
    for fl in face_locations:
        top, right, bottom, left = fl
        rect = patches.Rectangle(
            (left, top),
            right - left,
            bottom - top,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    # add objects
    for _, detection in enumerate(detection_result.detections):
        # add rectangle
        rect = patches.Rectangle(
            (
                detection.bounding_box.origin_x,
                detection.bounding_box.origin_y,
            ),
            detection.bounding_box.width,
            detection.bounding_box.height,
            linewidth=2,
            edgecolor="orange",
            facecolor="none",
        )
        ax.add_patch(rect)

        # name & score
        if labels:
            obj_name = detection.categories[0].category_name
            obj_score = detection.categories[0].score

            # add label
            txt_name = f"{_}: {obj_name}"
            txt_score = f"{obj_score*100:.2f}%"
            plt.text(
                detection.bounding_box.origin_x,
                detection.bounding_box.origin_y,
                txt_name,
                fontsize=6,
                backgroundcolor="orange",
                verticalalignment="top",
            )
            plt.text(
                detection.bounding_box.origin_x + detection.bounding_box.width,
                detection.bounding_box.origin_y + detection.bounding_box.height,
                txt_score,
                fontsize=6,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox={"alpha": 0.5, "color": "w"},
            )


def draw_scanpath(fn, sp):
    plt.figure(fn)
    # add saccades
    for r in range(1, sp.shape[0]):
        plt.plot(
            [sp.loc[sp["idx"] == r - 1]["x"], sp.loc[sp["idx"] == r]["x"]],
            [sp.loc[sp["idx"] == r - 1]["y"], sp.loc[sp["idx"] == r]["y"]],
            lw=4,
            c="#2c94ea",
        )

    # add fixations for individual plot
    s_min = 10
    s_max = 50
    if (sp["duration"].shape[0] == 1) or (sp["duration"].std() == 0):
        sp["size"] = (s_max + s_min) / 2
    else:
        sp["size"] = (sp["duration"] - np.min(sp["duration"])) / (
            np.max(sp["duration"]) - np.min(sp["duration"])
        ) * (s_max - s_min) + s_min
        sp["size"] = sp["size"].astype(int)

    for r in range(sp.shape[0]):
        ms = sp.loc[sp["idx"] == r]["size"].values
        plt.plot(
            sp.loc[sp["idx"] == r]["x"],
            sp.loc[sp["idx"] == r]["y"],
            "o",
            color="#2c94ea",
            mec="w",
            mew=2,
            ms=ms[0],
            alpha=0.8,
        )


# Main function
def calculate_object_detection_features(
    sp_file: str,
    path_obj_recog: str,
    path_ind_sp: str,
    output: bool = True,
) -> pd.DataFrame:
    # instantiate DataFrame
    df = None

    # image to scanpath
    img_file = ut.get_img_of_sp(sp_file)

    # load the input image
    image = mp.Image.create_from_file(img_file)

    # load object detector & detect
    detector = get_object_detector_object()
    detection_result = detector.detect(image)

    # fix detections based on manualy defined rules
    detection_result = process_detections(detection_result, img_file, output, path_obj_recog)

    # detect faces
    fr_image = face_recognition.load_image_file(img_file)
    face_locations = face_recognition.face_locations(fr_image, model="cnn")

    # ----- loop through scanpaths ----------
    sps = ut.load_scanpath(sp_file)
    for sp_i, sp in enumerate(sps):
        # id
        id = ut.get_sp_id(sp_file, sp_i)
        df_obj = pd.DataFrame(pd.Series(id), columns=["id"])

        # keep track of background fixations
        flag_fix_face = [False] * len(sp)

        # ----- Process faces in the images ----------
        df_obj["obj_n_fix_face"] = 0
        df_obj["obj_t_abs_on_face"] = 0
        df_obj["obj_t_rel_on_face"] = 0

        # loop fixations over faces
        for _, fix in sp.iterrows():
            # flag to skip 'p' if have been found on a face
            on_face = False

            # loop faces
            for face_location in face_locations:
                # prepare bbox
                top, right, bottom, left = face_location
                bbox_coords = [left, top, right - left, bottom - top]

                # check if fix inside bbox
                if (
                    intersect(bbox_coords, [int(fix["x"]), int(fix["y"]), 1, 1])
                    and not on_face
                ):
                    # update faces
                    df_obj["obj_n_fix_face"] += 1
                    df_obj["obj_t_abs_on_face"] += fix["duration"]

                    # update flag: fix_on_face
                    flag_fix_face[fix["idx"]] = True

                    # set flag to indicate that this 'p' was on a face already
                    on_face = True

        # calc relative time on faces
        df_obj["obj_t_rel_on_face"] = df_obj["obj_t_abs_on_face"] / sp["duration"].sum()

        # ----- Process the detected objects ----------
        df_obj["obj_n_fix_animate"] = 0
        df_obj["obj_n_fix_inanimate"] = 0
        df_obj["obj_n_fix_background"] = 0
        df_obj["obj_t_abs_on_animate"] = 0
        df_obj["obj_t_abs_on_inanimate"] = 0
        df_obj["obj_t_abs_on_background"] = 0
        df_obj["obj_t_rel_on_animate"] = 0
        df_obj["obj_t_rel_on_inanimate"] = 0
        df_obj["obj_t_rel_on_background"] = 0

        # loop fixations over objects
        for _, fix in sp.iterrows():
            # flag-list to skip 'p' if have been found on this kind of object
            on_object = []

            # loop detected objects
            for detection in detection_result.detections:
                # prepare bbox
                obj_name = detection.categories[0].category_name
                bbox_coords = [
                    detection.bounding_box.origin_x,
                    detection.bounding_box.origin_y,
                    detection.bounding_box.width,
                    detection.bounding_box.height,
                ]

                # create 'object' column if not done previously
                if f"obj_n_fix_{obj_name}_obj" not in df_obj.columns:
                    df_obj[f"obj_n_fix_{obj_name}_obj"] = 0
                    df_obj[f"obj_t_abs_on_{obj_name}_obj"] = 0

                # check if fix inside bbox
                if (
                    intersect(bbox_coords, [int(fix["x"]), int(fix["y"]), 1, 1])
                    and obj_name not in on_object
                ):
                    # update object
                    df_obj[f"obj_n_fix_{obj_name}_obj"] += 1
                    df_obj[f"obj_t_abs_on_{obj_name}_obj"] += fix["duration"]

                    # update animate / inanimate
                    if is_animate(obj_name):
                        df_obj["obj_n_fix_animate"] += 1
                        df_obj["obj_t_abs_on_animate"] += fix["duration"]
                    else:
                        df_obj["obj_n_fix_inanimate"] += 1
                        df_obj["obj_t_abs_on_inanimate"] += fix["duration"]

                    # set flag-list
                    on_object.append(obj_name)

            # check if fixation not on OBJECT nor FACE -> background
            if on_object == [] and not flag_fix_face[fix["idx"]]:
                df_obj["obj_n_fix_background"] += 1
                df_obj["obj_t_abs_on_background"] += fix["duration"]

        # calc relative time on "categories"
        df_obj["obj_t_rel_on_animate"] = min(
            [df_obj.loc[0, "obj_t_abs_on_animate"] / sp["duration"].sum(), 1]
        )
        df_obj["obj_t_rel_on_inanimate"] = min(
            [df_obj.loc[0, "obj_t_abs_on_inanimate"] / sp["duration"].sum(), 1]
        )
        df_obj["obj_t_rel_on_background"] = min(
            [df_obj.loc[0, "obj_t_abs_on_background"] / sp["duration"].sum(), 1]
        )
        for detection in detection_result.detections:
            obj_name = detection.categories[0].category_name
            df_obj[f"obj_t_rel_on_{obj_name}_obj"] = min(
                [
                    df_obj.loc[0, f"obj_t_abs_on_{obj_name}_obj"]
                    / sp["duration"].sum(),
                    1,
                ]
            )

        # ----- Concatenate to main DataFrame ----------
        df = pd.concat([df, df_obj], ignore_index=True)

        # ----- save figure to "data/.../" ----------
        if output:
            # create folder if not there
            path_objects = os.path.join(path_obj_recog)
            path_scanpath = os.path.join(path_ind_sp, "scanpath")
            path_sp_obj = os.path.join(path_ind_sp, "scanpath_objects")
            if not os.path.exists(path_scanpath):
                os.makedirs(path_scanpath)
            if not os.path.exists(path_sp_obj):
                os.makedirs(path_sp_obj)

            img = iio.imread(img_file)

            # fig 4 objects --------------------
            if sp_i == 0:
                plt.figure(
                    1,
                    figsize=(round(img.shape[1] * 0.015), round(img.shape[0] * 0.015)),
                    frameon=False,
                )
                ax = plt.gca()
                ax.set_axis_off()
                ax.imshow(img)

                draw_objects_faces(1, ax, face_locations, detection_result)

                plt.ylim(img.shape[0] - 1, 0)
                plt.xlim(0, img.shape[1] - 1)
                plt.tight_layout

                plt.savefig(
                    os.path.join(path_objects, os.path.basename(img_file)),
                    dpi=100,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()

            # fig 4 gaze behaviour --------------------
            plt.figure(
                2,
                figsize=(round(img.shape[1] * 0.015), round(img.shape[0] * 0.015)),
                frameon=False,
            )
            ax = plt.gca()
            ax.set_axis_off()
            ax.imshow(img)

            draw_scanpath(2, sp)

            plt.ylim(img.shape[0] - 1, 0)
            plt.xlim(0, img.shape[1] - 1)
            plt.tight_layout

            plt.savefig(
                os.path.join(path_scanpath, f"{id}.png"),
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # fig 4 objetcs & gaze behaviour --------------------
            plt.figure(
                3,
                figsize=(round(img.shape[1] * 0.015), round(img.shape[0] * 0.015)),
                frameon=False,
            )
            ax = plt.gca()
            ax.set_axis_off()
            ax.imshow(img)

            draw_objects_faces(
                3, ax, face_locations, detection_result, labels=False
            )
            draw_scanpath(3, sp)

            plt.ylim(img.shape[0] - 1, 0)
            plt.xlim(0, img.shape[1] - 1)
            plt.tight_layout

            plt.savefig(
                os.path.join(path_sp_obj, f"{id}.png"),
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

    return df


# --- main function to get scan_path features ---------------------------------
def get_features(
    output: bool = True,
    who: str = None,
    sal_mdl: str = None,
    slc=None,
) -> pd.DataFrame:
    # get files
    sp_files = sorted(ut.get_sp_files(who))

    # slice files
    if slc is not None:
        sp_files = sp_files[slice(slc[0], slc[1])]

    # instantiate df
    df = None

    # delete obj_recog files
    curdir = os.path.dirname(__file__)
    path_obj_recog = os.path.join(curdir, "..", "data", "obj_detection")
    path_ind_sp = os.path.join(curdir, "..", "data", "individual_scanpaths")
    if output:
        if not os.path.exists(path_obj_recog):
            os.mkdir(path_obj_recog)
        if not os.path.exists(path_ind_sp):
            os.mkdir(path_ind_sp)

    # loop sp files
    for sp_file in tqdm(sp_files):
        # extract features and concat to df
        df_file = calculate_sp_features(sp_file)

        # extract saliency features
        if sal_mdl is None:
            # -> based on DeepGazeIIE
            df_sal = calculate_saliency_features(sp_file, mdl="DeepGazeIIE")
            df_sal = df_sal.rename(
                columns={col: "dg_" + col for col in df_sal.columns if "sal_" in col}
            )
            df_file = df_file.merge(df_sal, on="id")

            # -> based on Sam_ResNET
            df_sal = calculate_saliency_features(sp_file, mdl="sam_resnet")
            df_sal = df_sal.rename(
                columns={col: "sam_" + col for col in df_sal.columns if "sal_" in col}
            )
            df_file = df_file.merge(df_sal, on="id")
        else:
            # -> from 'input_parameter'
            df_sal = calculate_saliency_features(sp_file, mdl=sal_mdl)
            df_file = df_file.merge(df_sal, on="id")

        # extract object detection features
        df_obj = calculate_object_detection_features(
            sp_file,
            path_obj_recog,
            path_ind_sp,
            output=output,
        )

        df_file = df_file.merge(df_obj, on="id")

        # concat file_df to complete_df
        df = pd.concat([df, df_file], ignore_index=True)

    # impute NaN's in object recognition features
    obj_cols = [col for col in df.columns if "obj_" in col]
    for c in obj_cols:
        df[[c]] = df[[c]].fillna(value=0)

    # return results
    return df


# --- if script is run by it's own --------------------------------------------
if __name__ == "__main__":
    curdir = os.path.dirname(__file__)

    sp_file = os.path.join(
        curdir,
        "..",
        "data",
        "Saliency4ASD",
        "TrainingData",
        "ASD",
        "ASD_scanpath_10.txt",
    )

    # path_df = os.path.join("..", "data", "df_sam.csv")
    # df = get_features(sal_mdl='sam_resnet')
    # df.to_csv(path_df, index=False)

    # df = get_features(who="td", sal_mdl="sam_resnet")
    # path_df = os.path.join(curdir, "..", "data", "df_sam_resnet_td.csv")
    # df.to_csv(path_df)

    # df = get_features(who="td", obj_save_fig=True, slc=[0, 10])
    # path_df = os.path.join(curdir, "..", "data", "df_deepgaze2e_td_3.csv")
    # df.to_csv(path_df)

    # calculate_sp_features(sp_file=sp_file)
    # calculate_saliency_features(sp_file=sp_file)

    # # delete obj_recog files
    # curdir = os.path.dirname(__file__)
    # path_obj_recog = os.path.join(curdir, "..", "data", "obj_recog")
    # if os.path.exists(path_obj_recog):
    #     shutil.rmtree(path_obj_recog)
    # os.mkdir(path_obj_recog)
    # calculate_object_detection_features(sp_file, path_obj_recog)

    path_df = os.path.join("..", "data", "df_deep_sam.csv")
    df = get_features()
    df.to_csv(path_df, index=False)
