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


# --- TEMPLATE 4 new features based on SCANPATH_*.txt fiels -------------------
def calculate_XXX_features(sp_file: str) -> pd.DataFrame:
    """calculate XXXXXXXXX features for *.txt file

    Args:
        sp_file (str): path to scanpath_*.txt

    Returns:
        DatFrame: pd.DatFrame containing calculated features
    """
    # instantiate df
    df = None

    # loop scanpaths
    sps = ut.load_scanpath(sp_file)
    for sp_i, sp in enumerate(sps):
        # id
        id = ut.get_sp_id(sp_file, sp_i)
        df_XXX = pd.DataFrame(pd.Series(id), columns=["id"])

        # -------- any features ------------------
        df_XXX["dummy_feature_name"] = len(sp)

        # concat to df
        df = pd.concat([df, df_XXX], ignore_index=True)

    return df


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
# Create an ObjectDetector object.
def get_object_detector_object():
    curdir = os.path.dirname(__file__)
    mdl_pth = os.path.join(curdir, "..", "models", "efficientdet.tflite")
    base_options = python.BaseOptions(model_asset_path=mdl_pth)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, score_threshold=0.5
    )
    detector = vision.ObjectDetector.create_from_options(options)
    return detector


# Check for intersection of object bounding box and scanpath coordinates
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


# test if obejct is animate
def is_animate(x):
    animate = [
        "bird",
        "person",
        "giraffe",
        "sheep",
        "horse",
        "bear",
        "dog",
        "cow",
        "cat",
        "elephant",
    ]
    return True if x in animate else False


# Main function
def calculate_object_detection_features(
    sp_file: str, obj_save_fig: bool = False
) -> pd.DataFrame:
    # Load object & face detector
    detector = get_object_detector_object()

    # Instantiate DataFrame
    df = None

    # image to scanpath
    img_file = ut.get_img_of_sp(sp_file)

    # Load the input image
    image = mp.Image.create_from_file(img_file)

    # Detect objects in the input image
    detection_result = detector.detect(image)

    # Detect faces
    fr_image = face_recognition.load_image_file(img_file)
    face_locations = face_recognition.face_locations(fr_image, model="cnn")

    # Loop through scanpaths
    sps = ut.load_scanpath(sp_file)
    for sp_i, sp in enumerate(sps):
        # id
        id = ut.get_sp_id(sp_file, sp_i)
        df_obj = pd.DataFrame(pd.Series(id), columns=["id"])

        # keep track of background fixations
        flag_fix_face = [False] * len(sp)

        # Process faces in the images
        df_obj["obj_n_fix_face"] = 0
        df_obj["obj_t_abs_on_face"] = 0
        df_obj["obj_t_rel_on_face"] = 0

        # Loop fixations
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

        # Process the detection result and extract rectangle coordinates
        df_obj["obj_n_fix_animate"] = 0
        df_obj["obj_n_fix_inanimate"] = 0
        df_obj["obj_n_fix_background"] = 0
        df_obj["obj_t_abs_on_animate"] = 0
        df_obj["obj_t_abs_on_inanimate"] = 0
        df_obj["obj_t_abs_on_background"] = 0
        df_obj["obj_t_rel_on_animate"] = 0
        df_obj["obj_t_rel_on_inanimate"] = 0
        df_obj["obj_t_rel_on_background"] = 0

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
                print(f"{id} : background at {fix['x']} / {fix['y']}")
                df_obj["obj_n_fix_background"] += 1
                df_obj["obj_t_abs_on_background"] += fix["duration"]

        # calc relative time on "categories"
        df_obj["obj_t_rel_on_animate"] = (
            df_obj["obj_t_abs_on_animate"] / sp["duration"].sum()
        )
        df_obj["obj_t_rel_on_inanimate"] = (
            df_obj["obj_t_abs_on_inanimate"] / sp["duration"].sum()
        )
        df_obj["obj_t_rel_on_background"] = (
            df_obj["obj_t_abs_on_background"] / sp["duration"].sum()
        )
        for detection in detection_result.detections:
            obj_name = detection.categories[0].category_name
            df_obj[f"obj_t_rel_on_{obj_name}_obj"] = (
                df_obj[f"obj_t_abs_on_{obj_name}_obj"] / sp["duration"].sum()
            )

        # Concatenate to main DataFrame
        df = pd.concat([df, df_obj], ignore_index=True)

        # save resulting figure to "images/obj_recog_results/"
        if obj_save_fig:
            # create folder if not there
            curdir = os.path.dirname(__file__)
            path_img = os.path.join(curdir, "..", "data", "obj_recog_results")
            if not os.path.exists(path_img):
                os.makedirs(path_img)

            img = iio.imread(img_file)
            plt.figure(
                figsize=(round(img.shape[1] * 0.015), round(img.shape[0] * 0.015))
            )
            ax = plt.gca()
            ax.imshow(img)

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
                rect = patches.Rectangle(
                    (detection.bounding_box.origin_x, detection.bounding_box.origin_y),
                    detection.bounding_box.width,
                    detection.bounding_box.height,
                    linewidth=2,
                    edgecolor="orange",
                    facecolor="none",
                )
                ax.add_patch(rect)

            # add fixations
            plt.plot(sp["x"], sp["y"], "+", color="k", mew=3, ms=40)
            plt.plot(sp["x"], sp["y"], "o", color="r", mec="k", mew=1.5, ms=10)

            # style
            plt.ylim(img.shape[0] - 1, 0)
            plt.xlim(0, img.shape[1] - 1)
            plt.tight_layout

            # save plot
            plt.savefig(os.path.join(path_img, f"{id}.png"), dpi=100)
            plt.close()

    return df


# --- main function to get scan_path features ---------------------------------
def get_features(
    who: str = None, sal_mdl: str = "DeepGazeIIE", obj_save_fig: bool = False, slc=None
) -> pd.DataFrame:
    """main function to get all the features. implement more functions here, if
    you want to add more features, i.e. saliency, or object driven ones

    Args:
        who (str, optional): specify if only sub-group should be calculated. Defaults to None.

    Returns:
        DataFrame: pd.DataFrane containing scanpath features
    """
    # get files
    sp_files = ut.get_sp_files(who)

    # slice files
    if slc is not None:
        sp_files = sorted(sp_files)
        sp_files = sp_files[slice(slc[0], slc[1])]

    # instantiate df
    df = None

    # loop sp files
    for sp_file in tqdm(sp_files):
        # extract features and concat to df
        df_file = calculate_sp_features(sp_file)

        # extract saliency features
        df_sal = calculate_saliency_features(sp_file, mdl=sal_mdl)
        df_file = df_file.merge(df_sal, on="id")

        # extract object detection features
        df_obj = calculate_object_detection_features(sp_file, obj_save_fig=obj_save_fig)
        df_file = df_file.merge(df_obj, on="id")

        # TEMPLATE: extract XXXXX features
        # df_XXX = calculate_XXX_features(sp_file)
        # df_file = df_file.merge(df_XXX, on="id")

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
        "ASD_scanpath_1.txt",
    )

    # get_features()

    # df = get_features(who="td", sal_mdl="sam_resnet")
    # path_df = os.path.join(curdir, "..", "data", "df_sam_resnet_td.csv")
    # df.to_csv(path_df)

    # df = get_features(who="td", obj_save_fig=True, slc=[0, 10])
    # path_df = os.path.join(curdir, "..", "data", "df_deepgaze2e_td_3.csv")
    # df.to_csv(path_df)

    # calculate_sp_features(sp_file=sp_file)
    # calculate_saliency_features(sp_file=sp_file)
    # calculate_object_detection_features(sp_file=sp_file)
