# it's all about the features
import os
import pandas as pd
import numpy as np
from PIL import Image
import mediapipe as mp
from scipy import ndimage
import json

if __name__ != "__main__":
    from scripts import utils as ut
else:
    import utils as ut

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
    image_size = Image.open(img_file).size

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

        # total and mean duration
        df_sp["sp_fix_duration_ms_total"] = np.sum(sp["duration"])
        df_sp["sp_fix_duration_ms_mean"] = np.mean(sp["duration"])

        # total, mean, and median saccade amplitude
        differences = np.diff(sp[["x", "y"]].values, axis=0)
        amplitudes = np.linalg.norm(differences, axis=1)

        df_sp["sp_len_px_total"] = np.sum(amplitudes)
        df_sp["sp_saccade_amplitude_px_mean"] = (
            np.mean(amplitudes) if len(amplitudes) else 0.0
        )
        df_sp["sp_saccade_amplitude_px_median"] = (
            np.median(amplitudes) if len(amplitudes) else 0.0
        )

        # mean and median fixation distance to centre
        dist_2_centre = np.sqrt(
            (sp["x"] - image_size[1] / 2) ** 2 + (sp["y"] - image_size[0] / 2) ** 2
        )

        df_sp["sp_distance_to_centre_px_mean"] = np.mean(dist_2_centre)
        df_sp["sp_distance_to_centre_px_median"] = np.median(dist_2_centre)

        # mean and median fixation distance to mean fixation
        mean_x, mean_y = sp["x"].mean(), sp["y"].mean()
        dist_to_mean = np.sqrt((sp["x"] - mean_x) ** 2 + (sp["y"] - mean_y) ** 2)

        df_sp["sp_distance_to_sp_mean_px_mean"] = np.mean(dist_to_mean)
        df_sp["sp_distance_to_sp_mean_px_median"] = np.median(dist_to_mean)

        # concat to df
        df = pd.concat([df, df_sp], ignore_index=True)

    return df

# --- Object detection from image features -------------
def detect objects(image_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_objectron = mp.solutions.objectron

    with mp_objectron.Objectron(static_image_mode=True) as objectron:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = objectron.process(image_rgb)

        detected_objects = []
        for detected_object in results.detected_objects:
            object_name = detected_object.class_name
            bounding_box = detected_object.bounding_box.flatten().tolist()
            detected_objects.append({"object_name": object_name, "bounding_box": bounding_box})

        return detected_objects

# --- Here are image object based features calculated for a given file -------------
def calculate_object_features():
    # get files
    sp_file = os.path.join(curdir, "..", "data", "Saliency4ASD", "TrainingData", "ASD", "ASD_scanpath_1.txt")
    images = os.path.join(curdir, "data","Saliency4ASD", "TrainingData", "Images")
    # instantiate df
    df = None
    extracted_features = []
    mp_objectron = mp.solutions.objectron

     # loop scanpaths
    sps = ut.load_scanpath(sp_file)
    for sp_i, sp in enumerate(sps):
        # id
        id = ut.get_sp_id(sp_file, sp_i)
        df_sp = pd.DataFrame(pd.Series(id), columns=["id"])

    images = load_images_from_folder(images_folder)
    

        for image, gaze_coords in zip(images, sps):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = objectron.process(image_rgb)

            if results.detected_objects:
                for detected_object in results.detected_objects:
                    bbox = detected_object.bounding_box

                    if bbox[0].x < gaze_coords[0] < bbox[2].x and bbox[0].y < gaze_coords[1] < bbox[2].y:
                        features = [...]  # Extract features here
                        extracted_features.append(features)

    extracted_features = np.array(extracted_features)


# --- main function to get scan_path features ---------------------------------
def get_features(who: str = None) -> pd.DataFrame:
    """main function to get all the features. implement more functions here, if
    you want to add more features, i.e. saliency, or object driven ones

    Args:
        who (str, optional): specify if only sub-group should be calculated. Defaults to None.

    Returns:
        DataFrame: pd.DataFrane containing scanpath features
    """
    # get files
    sp_files = ut.get_sp_files(who)

    # loop sp files
    for sp_file in sp_files:
        # extract features and concat to df
        df_file = calculate_sp_features(sp_file)

        # TEMPLATE: extract XXXXX features
        #df_XXX = calculate_XXX_features(sp_file)
        #df_file = df_file.merge(df_XXX, on="id")

        # concat file_df to complete_df
        df = pd.concat([df, df_file], ignore_index=True)

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

    # df = get_features(who="TD")
    # df = calculate_sp_features(sp_file=sp_file)
