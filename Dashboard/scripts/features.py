import os
import glob
import pickle
import numpy as np
import pandas as pd
import imageio.v3 as iio
from scipy import ndimage
from scripts import functions as fct


# region SCANPATH -------------------------------------------------
def scanpath(rec_file: str) -> pd.DataFrame:
    curdir = os.path.dirname(__file__)

    # instantiate df
    df = None

    # loop scanpaths
    sps = fct.load_scanpath(rec_file)
    for sp in sps:
        # img
        img = sp["img"].iloc[0]
        df_sp = pd.DataFrame(pd.Series(img), columns=["img"])

        # get size of image
        img_file = os.path.join(curdir, "content", "images", f"{img}.png")
        image_size = iio.imread(img_file).shape

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


# region SALIENCY -------------------------------------------------
def saliency(rec_file: str) -> pd.DataFrame:
    curdir = os.path.dirname(__file__)
    path_smaps = os.path.join(curdir, "content", "sal_pred")
    sal_dict = {"sam_resnet": "sam_", "DeepGazeIIE": "dg_"}

    # instantiate df
    df = None

    # loop scanpaths
    sps = fct.load_scanpath(rec_file)
    for sp in sps:
        # img
        img = sp["img"].iloc[0]
        df_sal = pd.DataFrame(pd.Series(img), columns=["img"])

        # loop models
        for mdl, prefix in sal_dict.items():

            # load saliency smap
            smap_file = glob.glob(os.path.join(path_smaps, mdl, f"{img}.*"))[0]
            loaded_sal_map = iio.imread(smap_file).astype(float)
            image_size = loaded_sal_map.shape

            # scanpath -> empirical saliency_map
            empirical_fix_map = np.zeros(image_size)
            empirical_fix_map[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = 1
            ndimage.gaussian_filter(
                empirical_fix_map, sigma=40, output=empirical_fix_map
            )

            # copy loaded salience map
            sal_map = loaded_sal_map.copy()
            sal_values = sal_map[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)]

            # features: fixation
            df_sal[prefix + "sal_first_fixation"] = sal_values[0]

            max_val = sal_map.max()
            for max_share in [0.75, 0.9]:
                try:
                    first_reaching = (
                        np.where(sal_values >= max_val * max_share)[0][0] + 1
                    )
                except IndexError:
                    first_reaching = max(20, len(sal_values) + 1)
                df_sal[prefix + "sal_first_above_{}*max_rank".format(max_share)] = (
                    first_reaching
                )

            # features: saliency
            df_sal[prefix + "sal_mean"] = np.mean(sal_values)
            df_sal[prefix + "sal_sum"] = np.sum(sal_values)
            df_sal[prefix + "sal_max"] = np.max(sal_values)

            # features: duration-by-saliency
            df_sal[prefix + "sal_weighted_duration_sum"] = np.sum(
                sp["duration"] * sal_values
            )
            df_sal[prefix + "sal_weighted_duration_mean"] = np.mean(
                sp["duration"] * sal_values
            )

            # features: normalisation for KLD
            empirical_fix_map -= empirical_fix_map.min()
            empirical_fix_map /= empirical_fix_map.sum()
            sal_map -= sal_map.min()
            if sal_map.sum() == 0:
                sal_map = np.ones_like(sal_map)
            sal_map /= sal_map.sum()

            eps = np.finfo(sal_map.dtype).eps
            df_sal[prefix + "sal_KLD"] = (
                empirical_fix_map * np.log(empirical_fix_map / (sal_map + eps) + eps)
            ).sum()

            # features: normalisation for NSS
            sal_map -= sal_map.mean()
            if sal_map.std() != 0:
                sal_map /= sal_map.std()

            df_sal[prefix + "sal_NSS"] = np.mean(
                sal_map[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)]
            )

        # concat to df
        df = pd.concat([df, df_sal], ignore_index=True)

    return df


# region OBJECTS -------------------------------------------------
# Check for intersection of object bounding box and scanpath coordinates
def intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Check if there is an intersection
    if min(x1 + w1, x2 + w2) >= max(x1, x2) and min(y1 + h1, y2 + h2) >= max(y1, y2):
        return True
    return False


# test if obejct is animate
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


def objects(rec_file: str) -> pd.DataFrame:
    curdir = os.path.dirname(__file__)

    # instantiate DataFrame
    df = None

    # load precalculated faces & objects
    detected_faces = pickle.load(
        open(os.path.join(curdir, "models", "objects_faces", "faces.pickle"), "rb")
    )
    detected_objects = pickle.load(
        open(os.path.join(curdir, "models", "objects_faces", "objects.pickle"), "rb")
    )

    # loop scanpaths
    sps = fct.load_scanpath(rec_file)
    for sp in sps:
        # img
        img = sp["img"].iloc[0]
        df_obj = pd.DataFrame(pd.Series(img), columns=["img"])

        # get detected faces & objetcs from dict
        objects = detected_objects[img]
        faces = detected_faces[img]

        # ----- Process faces in the images ----------
        df_obj["obj_n_fix_face"] = 0
        df_obj["obj_t_abs_on_face"] = 0
        df_obj["obj_t_rel_on_face"] = 0

        # keep track of background fixations
        flag_fix_face = [False] * len(sp)

        # loop fixations over faces
        for _, fix in sp.iterrows():
            # flag to skip 'p' if have been found on a face
            on_face = False

            # loop faces
            for face in faces:
                # check if fix inside bbox
                if (
                    intersect(face, [int(fix["x"]), int(fix["y"]), 1, 1])
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
            for obj in objects:
                # extract variables
                obj_name = obj["name"]
                bbox_coords = obj["bbox"]

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
        for obj in objects:
            obj_name = obj["name"]
            df_obj[f"obj_t_rel_on_{obj_name}_obj"] = min(
                [
                    df_obj.loc[0, f"obj_t_abs_on_{obj_name}_obj"]
                    / sp["duration"].sum(),
                    1,
                ]
            )

        # ----- Concatenate to main DataFrame ----------
        df = pd.concat([df, df_obj], ignore_index=True)

    return df
