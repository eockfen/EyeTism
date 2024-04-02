# it's all about the features
import os
import pandas as pd
import numpy as np
import imageio.v3 as iio
from scipy import ndimage

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


# --- main function to get scan_path features ---------------------------------
def get_features(who: str = None, sal_mdl: str = "sam_resnet") -> pd.DataFrame:
    """main function to get all the features. implement more functions here, if
    you want to add more features, i.e. saliency, or object driven ones

    Args:
        who (str, optional): specify if only sub-group should be calculated. Defaults to None.

    Returns:
        DataFrame: pd.DataFrane containing scanpath features
    """
    # get files
    sp_files = ut.get_sp_files(who)

    # instantiate df
    df = None

    # loop sp files
    for sp_file in sp_files:
        # extract features and concat to df
        df_file = calculate_sp_features(sp_file)

        # extract saliency features
        df_sal = calculate_saliency_features(sp_file, mdl=sal_mdl)
        df_file = df_file.merge(df_sal, on="id")

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

    # get_sp_features(who="TD")
    # calculate_sp_features(sp_file=sp_file)
    # calculate_saliency_features(sp_file=sp_file)
