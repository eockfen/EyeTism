# collection of helper/utility scripts
import os
import glob
import pandas as pd
import numpy as np
import imageio.v3 as iio


def get_sp_id(sp_file: str, sp_i: int) -> str:
    """get id for given scanpath file and scanpath_index

    Args:
        sp_file (str): path to scanpath_*.txt
        sp_i (int): index of scanpath in this txt

    Returns:
        str: unique scanpath_id
    """
    grp = "asd" if "ASD" in sp_file.split("/")[-2] else "td"
    img_id = int(sp_file.split("_")[-1].split(".")[0])
    return f"{grp}_{img_id:03.0f}_{sp_i:02.0f}"


def get_sp_files(who: str = None) -> str:
    """return ALL scanpath textfiles

    Args:
        who (str, optional): 'ASD' or 'TD' if you only want to load specific
                                groups'. Defaults to None.

    Returns:
        str: list of paths to the scanpath_*txt files
    """
    # get files
    curdir = os.path.dirname(__file__)
    path_data = os.path.join(curdir, "..", "data", "Saliency4ASD", "TrainingData")
    sp_files_asd = glob.glob(os.path.join(path_data, "ASD", "*.txt"))
    sp_files_td = glob.glob(os.path.join(path_data, "TD", "*.txt"))

    if who is None:
        return sp_files_asd + sp_files_td
    elif who == "ASD":
        return sp_files_asd
    elif who == "TD":
        return sp_files_td


def get_img_of_sp(sp_file: str) -> str:
    """return corresponding image for a given scanpath_*.txt file

    Args:
        sp_file (str): scanpath_*.txt file for which the image is looked for

    Returns:
        str: path to the corresponding image
    """
    # get image file
    sp_dir = os.path.dirname(sp_file)
    img_id = sp_file.split("_")[-1].split(".")[0]
    return os.path.join(sp_dir, "..", "Images", f"{img_id}.png")


def split_scanpaths(data: pd.DataFrame) -> list:
    """splits concatenated scanpaths

    Args:
        data (DataFrame): pd.DataFrame containing loaded scnapaths

    Returns:
        list: list of pd.DataFrames for each individual scanpath
    """
    starts = np.where(data["idx"] == 0)[0]
    ends = np.append(starts[1:], len(data))
    assert starts.shape == ends.shape
    return [data[start:end] for start, end in zip(starts, ends)]


def load_scanpath(file: str) -> list:
    """load scanpath txt file and split individual scanpaths

    Args:
        file (str): path to scanpath txt file

    Returns:
        list: list of pd.DataFrames for each individual scanpath
    """
    # read scanpat*.txt file
    sp = pd.read_csv(file, index_col=None)
    sp.columns = map(str.strip, sp.columns)
    sp.columns = map(str.lower, sp.columns)

    # split into separate scanpaths from individuals
    return split_scanpaths(sp)


def load_saliency_map(sp_file: str, model: str) -> np.array:
    """load predicted saliency map for specified model and scanpath_*.txt file.

    Args:
        sp_file (str): name of scanpath_*.txt file
        model (str): Model for saliency predictions

    Returns:
        np.array: loaded saliency map as numpy array
    """
    # path
    curdir = os.path.dirname(__file__)
    path_smaps = os.path.join(curdir, "..", "saliency_predictions")

    # convert sp -> smap
    fname = os.path.basename(sp_file).split("_")[-1].split(".")[0]
    smap_file = glob.glob(os.path.join(path_smaps, model, f"{fname}.*"))[0]

    # load + return image as nparray
    return iio.imread(smap_file).astype(float)


if __name__ == "__main__":
    pass
