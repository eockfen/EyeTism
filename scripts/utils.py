# collection of helper/utility scripts
import os
import glob
import pandas as pd
import numpy as np


def get_sp_files(who: str = None) -> str:
    """return ALL scanpath textfiles

    Returns:
        list[str]: list of paths to the scanpath_*txt files
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


if __name__ == "__main__":
    pass
