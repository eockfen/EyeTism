import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from skimage import exposure


def overlay_scanpath(fig, sp):
    # add saccades
    for r in range(1, sp.shape[0]):
        plt.plot(
            [sp.loc[sp["idx"] == r - 1]["x"], sp.loc[sp["idx"] == r]["x"]],
            [sp.loc[sp["idx"] == r - 1]["y"], sp.loc[sp["idx"] == r]["y"]],
            lw=6,
            c="#2c94ea",
        )

    # add fixations for individual plot
    s_min = 10
    s_max = 50
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
            mew=1.5,
            ms=ms[0],
            alpha=0.8,
        )

    return fig


def overlay_faces(fig, faces, lw: int = 1.5):
    ax = plt.gca()
    for face in faces:
        left, top, w, h = face
        rect = patches.Rectangle(
            (left, top),
            w,
            h,
            linewidth=lw,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    return fig


def overlay_objects(fig, objects, lw: int = 2, lbl: bool = True):
    ax = plt.gca()
    for obj in objects:
        # name & score
        obj_name = obj["name"]
        bbox_coords = obj["bbox"]

        # add rectangle
        rect = patches.Rectangle(
            (bbox_coords[0], bbox_coords[1]),
            bbox_coords[2],
            bbox_coords[3],
            linewidth=lw,
            edgecolor="orange",
            facecolor="none",
        )
        ax.add_patch(rect)

        # add label
        if lbl:
            txt_name = f"{obj_name}"
            plt.text(
                bbox_coords[0],
                bbox_coords[1],
                txt_name,
                fontsize=6,
                backgroundcolor="orange",
                verticalalignment="top",
            )
    return fig


def create_heatmap(img_nr, who):
    # image --------------------
    file_img = os.path.join(
        "content",
        "images",
        f"{img_nr}.png",
    )
    img = cv2.imread(file_img)
    image_size = img.shape[0:2]

    # scanpaths --------------------
    file_sp = os.path.join(
        "content",
        "scanpaths",
        f"{who}_{img_nr}.txt",
    )
    sp = pd.read_csv(file_sp, index_col=None)
    sp.columns = map(str.strip, sp.columns)
    sp.columns = map(str.lower, sp.columns)

    # individual fixation map --------------------
    fix_map = np.zeros(image_size)
    fix_map[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = 1
    ndimage.gaussian_filter(fix_map, sigma=40, output=fix_map)

    map_img = exposure.rescale_intensity(fix_map, out_range=(0, 255))
    map_img = np.uint8(map_img)
    heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)

    # merge map and frame
    hm_overlay = cv2.addWeighted(heatmap_img, 0.65, img, 0.5, 0)

    return hm_overlay
