import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v3 as iio


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


def create_heatmap(img):
    file_img_td_hm = os.path.join(
        "content",
        "images",
        f"{img}.png",
    )
    img = iio.imread(file_img_td_hm)
    fig = plt.figure(
        figsize=(round(img.shape[1] * 0.02), round(img.shape[0] * 0.02)),
        frameon=False,
    )
    ax = plt.gca()
    ax.set_axis_off()
    ax.imshow(img)

    return fig
