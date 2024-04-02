import os
import imageio.v3 as iio
import matplotlib.pyplot as plt


tst_set = [
    "112.png",
    "113.png",
    "120.png",
    "135.png",
    "138.png",
    "165.png",
    "166.png",
    "176.png",
    "191.png",
    "193.png",
    "20.png",
    "203.png",
    "207.png",
    "216.png",
    "233.png",
    "253.png",
    "258.png",
    "271.png",
    "272.png",
    "283.png",
    "287.png",
    "4.png",
    "45.png",
    "47.png",
    "73.png",
    "74.png",
    "8.png",
    "81.png",
    "95.png",
    "96.png",
]
val_set = [
    "100.png",
    "115.png",
    "116.png",
    "137.png",
    "143.png",
    "144.png",
    "147.png",
    "155.png",
    "158.png",
    "159.png",
    "16.png",
    "161.png",
    "205.png",
    "211.png",
    "223.png",
    "225.png",
    "238.png",
    "259.png",
    "279.png",
    "28.png",
    "280.png",
    "290.png",
    "298.png",
    "31.png",
    "35.png",
    "53.png",
    "70.png",
    "71.png",
    "72.png",
    "88.png",
]


def main():
    cur_dir = os.path.dirname(__file__)

    plt.figure(figsize=(10, 10))
    for i, file in enumerate(val_set):
        img = iio.imread(
            os.path.join(
                cur_dir, "..", "data", "Saliency4ASD", "TrainingData", "Images", file
            )
        )

        plt.subplot(6, 5, i + 1)
        plt.imshow(img)
        plt.title(file)
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )

    plt.tight_layout()
    plt.savefig(os.path.join(cur_dir, "..", "images", "val_set.png"), dpi=150)

    plt.figure(figsize=(10, 10))
    for i, file in enumerate(tst_set):
        img = iio.imread(
            os.path.join(
                cur_dir, "..", "data", "Saliency4ASD", "TrainingData", "Images", file
            )
        )

        plt.subplot(6, 5, i + 1)
        plt.imshow(img)
        plt.title(file)
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )

    plt.tight_layout()
    plt.savefig(os.path.join(cur_dir, "..", "images", "tst_set.png"), dpi=150)


# --- if script is run by it's own --------------------------------------------
if __name__ == "__main__":
    main()
