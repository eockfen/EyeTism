import os
import sys
import glob
import shutil
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import imageio.v3 as iio
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

import torch
from scipy.special import logsumexp
import deepgaze_pytorch

if __name__ != "__main__":
    from scripts import utils as ut
else:
    import utils as ut


def deepgaze(path_ref, path_pred, redo: bool = False, disp: bool = False):
    def normalize_log_density(log_density):
        """convertes a log density into a map of the cummulative distribution function."""
        density = np.exp(log_density)
        flat_density = density.flatten()
        inds = flat_density.argsort()[::-1]
        sorted_density = flat_density[inds]
        cummulative = np.cumsum(sorted_density)
        unsorted_cummulative = cummulative[np.argsort(inds)]
        return unsorted_cummulative.reshape(log_density.shape)

    def visualize_distribution(log_densities, ax=None):
        if ax is None:
            ax = plt.gca()
        t = normalize_log_density(log_densities)
        img = ax.imshow(t, cmap=plt.cm.viridis)
        levels = [0, 0.25, 0.5, 0.75, 1.0]
        cs = ax.contour(t, levels=levels, colors="black")
        # plt.clabel(cs)

        return img, cs

    # ------------------------------------------------------
    # check folders
    if not os.path.exists(path_pred):
        os.makedirs(path_pred)

    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    centerbias_template = np.load(
        os.path.join(curdir, "deepgaze_pytorch", "centerbias_mit1003.npy")
    )

    # loop referrence images
    refImages = glob.glob(os.path.join(path_ref, "*.png"))
    for img_file in tqdm(refImages):
        # load image
        image = iio.imread(img_file)

        # instantiate model
        DEVICE = None
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

        # rescale to match image size
        centerbias = ndimage.zoom(
            centerbias_template,
            (
                image.shape[0] / centerbias_template.shape[0],
                image.shape[1] / centerbias_template.shape[1],
            ),
            order=0,
            mode="nearest",
        )

        # renormalize log density / create tensors / predict
        centerbias -= logsumexp(centerbias)
        image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
        centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
        log_density_prediction = model(image_tensor, centerbias_tensor)

        # convert log-densitiy-maps to greyscale image
        smap = 1 - normalize_log_density(
            log_density_prediction.detach().cpu().numpy()[0, 0]
        )
        smap *= 254

        # write
        iio.imwrite(
            os.path.join(path_pred, os.path.basename(img_file)), smap.astype(np.uint8)
        )

        # plotting
        if disp:
            _, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
            # plt.set_cmap('Greys_r')
            axs[0].imshow(image)
            axs[0].set_axis_off()
            plt.set_cmap("Greys_r")
            axs[1].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])
            axs[1].set_axis_off()
            visualize_distribution(
                log_density_prediction.detach().cpu().numpy()[0, 0], ax=axs[2]
            )
            axs[2].set_axis_off()
            plt.show()


def search_SAM_predictions(path_ref, path_pred, redo: bool = False):
    # check folders
    if not os.path.exists(path_pred):
        os.makedirs(path_pred)

    # SAM paths + models
    path_SAM = os.path.join(curdir, "..", "data", "SAM_original")
    path_SAMimages = os.path.join(path_SAM, "images")
    mdls = ["sam_resnet", "sam_vgg"]

    # get files
    refImages = glob.glob(os.path.join(path_ref, "*.png"))
    samImages = glob.glob(os.path.join(path_SAMimages, "*.jpeg"))
    print(f"compare {len(refImages)} refs to {len(samImages)} MIT1003 images")

    # check variable
    check = []
    # loop ref images
    for refImg in tqdm(refImages):
        ref = Image.open(refImg)
        refImgBase = os.path.basename(refImg).split(".")[0]

        # check if already done / or / redo==True
        if (
            os.path.isfile(os.path.join(path_pred, mdls[0], f"{refImgBase}.jpg"))
            and os.path.isfile(os.path.join(path_pred, mdls[1], f"{refImgBase}.jpg"))
            and not redo
        ):
            check.append(refImgBase)
            continue

        # loop MIT1003 images
        for samImg in samImages:
            sam = Image.open(samImg)

            # check if identical
            if ImageChops.difference(ref, sam).getbbox() is None:
                for mdl in mdls:
                    # paths to SAM_salience_prediction
                    samImgBase = os.path.basename(samImg).split(".")[0]
                    file_mdl = os.path.join(path_SAM, mdl, f"{samImgBase}.jpg")

                    # path to copied & renamed file
                    file_pred = os.path.join(path_pred, mdl, f"{refImgBase}.jpg")
                    shutil.copy(file_mdl, file_pred)

                # go to next ref image
                check.append(refImgBase)
                break

    # assert
    if len(check) == len(refImages):
        return print("SAM saliency successfully found and sorted")
    else:
        return print("... something went wrong searching for SAM saliency maps")


def individual_fixation_maps(path_esm, redo: bool = False):
    # check folders
    if not os.path.exists(path_esm):
        os.makedirs(path_esm)

    # get files
    sp_files = ut.get_sp_files()

    # loop sp files
    for sp_file in tqdm(sp_files):
        # get size of image
        img_file = ut.get_img_of_sp(sp_file)
        image_size = iio.imread(img_file).shape[0:2]

        # loop scanpaths
        sps = ut.load_scanpath(sp_file)
        for sp_i, sp in enumerate(sps):
            # file to write
            id = ut.get_sp_id(sp_file, sp_i)
            ftw = os.path.join(path_esm, f"{id}.jpg")

            # check if already done / or / redo==True
            if os.path.isfile(ftw) and not redo:
                continue

            # individual fixation map
            ifm = np.zeros(image_size)
            ifm[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = 1
            ndimage.gaussian_filter(ifm, sigma=40, output=ifm)

            # scale to interval [0 - 254]
            ifm = ifm / ifm.max() * 254

            # write
            iio.imwrite(ftw, ifm.astype(np.uint8))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise NotImplementedError
    else:
        # paths
        curdir = os.path.dirname(__file__)
        data = os.path.join(curdir, "..", "data")
        ref_images = os.path.join(data, "Saliency4ASD", "TrainingData", "Images")
        sal_predictions = os.path.join(curdir, "..", "saliency_predictions")
        sal_pred_deepgaze = os.path.join(sal_predictions, "DeepGazeIIE")
        ifm = os.path.join(data, "Saliency4ASD", "TrainingData", "Individual_FixMaps")

        # check args
        redo = False
        if len(sys.argv) > 2 and sys.argv[2] == "1":
            redo = True

        # do what have to be done
        match sys.argv[1].upper():
            case "SAM":  # Saliency Attentive Model
                # run
                print(" -> searching for SAM predictions")
                search_SAM_predictions(ref_images, sal_predictions, redo=redo)

            case "IFM":  # Individual Fixation Maps
                # run
                print(" -> creating individual fixation maps")
                individual_fixation_maps(ifm, redo=redo)

            case "DG":  # DeepGaze IIE saliency maps
                # run
                print(" -> predict saliency maps via DeepGaze IIE")
                deepgaze(ref_images, sal_pred_deepgaze, redo=redo)

            case _:
                print(f"ERROR: argument '{sys.argv[1]}' not implemented")
                raise NotImplementedError
