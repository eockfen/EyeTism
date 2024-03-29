import os
import sys
import glob
import shutil
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import imageio.v3 as iio
from PIL import Image, ImageChops

if __name__ != "__main__":
    from scripts import utils as ut
else:
    import utils as ut


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
            ndimage.gaussian_filter(ifm, sigma=43, output=ifm)

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

            case _:
                print(f"ERROR: argument '{sys.argv[0]}' not implemented")
                raise NotImplementedError
