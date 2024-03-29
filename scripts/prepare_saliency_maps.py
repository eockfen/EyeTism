import os
import sys
import glob
import shutil
from tqdm import tqdm
from PIL import Image, ImageChops


def search_SAM_predictions(path_ref, path_pred, redo: bool = False):
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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise NotImplementedError
    else:
        curdir = os.path.dirname(__file__)
        path_data = os.path.join(curdir, "..", "data")
        path_ref = os.path.join(path_data, "Saliency4ASD", "TrainingData", "Images")
        path_predictions = os.path.join(curdir, "..", "saliency_predictions")

        match sys.argv[1]:
            case "SAM":
                # check args
                redo = False
                if len(sys.argv) > 2:
                    match sys.argv[2]:
                        case "0":
                            redo = False
                        case "1":
                            redo = True
                        case _:
                            raise SyntaxError
                # run
                print(" -> searching for SAM predictions")
                search_SAM_predictions(path_ref, path_predictions, redo=redo)

            case _:
                print(f"ERROR: argument '{sys.argv[0]}' not implemented")
                raise NotImplementedError
