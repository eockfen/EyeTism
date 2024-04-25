import os
import sys
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import imageio.v3 as iio

if __name__ != "__main__":
    from scripts import utils as ut
else:
    import utils as ut


def individual_maps(path_im, sal_mdl: str = "dg"):
    sal_name = {"dg": "DeepGazeIIE", "sam_resnet": "sam_resnet", "sam_vgg": "sam_vgg"}

    # check folders
    path_f = os.path.join(path_im, "fix")
    path_d = os.path.join(path_im, "dur")
    path_s = os.path.join(path_im, "sal" + "_" + sal_mdl)
    path_3 = os.path.join(path_im, "fds" + "_" + sal_mdl)

    if not os.path.exists(path_f):
        os.makedirs(path_f)
    if not os.path.exists(path_d):
        os.makedirs(path_d)
    if not os.path.exists(path_s):
        os.makedirs(path_s)
    if not os.path.exists(path_3):
        os.makedirs(path_3)

    # get files
    sp_files = ut.get_sp_files(who="ASD")

    # create empty max_value_lists
    ifm_max, idm_max, ism_max = 0, 0, 0

    # first, fetch all the data to find maximum values ---------------
    print("... first, load data to find global max...")
    # loop sp files
    for sp_file_ASD in tqdm(sp_files):
        # corresponding TD sp_file
        sp_file_TD = sp_file_ASD.replace("ASD/ASD_", "TD/TD_")
        sp_file = [sp_file_ASD, sp_file_TD]

        # get size of image
        img_file = ut.get_img_of_sp(sp_file[0])
        image_size = iio.imread(img_file).shape[0:2]

        # load SALIENCY PREDICTION map
        sal_map = ut.load_saliency_map(sp_file[0], sal_name[sal_mdl])

        # loop ASD/TD
        for spf in sp_file:
            # loop scanpaths - ASD
            sps = ut.load_scanpath(spf)
            for sp_i, sp in enumerate(sps):
                # individual fixation map --------------------
                ifm = np.zeros(image_size)
                ifm[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = 1
                ndimage.gaussian_filter(ifm, sigma=40, output=ifm)
                # store max
                ifm_max = max([ifm_max, ifm.max()])

                # individual duration map --------------------
                idm = np.zeros(image_size)
                idm[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = sp[
                    "duration"
                ].astype(int)
                ndimage.gaussian_filter(idm, sigma=40, output=idm)
                # store max
                idm_max = max([idm_max, idm.max()])

                # individual saliency map --------------------
                ism = np.zeros(image_size)
                ism[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = sal_map[
                    (sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)
                ]
                ndimage.gaussian_filter(ism, sigma=40, output=ism)
                # store max
                ism_max = max([ism_max, ism.max()])

    # then, run again through the scanpaths and save images right away --------
    print("... now creating images...")
    # loop sp files
    for sp_file_ASD in tqdm(sp_files):
        # corresponding TD sp_file
        sp_file_TD = sp_file_ASD.replace("ASD/ASD_", "TD/TD_")
        sp_file = [sp_file_ASD, sp_file_TD]

        # get size of image
        img_file = ut.get_img_of_sp(sp_file[0])
        image_size = iio.imread(img_file).shape[0:2]

        # load SALIENCY PREDICTION map
        sal_map = ut.load_saliency_map(sp_file[0], sal_name[sal_mdl])

        # loop ASD/TD
        for spf in sp_file:
            # loop scanpaths - ASD
            sps = ut.load_scanpath(spf)
            for sp_i, sp in enumerate(sps):
                # file to write
                id = ut.get_sp_id(spf, sp_i)
                ftw_f = os.path.join(path_f, f"{id}.jpg")
                ftw_d = os.path.join(path_d, f"{id}.jpg")
                ftw_s = os.path.join(path_s, f"{id}.jpg")
                ftw_3 = os.path.join(path_3, f"{id}.jpg")

                # individual fixation map --------------------
                ifm = np.zeros(image_size)
                ifm[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = 1
                ndimage.gaussian_filter(ifm, sigma=40, output=ifm)

                # individual duration map --------------------
                idm = np.zeros(image_size)
                idm[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = sp[
                    "duration"
                ].astype(int)
                ndimage.gaussian_filter(idm, sigma=40, output=idm)

                # individual saliency map --------------------
                ism = np.zeros(image_size)
                ism[(sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)] = sal_map[
                    (sp["y"].astype(int) - 1, sp["x"].astype(int) - 1)
                ]
                ndimage.gaussian_filter(ism, sigma=40, output=ism)

                itw_f = ifm / ifm_max * 254
                itw_d = idm / idm_max * 254
                itw_s = ism / ism_max * 254
                itw_3 = np.dstack((itw_f, itw_d, itw_s))

                iio.imwrite(ftw_f, itw_f.astype(np.uint8))
                iio.imwrite(ftw_d, itw_d.astype(np.uint8))
                iio.imwrite(ftw_s, itw_s.astype(np.uint8))
                iio.imwrite(ftw_3, itw_3.astype(np.uint8))

    print("... done creating 'all' maps.")


if __name__ == "__main__":
    # paths
    curdir = os.path.dirname(__file__)
    data = os.path.join(curdir, "..", "data")
    im = os.path.join(data, "individual_maps")

    # check args
    sal_mdl = "dg"
    if len(sys.argv) > 1 and "--redo" not in sys.argv[1]:
        sal_mdl = sys.argv[1].lower()

    if not any(check in sal_mdl for check in ["dg", "sam_resnet", "sam_vgg"]):
        print(f"ERROR: argument '{sal_mdl}' not implemented")
        raise NotImplementedError

    # run
    print(" -> creating individual fixation maps")
    individual_maps(im, sal_mdl=sal_mdl)
