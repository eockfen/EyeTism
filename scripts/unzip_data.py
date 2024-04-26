import os
import sys
import shutil
import zipfile


def default():
    curdir = os.path.dirname(__file__)
    datadir = os.path.join(curdir, "..", "data")
    os.mkdir(os.path.join(curdir, "tmp"))
    tmpdir = os.path.join(curdir, "tmp")

    # Saliency4ASD dataset
    print("extracting core dataset...")
    with zipfile.ZipFile(
        os.path.join(curdir, "..", "source", "Saliency4ASD.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(tmpdir)
    shutil.move(os.path.join(tmpdir, "Saliency4ASD"), datadir)

    # saliency predictions
    print("extracting saliency predictions...")
    with zipfile.ZipFile(
        os.path.join(curdir, "..", "source", "saliency_predictions.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(tmpdir)
    shutil.move(os.path.join(tmpdir, "saliency_predictions"), datadir)

    # deleting tmpdir again
    shutil.rmtree(tmpdir)
    print("... done")


def SAM():
    curdir = os.path.dirname(__file__)
    datadir = os.path.join(curdir, "..", "data")
    os.mkdir(os.path.join(curdir, "tmp"))
    tmpdir = os.path.join(curdir, "tmp")

    # SAM predictions
    print("extracting original SAM predictions...")
    with zipfile.ZipFile(
        os.path.join(curdir, "..", "source", "SAM_original.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(tmpdir)
    shutil.move(os.path.join(tmpdir, "SAM_original"), datadir)

    # deleting tmpdir again
    shutil.rmtree(tmpdir)
    print("... done")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        default()
    elif len(sys.argv) == 2 and sys.argv[1].upper() == "SAM":
        SAM()
    else:
        raise NotImplementedError
