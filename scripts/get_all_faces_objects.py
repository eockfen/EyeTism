import os
import pickle
import face_recognition
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


def get_object_detector_object(img_file):
    curdir = os.path.dirname(__file__)
    mdl_pth = os.path.join(curdir, "models", "mediapipe", "efficientdet_lite0.tflite")

    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=mdl_pth),
        score_threshold=0.2,
        max_results=10,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.ObjectDetector.create_from_options(options)
    image = mp.Image.create_from_file(img_file)
    detections = detector.detect(image)
    return detections


def process_detections(detections, img_nr):
    ignore = {
        4: [1, 2, 3, 4, 6, 7],
        10: [3, 4, 5, 6],
        11: [0, 1],
        13: [1],
        16: [1, 2, 3],
        17: [6],
        19: [2, 3],
        23: list(range(3, 10)),
        24: [1],
        27: [3],
        28: [2, 3],
        33: [2, 3, 4, 5, 6, 7, 8],
        34: [1],
        36: [1],
        38: [1],
        39: list(range(1, 5)),
        42: [4],
        44: [8],
        46: [7, 8, 9],
        51: [2],
        52: [0],
        53: [3],
        54: [2],
        55: [0, 1],
        63: [7],
        64: [1, 3],
        65: [2, 3],
        66: [0, 1],
        67: [2, 3],
        68: [2, 4, 9],
        70: [1, 2],
        72: [7],
        73: [0],
        74: [5, 7],
        76: [1, 2, 3],
        77: [0],
        79: [0],
        81: [0],
        83: [2, 3],
        84: [0, 3, 4, 5, 7, 8],
        87: [1],
        89: [2, 3],
        98: [1],
        102: [4],
        108: [2, 3],
        111: [2],
        112: [0],
        114: [1],
        115: [6],
        116: [4, 7],
        119: [1],
        123: [2],
        125: [0, 1, 2, 3, 4],
        127: [7],
        130: [6],
        131: [0],
        132: [1],
        134: [3, 6],
        135: [1],
        136: [1],
        137: [2],
        138: [4],
        139: [4, 6],
        141: [3, 6],
        143: [0],
        148: [1, 2],
        149: [6, 7],
        150: [4, 5],
        151: [2, 4, 5, 6],
        153: [0],
        154: [2],
        160: [5, 6, 7, 9],
        161: [2],
        162: [1],
        163: [5],
        167: [9],
        169: [2, 3],
        171: [2],
        172: [7, 8, 9],
        173: [2],
        175: [0],
        177: [2],
        183: [0, 4],
        186: [1],
        188: list(range(2, 16)),
        192: [7],
        198: [0],
        200: [7],
        201: [0],
        202: [3, 4, 5, 6],
        205: [4],
        206: [1],
        207: [3],
        208: [1],
        209: [4, 5, 7, 8, 9],
        210: [8],
        211: [0, 2],
        214: [7],
        215: [4, 9],
        217: [1],
        220: [1],
        224: [1],
        225: list(range(1, 6)),
        226: [1],
        228: [4],
        229: [1],
        231: [7, 9],
        232: [1],
        234: [5, 6, 7, 8, 9],
        237: [4],
        239: [1],
        241: [1, 2, 3, 4],
        242: [3],
        246: [1],
        247: [0, 1, 2],
        250: [0, 3, 4, 5],
        252: [1, 2, 3],
        254: [3],
        255: [1],
        256: [1],
        257: [5, 6],
        258: [0],
        259: [0, 5],
        261: [1],
        265: [8],
        266: [1, 2, 3],
        271: [1, 2, 3],
        273: [1, 3],
        274: [6],
        281: [4],
        282: [1, 2],
        284: [2, 3, 4, 6, 7, 8],
        285: [1, 4, 5, 6, 8],
        287: [1, 4],
        290: [3, 5],
        293: list(range(7)),
        295: [2],
        297: [6],
    }
    switch = {
        2: {1: "guitar"},
        3: {1: "guitar"},
        4: {0: "icecream"},
        8: {0: "leopard"},
        13: {0: "rodent"},
        15: {0: "tiger"},
        17: {4: "sealion"},
        18: {0: "lion"},
        22: {0: "deer"},
        23: {2: "fan"},
        26: {0: "cat"},
        27: {4: "orange"},
        29: {5: "handbag"},
        36: {0: "statue"},
        40: {0: "deer", 1: "deer", 2: "deer"},
        44: {7: "bicycle"},
        53: {2: "guitar"},
        67: {1: "airplane"},
        74: {8: "bag"},
        77: {2: "meat", 3: "meat", 5: "bread"},
        80: {0: "machine"},
        84: {1: "cat", 2: "cat", 6: "cat"},
        86: {0: "lion", 1: "lion"},
        91: {0: "dummy"},
        93: {2: "book"},
        94: {0: "butterfly"},
        96: {
            0: "gorilla",
            1: "gorilla",
            2: "gorilla",
            3: "plant",
            4: "gorilla",
            5: "gorilla",
        },
        97: {0: "squirrel"},
        98: {0: "deer"},
        100: {0: "deer"},
        101: {3: "statue"},
        106: {0: "toy"},
        109: {1: "trophy"},
        110: {0: "alligator"},
        111: {1: "book"},
        116: {3: "table", 5: "paper", 6: "couch"},
        117: {2: "toy", 3: "person", 4: "toy"},
        118: {1: "toy"},
        119: {3: "person"},
        124: {0: "elk"},
        135: {0: "rodent"},
        137: {1: "statue"},
        138: {3: "can"},
        146: {7: "art"},
        153: {1: "lock"},
        155: {0: "alligator"},
        156: {0: "fish", 1: "fish", 2: "fish", 3: "fish"},
        159: {0: "bird", 5: "bird"},
        167: {4: "cup"},
        169: {0: "balloon"},
        170: {0: "plant"},
        172: {6: "bag"},
        179: {0: "food"},
        181: {0: "insect"},
        188: {1: "sign"},
        202: {1: "glass"},
        205: {3: "sign"},
        208: {0: "car"},
        210: {9: "instrument"},
        212: {1: "cup"},
        219: {0: "fish"},
        220: {0: "fish"},
        223: {2: "hat", 3: "bottle"},
        242: {4: "bowl"},
        249: {0: "squirrel"},
        252: {4: "book"},
        259: {6: "bag"},
        263: {0: "ape"},
        267: {8: "plate"},
        272: {3: "person", 7: "person"},
        273: {2: "table"},
        287: {3: "suitcase", 6: "box "},
        288: {0: "statue"},
        290: {4: "hat"},
        294: {4: "tie"},
    }

    # handle RENAMINGS of detections
    if img_nr in switch.keys():
        for i in switch[img_nr].keys():
            detections.detections[i].categories[0].category_name = switch[img_nr][i]

    # handle DELETIONS of detections
    if img_nr in ignore.keys():
        to_delete = ignore[img_nr]
        detections.detections = [
            detections.detections[i]
            for i in range(len(detections.detections))
            if i not in to_delete
        ]

    return detections


curdir = os.path.dirname(__file__)
faces = {}
objects = {}

for img in tqdm(range(1, 301)):
    # image to scanpath
    img_file = os.path.join(
        curdir, "..", "data", "Saliency4ASD", "TrainingData", "Images", f"{img}.png"
    )

    # detect object & fix detections based on manualy defined rules
    detection_result = get_object_detector_object(img_file)
    detection_result = process_detections(detection_result, img)

    dtctns = []
    for detection in detection_result.detections:
        dict = {
            "name": detection.categories[0].category_name,
            "bbox": [
                detection.bounding_box.origin_x,
                detection.bounding_box.origin_y,
                detection.bounding_box.width,
                detection.bounding_box.height,
            ],
        }
        dtctns.append(dict)
    objects[img] = dtctns

    # detect faces
    fr_image = face_recognition.load_image_file(img_file)
    fr_results = face_recognition.face_locations(fr_image, model="cnn")

    fcs = []
    for face_location in fr_results:
        # prepare bbox
        top, right, bottom, left = face_location
        fcs.append([left, top, right - left, bottom - top])
    faces[img] = fcs

pickle.dump(faces, open("Dashboard/models/faces2.pickle", "wb"))
pickle.dump(objects, open("Dashboard/models/objects2.pickle", "wb"))
