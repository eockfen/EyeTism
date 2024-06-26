import os
import utils as ut
import numpy as np
import cv2


# load scanpath ----------------------------------------
def load_scanpath():
    if isinstance(sp_nrs, int):
        sp_nr = [sp_nrs] * len(image_nrs)
    else:
        sp_nr = sp_nrs

    sp = []
    for i, img in enumerate(image_nrs):
        sp_file = os.path.join(
            curdir,
            "..",
            "data",
            "Saliency4ASD",
            "TrainingData",
            f"{group.upper()}",
            f"{group.upper()}_scanpath_{img}.txt",
        )

        # load specific scanpath
        loaded_sp = ut.load_scanpath(sp_file)
        if sp_nr[i] >= len(loaded_sp):
            sp_nr[i] = 0

        # make sure, that it is long enough (> threshold)
        img_sp = loaded_sp[sp_nr[i]]
        # while img_sp.duration.sum() < sp_dur_thresh:
        #     sp_nr[i] = sp_nr[i] + 1
        #     img_sp = loaded_sp[sp_nr[i]]

        # set new index
        img_sp = img_sp.set_index("idx")

        # crop too long scanpaths
        if img_sp.duration.sum() > 3000:
            time_cutoff = np.argmax(img_sp.duration.cumsum() > 3000)
            if time_cutoff == 0:
                img_sp = img_sp.iloc[0:1, :]
                img_sp.loc[img_sp.index == 0, "duration"] = 3000
            else:
                img_sp = img_sp.iloc[:time_cutoff, :]

        # extend last fix to end of 3000 ms
        if img_sp.duration.sum() < 3000:
            add_dur = (3000 - img_sp.duration.sum()) / img_sp.shape[0]
            img_sp["duration"] = (
                img_sp["duration"].apply(lambda x: x + add_dur).astype(int)
            )

        if img_sp.duration.sum() > 3000:
            raise "duration > 3000 ms ! check this our..."

        # further calculations
        img_sp.duration = img_sp.duration / 1000
        img_sp["time_total"] = img_sp.duration.cumsum()
        img_sp["max_frame"] = img_sp["time_total"] / fr
        img_sp["max_frame"] = img_sp["max_frame"].apply(np.ceil).astype(int)

        sp.append(img_sp)

    return sp, sp_nr


# load image ----------------------------------------
def load_images():
    if not isinstance(image_nrs, list):
        raise "Images need to be stored in a list"

    img = []
    for i in image_nrs:
        imgfile = os.path.join(
            curdir,
            "..",
            "data",
            "Saliency4ASD",
            "TrainingData",
            "Images",
            f"{i}.png",
        )
        img.append(cv2.imread(imgfile, cv2.COLOR_BGR2RGB))

    return img


# create video ----------------------------------------
def create_video():
    # check if destination folder exists -------------------------
    path_video = os.path.join(curdir, "..", "data", "videos")
    if not os.path.exists(path_video):
        os.mkdir(path_video)

    # ------ loading stuff ----------------------------------------
    imgs = load_images()
    sps, sp_nrs = load_scanpath()

    if join_videos:
        print("__print experimental video__")
        # first, find common canvas -----------------------
        common_x = []
        common_y = []
        for i, img in enumerate(imgs):
            common_x.append(img.shape[1])
            common_y.append(img.shape[0])
        common = [max(common_x), max(common_y)]

        # grey screen ------------------------------------
        img_grey = np.ones((common[1], common[0], 3), np.uint8) * 200

        # fix_cross screen ------------------------------------
        img_fix = np.ones((common[1], common[0], 3), np.uint8) * 200
        cv2.drawMarker(
            img_fix,
            (int(common[0] / 2), int(common[1] / 2)),
            (0, 0, 0),
            cv2.MARKER_CROSS,
            fc_size,
            fc_lw,
        )
        # Draw the circle on the canvas
        canvas = np.zeros_like(img_fix)
        cv2.circle(
            canvas,
            (int(common[0] / 2), int(common[1] / 2)),
            c_radius,
            c_edge_color,
            c_edge_lw,
        )
        cv2.circle(
            canvas, (int(common[0] / 2), int(common[1] / 2)), c_radius, c_color, -1
        )
        # put both together
        img_fix_circle = cv2.addWeighted(img_fix, 1, canvas, 10, 0)

        # init video ----------------------------------------
        video_name = os.path.join(path_video, f"{group}_all_images.mp4")
        video = cv2.VideoWriter(
            video_name, fourcc, fps, (img_fix.shape[1], img_fix.shape[0])
        )

        # 1-sec-grey-screen
        for frame in range(fps):
            video.write(img_grey)

        # countdown-screen
        for t in range(3, 0, -1):
            for frame in range(fps):
                img_cnt = np.ones((common[1], common[0], 3), np.uint8) * 200
                cv2.putText(
                    img_cnt,
                    str(t),
                    (int(common[0] / 2) - 60, int(common[1] / 2) + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    6,
                    (0, 0, 0),
                    12,
                )
                video.write(img_cnt)

        # loop images ----------------------------------------
        for i, img in enumerate(imgs):
            # get scanpath
            sp = sps[i]

            # prepare image on grey background
            img_full = img_fix.copy()
            x_off = int((img_full.shape[1] - img.shape[1]) / 2)
            y_off = int((img_full.shape[0] - img.shape[0]) / 2)
            img_full[y_off:y_off + img.shape[0], x_off:x_off + img.shape[1]] = img

            # 1-sec-grey-screen
            for frame in range(fps):
                video.write(img_fix_circle)

            # 3-sec-image
            for frame in range(frames):
                # find coordinates for this frame
                coord_idx = np.argmax(frame < sp.max_frame)
                x = sp.loc[coord_idx, "x"] + x_off
                y = sp.loc[coord_idx, "y"] + y_off

                # Draw the circle on the canvas
                canvas = np.zeros_like(img_full)
                cv2.circle(canvas, (x, y), c_radius, c_edge_color, c_edge_lw)
                cv2.circle(
                    canvas, (x, y), c_radius, c_color, -1
                )  # -1 -> fills the circle

                # Draw scanpath
                cv2.line(
                    canvas,
                    (int(img_full.shape[1] / 2), int(img_full.shape[0] / 2)),
                    (int(sp["x"][0] + x_off), int(sp["y"][0] + y_off)),
                    sp_color,
                    sp_lw,
                )
                for idx in range(coord_idx):
                    x0, y0 = sp.loc[idx, ["x", "y"]]
                    x1, y1 = sp.loc[idx + 1, ["x", "y"]]
                    cv2.line(
                        canvas,
                        (int(x0 + x_off), int(y0 + y_off)),
                        (int(x1 + x_off), int(y1 + y_off)),
                        sp_color,
                        sp_lw,
                    )

                # Overlay the circle canvas on top of the original image
                output_image = cv2.addWeighted(img_full, 1, canvas, 4, 0)

                # Write frame
                video.write(output_image)

        # 2-sec-grey-screen
        for frame in range(1 * fps):
            # Write frame
            video.write(img_fix)

        cv2.destroyAllWindows()
        video.release()
    else:
        for i, img in enumerate(imgs):
            # ------ get scanpath ----------------------------------------
            sp = sps[i]

            # ------ init video ----------------------------------------
            video_name = os.path.join(
                path_video, f"{group}_img{image_nrs[i]}_sp{sp_nrs[i]}.avi"
            )
            video = cv2.VideoWriter(
                video_name, fourcc, fps, (img.shape[1], img.shape[0])
            )

            # ------ print frames ----------------------------------------
            for frame in range(frames):
                # find coordinates for this frame
                coord_idx = np.argmax(frame < sp.max_frame)
                x = sp.loc[coord_idx, "x"]
                y = sp.loc[coord_idx, "y"]

                # Draw the circle on the canvas
                canvas = np.zeros_like(img)
                cv2.circle(canvas, (x, y), c_radius, c_edge_color, c_edge_lw)
                cv2.circle(
                    canvas, (x, y), c_radius, c_color, -1
                )  # -1 -> fills the circle

                # Draw scanpath
                cv2.line(
                    canvas,
                    (int(img.shape[1] / 2), int(img.shape[0] / 2)),
                    (int(sp["x"][0]), int(sp["y"][0])),
                    sp_color,
                    sp_lw,
                )
                for idx in range(coord_idx):
                    x0, y0 = sp.loc[idx, ["x", "y"]]
                    x1, y1 = sp.loc[idx + 1, ["x", "y"]]
                    cv2.line(
                        canvas, (int(x0), int(y0)), (int(x1), int(y1)), sp_color, sp_lw
                    )

                # Overlay the circle canvas on top of the original image
                output_image = cv2.addWeighted(img, 1, canvas, 2, 0)

                # Write frame
                video.write(output_image)

            cv2.destroyAllWindows()
            video.release()


# --- if script is run by it's own --------------------------------------------
if __name__ == "__main__":
    curdir = os.path.dirname(__file__)

    # images to generate as videos
    # n = 14
    # image_nrs = [95] * n  # need to be a list
    image_nrs = [203, 47, 95, 96, 138, 166, 191, 253, 287]  # need to be a list

    # which scanpaths to use
    group = "asd"
    sp_nrs = [6, 3, 0, 9, 4, 9, 8, 9, 10]  # ASD
    # group = "td"
    # sp_nrs = [7, 4, 7, 1, 9, 4, 0, 2, 5]  # TD
    # sp_nrs = list(range(n))  # if int -> sp is used for all images

    # scanpant_settings
    sp_dur_thresh = 2500  # sp must be at least that long

    # video settings - per image
    join_videos = True
    fps = 30
    L = 3000
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # circle
    c_radius = 23
    c_color = (0, 0, 255)
    c_edge_color = (255, 255, 255)
    c_edge_lw = 8

    # scanpath_lines
    sp_color = (0, 0, 255)
    sp_lw = 2

    # fix_cross
    fc_size = 80
    fc_lw = 5

    # calculate some variables
    fr = 1 / fps
    frames = int(L / 1000 * fps)

    # ------ create video ----------------------------------------
    create_video()

    # -----
    # ffmpeg -y -i td_all_images.mp4 -vcodec libx264 eyetism/td.mp4
    # ffmpeg -y -i asd_all_images.mp4 -vcodec libx264 eyetism/asd.mp4
