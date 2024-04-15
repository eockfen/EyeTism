import streamlit as st
import utils as ut
import pandas as pd
# import time
import os
from datetime import datetime

# setup vars, menu, style, and so on --------------------
ut.init_vars()
ut.default_style()
ut.create_menu()


# functions ---------------------------------------------
def save_recording(pat, kind):
    kind = "ASD" if kind == "ASD" else "TD"

    # patient vars
    pat_id = int(pat.split(":")[0])
    pat_idx = st.session_state.pat_db.id == pat_id

    # img & scanpath vars
    images = st.session_state.images
    sps = st.session_state.sp_idx_asd if kind == "ASD" else st.session_state.sp_idx_td

    if st.session_state.debug:
        print("---------------")
        print(pat)
        print(kind)
        print(sps)
        print("- - - -")
        print(" > patient id: " + str(pat_id))
        print(pat_idx)
        print("---------------")

    # get scanpaths
    df_sp = None
    for i, img in enumerate(images):
        sp = ut.load_scanpath(kind, img, sps[i])
        sp["img"] = img
        df_sp = pd.concat([df_sp, sp], ignore_index=True)

        if st.session_state.debug:
            print("- - - -")
            print("img :", img, "sp_i: ", sps[i])
            print(sp)

    # save csv file
    dt = datetime.today().strftime("%Y-%m-%d_%H-%M-%S.csv")
    name = f"id-{pat_id}_{dt}"
    df_sp.to_csv(os.path.join("recordings", f"{name}"), index=False)

    # save session state
    if "last_recording" not in st.session_state:
        st.session_state.last_rec = name

    # update DB_records
    ut.update_DB_recordings()


# page style ---------------------------------------------
st.title("Record Gaze")
st.markdown("---")

# select patient ------------------------------
st.subheader("Select Patient")

rec_patient = st.selectbox(
    "Select Patient:",
    [
        f"{int(r['id'])}: {r['name']} (age: {int(r['age'])})"
        for (_, r) in st.session_state.pat_db.iterrows()
    ],
    label_visibility="collapsed",
)

# note ------------------------------
ut.h_spacer(height=3)
st.empty().info(
    """**PLEASE sNOTE:**\n
_in the **final product**, an eye tracking software would be
implemented to capture actual eye movement data_\n
_until then, we **simulate** a data acquisition process by showing the actual
sequence of pictures and overlaying it with eye gaze data_\n
_the displayed **gaze fixation point** as well as the **scanpath** will not be
visible to the patient in the real-world-recording situation_"""
)
ut.h_spacer(height=3)

# record  ------------------------------
st.subheader("Start Recording")

example = st.radio(
    "choose an example:",
    options=["Typical Developed", "ASD"],
    horizontal=True,
    label_visibility="visible",
)

# choose example
if example == "ASD":
    video_file = open("videos/asd.mp4", "rb")
else:
    video_file = open("videos/td.mp4", "rb")

# play the video
video_bytes = video_file.read()
st.video(video_bytes)

# button
saved = st.button(
    "Save Recording",
    on_click=save_recording,
    args=(
        rec_patient,
        example,
    ),
)

# feedback
container_saved = st.empty()
if saved:
    container_saved.success(
        f"Recording saved successfully in '{st.session_state.last_rec}'."
    )

# ------------------------------------------------------------
if st.session_state.debug:
    st.session_state
