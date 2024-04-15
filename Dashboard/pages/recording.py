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

    # patient, img & scanpath vars
    pat_id = int(pat.split(":")[0])
    images = st.session_state.images
    sps = st.session_state.sp_idx_asd if kind == "ASD" else st.session_state.sp_idx_td

    # get scanpaths
    df_sp = None
    for i, img in enumerate(images):
        sp = ut.load_scanpath(kind, img, sps[i])
        sp["img"] = img
        df_sp = pd.concat([df_sp, sp], ignore_index=True)

    # save csv file
    dt = datetime.today().strftime("%Y-%m-%d_%H-%M-%S.csv")
    name = f"id-{pat_id}_{dt}"
    df_sp.to_csv(os.path.join("recordings", f"{name}"), index=False)

    # save session state
    if st.session_state.last_saved_recording is None:
        st.session_state.last_saved_recording = name


# page style ---------------------------------------------
st.title("Record Gaze")
st.markdown("---")

# select patient ------------------------------
st.subheader("Select Patient")

rec_patient = st.selectbox(
    "Record_Patient",
    st.session_state.patient_list,
    label_visibility="collapsed",
)

# note ------------------------------
ut.h_spacer(height=3)
st.empty().info(
    """**PLEASE NOTE:**\n
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

# choose example -----
example = st.radio(
    "choose an example:",
    options=["Typical Developed", "ASD"],
    horizontal=True,
    label_visibility="visible",
)

# load video -----
if example == "ASD":
    video_file = open("videos/asd.mp4", "rb")
else:
    video_file = open("videos/td.mp4", "rb")

# play the video -----
video_bytes = video_file.read()
st.video(video_bytes)

# button -----
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
        f"Recording saved successfully in '{st.session_state.last_saved_recording}'."
    )
    # update DB_records
    ut.update_DB_recordings()

    st.session_state.last_saved_recording = None


# ------------------------------------------------------------
if st.session_state.debug:
    st.session_state
