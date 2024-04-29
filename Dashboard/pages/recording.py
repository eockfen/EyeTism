import os
import streamlit as st
from scripts import utils as ut
from scripts import functions as fct

# setup vars, menu, style, and so on --------------------
ut.init_vars()
ut.default_style()
ut.create_menu()

# page style ---------------------------------------------
st.title("Recording")
st.markdown("---")

# select patient ---------------------------------------------
st.subheader("Select Patient")

rec_patient = st.selectbox(
    "Record_Patient",
    st.session_state.patient_list,
    label_visibility="collapsed",
)
ut.h_spacer(height=2)

# record  ---------------------------------------------
st.subheader("Start Recording")

# choose example -----
st.radio(
    "choose an example:",
    options=["Typical Developed", "ASD"],
    horizontal=True,
    label_visibility="visible",
    key="record_example",
)

col_rec, col_note = st.columns([0.7, 0.3])
with col_rec:
    # load video -----
    if st.session_state.record_example == "ASD":
        video_file = open(os.path.join("content", "videos", "asd.mp4"), "rb")
    else:
        video_file = open(os.path.join("content", "videos", "td.mp4"), "rb")

    # play the video -----
    video_bytes = video_file.read()
    st.video(video_bytes)

    # save recording ---------------------------------------------
    # button -----
    saved = st.button(
        "Save Recording",
        on_click=fct.save_recording,
        args=(
            rec_patient,
            st.session_state.record_example,
        ),
    )

    # feedback
    container_saved = st.empty()
    if saved:
        container_saved.success(
            f"Recording saved successfully in '{st.session_state.last_saved_recording}'."
        )
        # update DB_records & session.state
        fct.update_rec_DB()
        st.session_state.last_saved_recording = None

# note ---------------------------------------------
with col_note:
    st.empty().info(
        """**PLEASE NOTE:**\n
_in the **final product**, an eye tracking software would be
implemented to capture actual eye movement data_\n
_until then, we **simulate** a data acquisition process by showing the actual
sequence of pictures and overlaying it with (real) eye gaze data_\n
_the displayed **gaze fixation point** as well as the **scanpath** will not be
visible to the patient in the real-world-recording situation_"""
    )

# ------------------------------------------------------------
if st.session_state.debug:
    st.session_state
