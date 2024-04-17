import streamlit as st
import utils as ut

# setup vars, menu, style, and so on --------------------
ut.init_vars()
ut.default_style()
ut.create_menu()

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
if example == 'ASD':
    video_file = open("videos/asd.mp4", "rb")
else:
    video_file = open("videos/td.mp4", "rb")

# play the video
video_bytes = video_file.read()
st.video(video_bytes)

# ------------------------------------------------------------
# st.session_state
