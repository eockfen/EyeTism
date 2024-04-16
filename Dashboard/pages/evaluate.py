import streamlit as st
import utils as ut
import ET_functions as etf

# setup vars, menu, style, and so on --------------------
ut.init_vars()
ut.default_style()
ut.create_menu()

# page style ---------------------------------------------
st.title("Evaluate Patient")
st.markdown("---")

# selecting patient & recording ---------------------------------------------
st.subheader("Select Patient & Measurement")

col_name, col_recording = st.columns([1, 1], gap="small")
with col_name:
    st.selectbox(
        "Evaluate_Patient",
        st.session_state.patient_list,
        label_visibility="collapsed",
        key="eval_pat",
    )

with col_recording:
    st.selectbox(
        "Evaluate_Measurement",
        options=st.session_state.rec_db[st.session_state.eval_pat][1],
        label_visibility="collapsed",
        key="eval_meas",
    )

ut.h_spacer(0)

# calculate features ---------------------------------------------
_, col_load, _ = st.columns([0.3, 0.4, 0.3], gap="small")

# button -----
with col_load:
    loaded = st.button(
        "Analyse Eyetracking Data",
        on_click=etf.extract_features,
        )


# ------------------------------------------------------------
if st.session_state.debug:
    st.session_state
