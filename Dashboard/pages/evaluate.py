import streamlit as st
import utils as ut
import time
import functions as fct

# setup vars, menu, style, and so on --------------------
ut.init_vars()
ut.default_style()
ut.create_menu()

# page style ---------------------------------------------
st.title("Evaluate Patient")
st.markdown("---")

# selecting patient & recording ---------------------------------------------
st.subheader("Select Patient & Measurement")

col_name, col_recording, col_btn = st.columns([4, 6, 2], gap="small")
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

with col_btn:
    go_analyse = st.button(
        "Analyse",
        )

# preogress bar
prog_bar = st.progress(0, text="...")

ut.h_spacer(0)

# feedback
if go_analyse:
    # extract features
    prog_bar.progress(5, text="extracting features")
    df = fct.extract_features()

    # clean up features
    prog_bar.progress(50, text="preprocessing features")
    df = fct.clean_features(df)
    time.sleep(0.75)

    # run predictions
    prog_bar.progress(70, text="predict TD/ASD")
    pred, proba = fct.predict(df)
    time.sleep(0.75)

    prog_bar.progress(95, text="visualizing results")
    time.sleep(2)

    prog_bar.progress(100, text="done")

    if st.session_state.debug:
        st.dataframe(df)

# ------------------------------------------------------------
if st.session_state.debug:
    st.session_state
