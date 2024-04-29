import streamlit as st
import utils as ut
import time
import os
import pandas as pd
import functions as fct
import imageio.v3 as iio
import image_processing as ip

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
    go_analyse = False
    if st.session_state.eval_meas:
        go_analyse = st.button(
            "Analyse",
        )

# preogress bar
prog_bar = st.progress(0, text="...")

ut.h_spacer(0)
# feedback
if go_analyse:
    # vars ---------------
    id = st.session_state.eval_pat.split(":")[0]
    rec_date = ut.ugly_date(st.session_state.eval_meas)
    path_evaluation = os.path.join("evaluation", f"id-{id}_{rec_date}")
    df_file = os.path.join(path_evaluation, "feat.csv")

    if not os.path.exists(path_evaluation):
        os.makedirs(path_evaluation)

    # extract features ---------------
    if os.path.exists(df_file):
        prog_bar.progress(3, text="load features")
        df = pd.read_csv(df_file, index_col=0)
        flag_done = True
    else:
        prog_bar.progress(3, text="extract features")
        df = fct.extract_features()

        # clean up features
        prog_bar.progress(50, text="preprocesse features")
        df = fct.clean_features(df)
        time.sleep(0.5)

        # saving df
        df.to_csv(df_file)
        flag_done = False

    # load classifiers ---------------
    prog_bar.progress(55, text="load classifiers")
    clf = fct.load_classifiers()
    time.sleep(0.5)

    # run predictions ---------------
    prog_bar.progress(60, text="predict TD/ASD")
    pred, proba = fct.predict(df, clf)
    asd = fct.hard_vote(pred)
    time.sleep(0.5)

    # save gaze behaviour images ---------------
    prog_bar.progress(70, text="create scanpath images")
    if not flag_done:
        fct.save_scanpath_figs()

    # visualize results ---------------
    prog_bar.progress(90, text="visualize results")
    time.sleep(1)
    prog_bar.progress(100, text="done")

    # container > RESULTS
    with st.container(border=True):
        ut.h_spacer(1)
        st.caption("classified as:")
        st.title(f"{asd}")
        st.divider()
        st.markdown(
            f"Eye movement data for **{sum(pred)}** (of {len(pred)}) images indicates ASD."
        )
        ut.h_spacer(1)

    # container > DETAILS
    with st.container(border=True):
        abc = "ABCDEFGHIJKLMNOP"
        ut.h_spacer(1)
        st.subheader("Overview of results for individual images")
        tabs = st.tabs(
            [
                f"Image **{abc[i]}**".center(15, "\u2001")
                for i, img in enumerate(st.session_state.opt["images"])
            ]
        )
        for i, img in enumerate(st.session_state.opt["images"]):
            # files
            file_img_sp = os.path.join(path_evaluation, f"{img}.png")
            file_img = os.path.join("content", "images", f"{img}.png")
            file_img_sal = os.path.join(
                "content", "sal_pred", "DeepGazeIIE", f"{img}.png"
            )
            fig_img_td_hm = ip.create_heatmap(img)

            # fill tab
            with tabs[i]:
                # general information ----------------------
                img_size = iio.imread(file_img_sp)
                if img_size.shape[0] < img_size.shape[1]:
                    c1, c2 = st.columns([2, 3], gap="medium")
                else:
                    c1, c2 = st.columns([1, 2], gap="medium")

                # recorded scanpath
                c1.image(file_img_sp)

                with c2:
                    # classified as
                    cls = (
                        "Autism Spectrum Disorder"
                        if pred[i] == 1
                        else "Typical Developed"
                    )
                    st.caption("Classified as:")
                    st.markdown(f"**{cls}**")
                    ut.h_spacer(3)

                    # model used
                    mdlnm = st.session_state.img2mdl[img]["name"]
                    st.caption("Model used:")
                    st.markdown(f"**{mdlnm}**")
                    ut.h_spacer(3)

                    # probalibility
                    thrshld = st.session_state.mdl_thresh[
                        st.session_state.img2mdl[img]["mdl"]
                    ]
                    st.caption("Probability of ASD (threshold):")
                    st.markdown(
                        f"**{round(100*proba[i],2)} %** ({round(thrshld*100,2)} %)"
                    )

                st.divider()

                # reference images ----------------------
                st.markdown("Reference Images")
                c21, c22, c23 = st.columns([1, 1, 1], gap="medium")
                with c21:
                    st.caption("Stimulus Image")
                    st.image(file_img)
                with c22:
                    st.caption("Saliency Prediction (DeepGazeIIE)")
                    st.image(file_img_sal)
                with c23:
                    st.caption("TD Heatmap")
                    st.pyplot(fig_img_td_hm)

    if st.session_state.debug:
        st.dataframe(df)
        st.write(pred)
        st.write(proba)

# ------------------------------------------------------------
if st.session_state.debug:
    st.session_state
