import streamlit as st
import pandas as pd
import glob
import os
import datetime
import utils as ut


# initialize session_state variabled ------------------------------------------
def init_vars():
    # fetch scanpath from images folder
    st.session_state.opt = {
        "images": [203, 47, 95, 96, 138, 166, 191, 253, 287],
        "sp_idx_asd": {
            47: 3,
            95: 0,
            96: 9,
            138: 4,
            166: 9,
            191: 8,
            203: 6,
            253: 9,
            287: 10,
        },
        "sp_idx_td": {
            47: 4,
            95: 7,
            96: 1,
            138: 9,
            166: 4,
            191: 0,
            203: 7,
            253: 2,
            287: 5,
        },
    }

    DB = pd.read_csv(os.path.join("files", "patients.csv"))
    if "patient_db" not in st.session_state:
        st.session_state.pat_db = DB
    if "edited_patient_db" not in st.session_state:
        st.session_state.pat_db_update = DB.copy()

    if "patient_list" not in st.session_state:
        st.session_state.patient_list = [
            f"{int(r['id'])}: {r['name']} (age: {int(r['age'])})"
            for (_, r) in st.session_state.pat_db.iterrows()
        ]

    if "record_example" not in st.session_state:
        st.session_state.record_example = "Typical Developed"

    if "last_saved_recording" not in st.session_state:
        st.session_state.last_saved_recording = None

    if "recordings_db" not in st.session_state:
        DB_REC = {}
        for p in st.session_state.patient_list:
            id = p.split(":")[0]
            recs = sorted(
                [
                    f.split("/")[1].split(f"id-{id}_")[1]
                    for f in glob.glob("recordings/*.csv")
                    if "/id-" + str(id) + "_" in f
                ]
            )
            recs_nice = [nice_date(f) for f in recs]
            DB_REC[p] = [recs, recs_nice]
            # DB_REC[p] = recs
        st.session_state.rec_db = DB_REC

    if "img2mdl" not in st.session_state:
        st.session_state.img2mdl = {
            47: {"mdl": "NB_s", "name": "Naive Bayes - stacked on RF+XGB+SVC"},
            95: {"mdl": "SVC", "name": "Support Vector Classifier"},
            96: {"mdl": "NB_s", "name": "Naive Bayes - stacked on RF+XGB+SVC"},
            138: {"mdl": "RF", "name": "Random Forest"},
            166: {
                "mdl": "KNN_s",
                "name": "K Nearest Neighbors - stacked on RF+XGB+SVC",
            },
            191: {
                "mdl": "KNN_s",
                "name": "K Nearest Neighbors - stacked on RF+XGB+SVC",
            },
            203: {"mdl": "SVC", "name": "Support Vector Classifier"},
            253: {"mdl": "XGB", "name": "XGBoost"},
            287: {"mdl": "SVC", "name": "Support Vector Classifier"},
        }
    if "mdl_thresh" not in st.session_state:
        st.session_state.mdl_thresh = {
            "RF": 0.38,
            "XGB": 0.42,
            "SVC": 0.36,
            "KNN_s": 0.44,
            "NB_s": 0.16,
        }


# ------------------------------------------
def create_menu():
    # sidebar menu
    st.sidebar.image("images/Logo_wide.png", width=200, use_column_width="never")
    st.sidebar.page_link("app.py", label="Capstone Project")
    st.sidebar.page_link("pages/about_ASD.py", label="About ASD")
    st.sidebar.page_link("pages/dataset_features.py", label="Dataset & Features")
    st.sidebar.page_link("pages/models.py", label="Models")
    st.sidebar.page_link("pages/about_ET.py", label="About 'EyeTism'")
    st.sidebar.page_link("pages/about_us.py", label="About Us")
    st.sidebar.markdown("---")
    st.sidebar.markdown("# Diagnostics")
    st.sidebar.page_link("pages/patients.py", label="Patients")
    st.sidebar.page_link("pages/recording.py", label="Record")
    st.sidebar.page_link("pages/evaluate.py", label="Evaluate")
    ut.h_spacer(7, sb=True)
    st.sidebar.markdown("---")
    st.session_state.debug = st.sidebar.checkbox(label="show debug info", value=False)


# ------------------------------------------
def default_style():
    css = """
    <style>
        [data-testid="stSidebar"]{
            min-width: 250px;
            max-width: 250px;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ------------------------------------------
def h_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write("\n")
        else:
            st.write("\n")


# ------------------------------------------
def nice_date(f):
    s = f.split(".")[0]
    d = datetime.datetime.strptime(s, "%Y-%m-%d_%H-%M-%S")
    return datetime.date.strftime(d, "%A, %d.%m.%Y / %H:%M:%S")


# ------------------------------------------
def ugly_date(f):
    d = datetime.datetime.strptime(f, "%A, %d.%m.%Y / %H:%M:%S")
    return datetime.date.strftime(d, "%Y-%m-%d_%H-%M-%S")
