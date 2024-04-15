import streamlit as st
import pandas as pd
import numpy as np
import glob
import os


# initialize session_state variabled ------------------------------------------
def init_vars():
    st.session_state.debug = True
    st.session_state.images = [207, 95, 203, 81, 271, 176, 193, 272]
    st.session_state.sp_idx_asd = [4, 0, 6, 1, 10, 3, 2, 7]
    st.session_state.sp_idx_td = [1, 7, 7, 7, 7, 7, 7, 2]

    DB = pd.read_csv(os.path.join("files", "patients.csv"))
    if "patient_db" not in st.session_state:
        st.session_state.pat_db = DB
    if "edited_patient_db" not in st.session_state:
        st.session_state.pat_db_update = DB.copy()


# ------------------------------------------
def create_menu():
    # sidebar menu
    st.sidebar.image("images/Logo_wide.png", width=200, use_column_width="never")
    st.sidebar.page_link("app.py", label="Capstone Project")
    st.sidebar.page_link("pages/dataset_features.py", label="Dataset & Features")
    st.sidebar.page_link("pages/models.py", label="Models")
    st.sidebar.page_link("pages/about_ET.py", label="About 'EyeTism'")
    st.sidebar.page_link("pages/about_us.py", label="About Us")
    st.sidebar.markdown("---")
    st.sidebar.markdown("# Diagnostics")
    st.sidebar.page_link("pages/patients.py", label="Patients")
    st.sidebar.page_link("pages/recording.py", label="Record")
    st.sidebar.page_link("pages/evaluate.py", label="Evaluate")


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
def load_scanpath(kind: str, img: int, sp_idx: int) -> list:
    """load scanpath txt file and split individual scanpaths

    Args:
        file (str): path to scanpath txt file

    Returns:
        list: list of pd.DataFrames for each individual scanpath
    """

    # path to scanpath file
    sp_path = os.path.join(
        "..",
        "data",
        "Saliency4ASD",
        "TrainingData",
        kind,
        f"{kind}_scanpath_{img}.txt",
    )

    # read scanpat*.txt file
    sp = pd.read_csv(sp_path, index_col=None)
    sp.columns = map(str.strip, sp.columns)
    sp.columns = map(str.lower, sp.columns)

    starts = np.where(sp["idx"] == 0)[0]
    ends = np.append(starts[1:], len(sp))
    all_sp = [sp[start:end] for start, end in zip(starts, ends)]

    return all_sp[sp_idx]


# ------------------------------------------
def update_DB_recordings():
    for (_, r) in st.session_state.pat_db.iterrows():
        id = r["id"]
        recs = [f.split('_')[1:] for f in glob.glob("recordings/*.csv") if f"id-{id}_" in f]
        n_rec = len(recs)
        print(f"{id}  ->  {n_rec}")
