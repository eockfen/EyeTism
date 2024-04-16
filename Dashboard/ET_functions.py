import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import utils as ut
import ET_features as feat

# import utils as ut


# -----------------------------------------------------------------------------
# region PATIENT DATABASE -----------------------------------------------------
# -----------------------------------------------------------------------------
def add_pat(name, age):
    if name == "":
        return False

    # handle name
    name.strip()

    # new patient
    new_pat = pd.DataFrame(
        {"id": 1, "name": name, "age": int(age), "n_rec": 0, "last_rec": 0},
        index=[0],
    )

    # handle adding patient to DB
    dbl = list(st.session_state.pat_db.index)
    if len(dbl) == 0:
        if st.session_state.debug:
            print("empty db")
        st.session_state.pat_db = new_pat
    else:
        if st.session_state.debug:
            print(f"{len(dbl)} patients in db")
        new_pat["id"] = max(st.session_state.pat_db["id"]) + 1
        st.session_state.pat_db = pd.concat(
            [st.session_state.pat_db, new_pat], axis=0, ignore_index=True
        )

    # save updated DB
    st.session_state.pat_db.to_csv(os.path.join("files", "patients.csv"), index=False)

    st.session_state.patient_list = [
        f"{int(r['id'])}: {r['name']} (age: {int(r['age'])})"
        for (_, r) in st.session_state.pat_db.iterrows()
    ]
    return True


def update_pat_DB():
    st.session_state.pat_db = st.session_state.pat_db_update
    st.session_state.pat_db.to_csv(os.path.join("files", "patients.csv"), index=False)

    return True


def del_patient(x):
    print(x)
    del_id = int(x.split(":")[0])
    del_idx = st.session_state.pat_db.id == del_id
    st.session_state.pat_db = st.session_state.pat_db.loc[np.invert(del_idx), :]
    st.session_state.pat_db.to_csv(os.path.join("files", "patients.csv"), index=False)

    if st.session_state.debug:
        print(" id to delete: " + str(del_id))
        print(del_idx)
        print(st.session_state.pat_db.loc[del_idx, :])
        print(st.session_state.pat_db.loc[np.invert(del_idx), :])

    return True


# -----------------------------------------------------------------------------
# region RECORDING ------------------------------------------------------------
# -----------------------------------------------------------------------------
def example_load_scanpath(kind: str, img: int, sp_idx: int) -> list:
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


def load_scanpath(sp_path: str) -> list:
    # read scanpat*.txt file
    sp = pd.read_csv(sp_path, index_col=None)
    sp.columns = map(str.strip, sp.columns)
    sp.columns = map(str.lower, sp.columns)

    starts = np.where(sp["idx"] == 0)[0]
    ends = np.append(starts[1:], len(sp))
    all_sp = [sp[start:end] for start, end in zip(starts, ends)]

    return all_sp


def save_recording(pat, kind):
    kind = "ASD" if kind == "ASD" else "TD"

    # patient, img & scanpath vars
    pat_id = int(pat.split(":")[0])
    images = st.session_state.opt["images"]
    sps = (
        st.session_state.opt["sp_idx_asd"]
        if kind == "ASD"
        else st.session_state.opt["sp_idx_td"]
    )

    # get scanpaths
    df_sp = None
    for img in images:
        sp = example_load_scanpath(kind, img, sps[img])
        sp["img"] = img
        df_sp = pd.concat([df_sp, sp], ignore_index=True)

    # save csv file
    dt = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S.csv")
    name = f"id-{pat_id}_{dt}"
    df_sp.to_csv(os.path.join("recordings", f"{name}"), index=False)

    # save session state
    if st.session_state.last_saved_recording is None:
        st.session_state.last_saved_recording = name


def update_rec_DB():
    tmp = st.session_state.pat_db.copy()
    for p in st.session_state.patient_list:
        id = int(p.split(":")[0])

        recs = st.session_state.rec_db[p][0]
        n_rec = len(recs)

        tmp.loc[tmp["id"] == id, "n_rec"] = n_rec
        if n_rec > 0:
            last_rec = st.session_state.rec_db[p][1][-1]
            tmp.loc[tmp["id"] == id, "last_rec"] = last_rec
        else:
            tmp.loc[tmp["id"] == id, "last_rec"] = "---"

    # save updated DB
    tmp.to_csv(os.path.join("files", "patients.csv"), index=False)


# -----------------------------------------------------------------------------
# region ANALYSIS -------------------------------------------------------------
# -----------------------------------------------------------------------------
def extract_features():
    # filename
    id = st.session_state.eval_pat.split(":")[0]
    rec_date = ut.ugly_date(st.session_state.eval_meas)
    csv_file = os.path.join("recordings", f"id-{id}_{rec_date}.csv")

    # curdir = os.path.dirname(__file__)
    # csv_file = os.path.join(curdir, "recordings", "id-1_2024-01-08_17-22-45.csv")

    # extract features
    df = feat.scanpath(csv_file)
    df_sal = feat.saliency(csv_file)
    df_obj = feat.objects(csv_file)

    df = df.merge(df_sal, on="img")
    df = df.merge(df_obj, on="img")

    return df


def clean_features(df):
    # set id as index
    df = df.set_index("img", drop=True)
    df = df.drop(columns=[col for col in df.columns if "_obj" in col])  # drop 'object' columns
    df = df[df["sp_fix_duration_ms_total"] <= 5000]

    return df


# --- if script is run by it's own --------------------------------------------
if __name__ == "__main__":
    curdir = os.path.dirname(__file__)

    extract_features()
