import os
import streamlit as st
import pandas as pd
import numpy as np
import utils as ut
import time


# setup vars, menu, style, and so on --------------------
ut.init_vars()
ut.default_style()
ut.create_menu()


# functions ---------------------------------------------
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
        print("empty db")
        st.session_state.pat_db = new_pat
    else:
        print(f"{len(dbl)} patients in db")
        new_pat["id"] = max(st.session_state.pat_db["id"]) + 1
        st.session_state.pat_db = pd.concat(
            [st.session_state.pat_db, new_pat], axis=0, ignore_index=True
        )

    # save updated DB
    st.session_state.pat_db.to_csv(os.path.join("files", "patients.csv"), index=False)

    return True


def update_DB():
    st.session_state.pat_db = st.session_state.pat_db_update
    st.session_state.pat_db.to_csv(os.path.join("files", "patients.csv"), index=False)

    return True


def del_patient(x):
    del_id = int(x.split(":")[0])
    print(" id to delete: " + str(del_id))
    del_idx = st.session_state.pat_db.id == del_id
    print(del_idx)
    print(st.session_state.pat_db.loc[del_idx, :])
    print(st.session_state.pat_db.loc[np.invert(del_idx), :])
    st.session_state.pat_db = st.session_state.pat_db.loc[np.invert(del_idx), :]
    st.session_state.pat_db.to_csv(os.path.join("files", "patients.csv"), index=False)

    return True


# page style ---------------------------------------------
st.title("Manage Patients")
tab1, tab2, tab3 = st.tabs(
    [s.center(18, "\u2001") for s in ["List Patients", "Edit Patients", "New Patient"]]
)

# list all patients ---------------------------------------------
with tab1:
    st.subheader("List all patients")

    # show db
    st.dataframe(
        st.session_state.pat_db,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.Column(label="id", width="small"),
            "name": st.column_config.Column(label="Name", width="medium"),
            "age": st.column_config.Column(label="Age"),
            "n_rec": st.column_config.Column(label="# of rec."),
            "last_rec": st.column_config.Column(label="last rec."),
        },
    )

# edit patients ---------------------------------------------
with tab2:
    # edit patient entries -----------
    st.subheader("Edit patient entries")

    # show db
    st.session_state.pat_db_update = st.data_editor(
        st.session_state.pat_db,
        use_container_width=True,
        hide_index=True,
        disabled=["id", "n_rec", "last_rec"],
        column_config={
            "id": st.column_config.Column(label="id", width="small"),
            "name": st.column_config.TextColumn(
                label="Name", width="medium", max_chars=50, required=True
            ),
            "age": st.column_config.NumberColumn(
                label="Age", min_value=1, max_value=99, required=True
            ),
            "n_rec": st.column_config.Column(label="# of rec."),
            "last_rec": st.column_config.Column(label="last rec."),
        },
    )

    # button
    updated = st.button("Save Changes", on_click=update_DB)

    # feedback
    container_update = st.empty()
    if updated:
        container_update.success("Database updated successfully.")
        time.sleep(1)
        container_update.empty()

    # delete patient entry -----------
    st.subheader("Delete patient entry")

    sel_patient = st.selectbox(
        "Select patient to delete:",
        [
            f"{int(r['id'])}: {r['name']} (age: {int(r['age'])})"
            for (_, r) in st.session_state.pat_db.iterrows()
        ],
        label_visibility="collapsed",
    )

    # button
    deleted = st.button("Delete Patient", on_click=del_patient, args=(sel_patient,))

    # feedback
    container_del = st.empty()
    if deleted:
        container_del.success("Patient deleted successfully.")
        time.sleep(1)
        container_del.empty()


# add new patients ---------------------------------------------
with tab3:
    st.subheader("Create new patient entry")

    with st.form("add_patient", clear_on_submit=True):
        # name
        name = st.text_input("Name:", value="", max_chars=50)

        # age
        _, col_age, _ = st.columns([0.03, 0.5, 0.46], gap="small")
        with col_age:
            age = st.slider("Age:", min_value=1, max_value=99)

        # submit button
        submitted = st.form_submit_button("Save Patient")

        # handle submission
        if submitted:
            status = add_pat(name, age)
            container_add = st.empty()
            if status:
                container_add.success("New patient saved successfully.")
                time.sleep(1)
                container_add.empty()
                st.rerun()
            else:
                container_add.error("... patient was not saved...")
                time.sleep(2)

# st.session_state
