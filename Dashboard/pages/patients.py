import time
import streamlit as st
from scripts import utils as ut
from scripts import functions as fct

# setup vars, menu, style, and so on --------------------
ut.init_vars()
ut.default_style()
ut.create_menu()

# page style ---------------------------------------------
st.title("Manage Patients")
tab1, tab2 = st.tabs(
    [s.center(30, "\u2001") for s in ["List Patients", "Manage Patients"]]
)

# list all patients ---------------------------------------------
with tab1:
    st.subheader("List All Patients")

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

# manage patients --------------------------------------------------------
with tab2:
    # edit patient entries ---------------------------------
    st.subheader("Edit Patient Entries")

    with st.container(border=True):
        # show db
        st.session_state.pat_db_update = st.data_editor(
            st.session_state.pat_db[["id", "name", "age"]],
            use_container_width=False,
            hide_index=True,
            disabled=["id", "n_rec", "last_rec"],
            column_config={
                "id": st.column_config.Column(label="id", width=50),
                "name": st.column_config.TextColumn(
                    label="Name", width=250, max_chars=50, required=True
                ),
                "age": st.column_config.NumberColumn(
                    label="Age", width="small", min_value=1, max_value=99, required=True
                ),
            },
        )

        # submit button
        col_btn1, col_fdb1, _ = st.columns([1, 3, 4], gap="small")
        with col_btn1:
            submitted_updated = st.button("Save Changes", on_click=fct.update_pat_DB)

        # feedback
        with col_fdb1:
            container_edit = st.empty()

        # handle submission
        if submitted_updated:
            if st.session_state.pat_db_update.shape[0] > 0:
                st.session_state.flag_edit_ok = True
            else:
                st.session_state.flag_edit_err = True

    ut.h_spacer(3)

    col_new, col_del = st.columns([1, 1], gap="medium")
    with col_new:
        # add new patient entry ---------------------------------
        st.subheader("Create New Patient")

        with st.form("add_patient", clear_on_submit=True):
            # name
            name = st.text_input("Name:", value="", max_chars=50)

            # age
            _, col_age, _ = st.columns([0.01, 0.8, 0.19], gap="small")
            with col_age:
                age = st.slider("Age:", min_value=1, max_value=99)

            # submit button
            col_btn2, col_fdb2 = st.columns([1, 3], gap="small")
            with col_btn2:
                submitted_add = st.form_submit_button("Save Patient")

            # feedback
            with col_fdb2:
                container_add = st.empty()

            # handle submission
            if submitted_add:
                status = fct.add_pat(name, age)
                if status:
                    st.session_state.flag_add_ok = True
                else:
                    st.session_state.flag_add_err = True
                st.rerun()

    with col_del:
        # delete patient entry ---------------------------------
        st.subheader("Delete Patient")

        with st.form("del_patient", clear_on_submit=True):
            sel_patient = st.selectbox(
                "Select patient to delete:",
                st.session_state.patient_list,
                label_visibility="collapsed",
            )

            # submit button
            col_btn3, col_fdb3 = st.columns([1, 3], gap="small")
            with col_btn3:
                # submitted_delete = st.form_submit_button(
                #     "Delete Patient", on_click=fct.del_patient, args=(sel_patient,)
                # )
                submitted_delete = st.form_submit_button("Delete Patient")

            # feedback
            with col_fdb3:
                container_del = st.empty()

            # handle submission
            if submitted_delete:
                if st.session_state.patient_list != []:
                    fct.del_patient(sel_patient)
                    st.session_state.flag_del_ok = True
                else:
                    st.session_state.flag_del_err = True
                st.rerun()

    # display messages ---------------------------------
    if st.session_state.flag_edit_ok:
        st.session_state.flag_edit_ok = False
        container_edit.success("Database updated successfully.")
        time.sleep(2)
        container_edit.empty()

    if st.session_state.flag_edit_err:
        st.session_state.flag_edit_err = False
        container_edit.error("No Patients in Database.")
        time.sleep(2)
        container_edit.empty()

    if st.session_state.flag_add_ok:
        st.session_state.flag_add_ok = False
        container_add.success("New patient saved successfully.")
        time.sleep(2)
        container_add.empty()

    if st.session_state.flag_add_err:
        st.session_state.flag_add_err = False
        container_add.error("Please enter a 'Name'.")
        time.sleep(2)
        container_add.empty()

    if st.session_state.flag_del_ok:
        st.session_state.flag_del_ok = False
        container_del.success("Patient deleted successfully.")
        time.sleep(2)
        container_del.empty()

    if st.session_state.flag_del_err:
        st.session_state.flag_del_err = False
        container_del.error("No Patients in Database.")
        time.sleep(2)
        container_del.empty()

# ------------------------------------------------------------
if st.session_state.debug:
    st.session_state
