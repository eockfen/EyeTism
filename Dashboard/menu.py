import streamlit as st


def create_menu():
    # sidebar menu
    st.sidebar.page_link("app.py", label="Home")
    st.sidebar.text('Capstone Project')
    st.sidebar.text('Dataset & Features')
    st.sidebar.text('Models')
    st.sidebar.text('About EyeTism')
    st.sidebar.text('About us')
    st.sidebar.markdown('---')
    st.sidebar.markdown('# Diagnostics')
    st.sidebar.page_link("pages/patients.py", label="Patients")
    st.sidebar.page_link("pages/recording.py", label="Record")
    st.sidebar.page_link("pages/evaluate.py", label="Evaluate")
