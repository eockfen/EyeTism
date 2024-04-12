import streamlit as st


def create_menu():
    # sidebar menu
    st.sidebar.image('images/Logo_wide.png', width=200, use_column_width="never")
    st.sidebar.page_link("app.py", label="Capstone Project")
    st.sidebar.page_link("pages/dataset_features.py", label="Dataset & Features")
    st.sidebar.page_link("pages/models.py", label="Models")
    st.sidebar.page_link("pages/about_ET.py", label="About 'EyeTism'")
    st.sidebar.page_link("pages/about_us.py", label="About Us")
    st.sidebar.markdown('---')
    st.sidebar.markdown('# Diagnostics')
    st.sidebar.page_link("pages/patients.py", label="Patients")
    st.sidebar.page_link("pages/recording.py", label="Record")
    st.sidebar.page_link("pages/evaluate.py", label="Evaluate")


def default_style():
    css = '''
    <style>
        [data-testid="stSidebar"]{
            min-width: 250px;
            max-width: 250px;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
