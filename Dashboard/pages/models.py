import streamlit as st
from scripts import utils as ut


# load default style settings
ut.default_style()

# sidebar menu
ut.create_menu()

st.title("Models")
st.markdown("---")


st.image("content/Workflow.png")
st.write(
    """This workflow depicts an overview over the architecture of our machine
    learning approach to be used for screening of ASD. Visible are the steps
    for several layers of feature engineering, the considerations of different
    machine learning model, as well as the last processing steps leading to the
    final evaluation. Each process can be seen in detail in our
    [Github repository](https://github.com/eockfen/EyeTism).
"""
)
