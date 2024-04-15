import streamlit as st
import utils as ut

# setup vars, menu, style, and so on --------------------
st.set_page_config(layout="wide")  # need to be first 'st' command !!!
ut.default_style()
ut.create_menu()

# home --------------------------------------------------
st.image('images/Logo_Eyetism.png', use_column_width="auto")
