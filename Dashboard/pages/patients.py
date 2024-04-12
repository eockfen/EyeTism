import streamlit as st
import utils as ut

# load default style settings
ut.default_style()

# sidebar menu
ut.create_menu()

st.title("Manage Patients")
st.markdown("---")
