import streamlit as st
from scripts import utils as ut

# load default style settings
ut.default_style()

# sidebar menu
ut.create_menu()

# vars
space = 2

# page -------------------
st.title("About us")
st.markdown("---")

st.subheader("Elena Ockfen")
c_left, c_right = st.columns([1, 4])
with c_left:
    st.image(image="content/us/EO.jpg", use_column_width="auto")
with c_right:
    st.markdown(
        """**Background:** Biologist (Immune-Oncology)

**Github:** [eockfen](https://github.com/eockfen)

**LinkedIn:** [E. Ockfen](https://www.linkedin.com/in/elena-ockfen-1a41251a2/)"""
    )

ut.h_spacer(space)
st.subheader("Mariano Santoro")
c_left, c_right = st.columns([1, 4])
with c_left:
    st.image(image="content/us/MS.png", use_column_width="auto")
with c_right:
    st.markdown(
        """**Background:** Biologist (Microbiology)

**Github:** [MSantoro87](https://github.com/MSantoro87)

**LinkedIn:** [M. Santoro](https://www.linkedin.com/in/mariano-santoro-090024151/)

**ORCID:** [0000-0003-2171-4785](https://orcid.org/0000-0003-2171-4785)"""
    )

ut.h_spacer(space)
st.subheader("Stefan Schlögl")
c_left, c_right = st.columns([1, 4])
with c_left:
    st.image(image="content/us/SS.png", use_column_width="auto")
with c_right:
    st.markdown(
        """**Background:** M. Sc. Computer Science & Marketing Expert

**Github:** [CrazyTrain93](https://github.com/CrazyTrain93)

**LinkedIn:** [S. Schlögl](https://www.linkedin.com/in/stefan-schloegl-003512299/)"""
    )

ut.h_spacer(space)
st.subheader("Dennis Dombrovskij")
c_left, c_right = st.columns([1, 4])
with c_left:
    st.image(image="content/us/DD.png", use_column_width="auto")
with c_right:
    st.markdown(
        """**Background:** Biologist (Molecular Biology)

**Github:** [DDombrovskij](https://github.com/DDombrovskij)"""
    )

ut.h_spacer(space)
st.subheader("Dr. Adam Zabicki")
c_left, c_right = st.columns([1, 4])
with c_left:
    st.image(image="content/us/AZ.jpg", use_column_width="auto")
with c_right:
    st.markdown(
        """**Background:** Neuroscience, Movement Science, Physics

**Github:** [azabicki](https://github.com/azabicki)

**LinkedIn:** [A Zabicki](https://www.linkedin.com/in/adam-zabicki/)

**ORCID:** [0000-0002-0527-2705](https://orcid.org/0000-0002-0527-2705)"""
    )
