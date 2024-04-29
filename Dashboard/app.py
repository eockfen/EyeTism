import os
import streamlit as st
from scripts import utils as ut

# setup vars, menu, style, and so on --------------------
st.set_page_config(layout="wide")  # need to be first 'st' command !!!
ut.init_vars()
ut.default_style()
ut.create_menu()

# home --------------------------------------------------
st.image(os.path.join("content", "Logo_Eyetism.png"), use_column_width="auto")
st.text("")
st.write(
    """Welcome to our dashboard. This site displays the results of our Capstone
Project from the neuefische Data Science Bootcamp HH-24-1.

As a group, we committed ourselves to dive deep into the potential of AI and
machine learning in the healthcare sector.

The Capstone Project itself was inspired by the [**Grand Challenge Saliency4ASD:
Visual attention modeling for Autism Spectrum Disorder**][1]
organized by the IEEE International Conference on Multimedia and Expo (ICME) 2019.

The primary goals of this initiative were twofold:

- to focus and coordinate the efforts of the visual attention modeling
community towards addressing a healthcare-related societal challenge, and
- to furnish essential datasets and tools necessary for the advancement and
assessment of visual attention (VA) models.

ASD and typically developed (TD) children exhibit differing gaze behaviors,
noticeable in social and interactive settings and studied via eye tracking
experiments. During the challenge, two research questions were postulated to be
tackled during the competition:

[1]: https://www.sciencedirect.com/science/article/pii/S0923596520302150"""
)


with st.expander(
    """**Track 1**: Given an image, to predict the saliency maps that fit gaze
    behavior of people with ASD""",
    expanded=True,
):
    left_co, right_co = st.columns([2, 3])
    with right_co:
        st.image(os.path.join("content", "Track1.png"), width=300)

    with left_co:
        st.write(
            """Various models aim to predict the visual attention features of images
for healthy humans, but few address individuals with impairments like ASD.

The objective of this track is to develop machine learning models that
accurately predict where individuals with ASD focus their visual attention.
Such models are crucial for creating customized Computer-Human Interfaces
(CHIs) tailored for individuals with ASD, potentially enhancing
accessibility and usability for this group."""
        )

ut.h_spacer(1)

with st.expander(
    """**Track 2 - Capstone Project**: Given an image and the sequence of
    fixations of one observer, to classify if he/she is a person with ASD or TD.""",
    expanded=True,
):
    left_co, right_co = st.columns([2, 3])
    with right_co:
        st.image(os.path.join("content", "Track2.png"), width=400)

    with left_co:
        st.write(
            """Track 2 involves classifying individuals as either having Autism
Spectrum Disorder (ASD) or being typically developing (TD) based on
their gaze patterns.

By this, learning the systematic VA patterns of
children with ASD can help to identify those very early during the
childs development, providing early treatment, support and developmental
therapy for affected families. WHile the scientific community tries to
create a single model to improve the overall metrics of the models, our
Capstone project focuses on the deployment of  a diagnostic tool to
screen for ASD."""
        )

st.markdown("---")
st.subheader("""Acknowledgements""")

left_co, cent_co = st.columns([3, 2])
with left_co:
    st.write(
        """We would like to express our sincere appreciation to the Data Science
team at [neuefische GmbH](https://neuefische.de) for their invaluable support
and guidance throughout our journey.

A special thank you goes to **Lina Willing**, **Aljoscha Wilhelm**, and **Nico Steffen**
for their outstanding coaching and mentorship. We also extend our gratitude
to all the other coaches who contributed their expertise and insights,
enriching our learning experience. Your dedication and support have been
instrumental in our growth and success."""
    )

with cent_co:
    st.image(os.path.join("content", "neuefische.png"), width=400)
