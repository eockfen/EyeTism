import streamlit as st
import utils as ut

# load default style settings
ut.default_style()

# sidebar menu
ut.create_menu()

st.title("Capstone Project")
st.markdown("---")





st.write("""This tool stems from the Capstone Project from the neuefische Data Science Bootcamp HH-24-1.
As a group, we committed ourselves to dive deep into the potential of AI and machine learning in the healthcare sector.
         
The Capstone Project itself was inspired by the [Grand Challenge Saliency4ASD: Visual attention modeling for Autism Spectrum Disorder](https://www.sciencedirect.com/science/article/pii/S0923596520302150)
organized by the IEEE International Conference on Multimedia and Expo (ICME) 2019.
         
The primary goals of this initiative were twofold:
         
First, to focus and coordinate the efforts of the visual attention modeling community towards addressing a healthcare-related societal challenge,
and second, to furnish essential datasets and tools necessary for the advancement and assessment of visual attention (VA) models.
         
ASD and typically developed (TD) children exhibit differing gaze behaviors, noticeable in social and interactive settings and studied via eye tracking experiments.
During the challenge, two research questions were postulated to be tackled during the competition:""")



with st.expander("**Track 1**: Given an image, to predict the saliency maps that fit gaze behavior of people with ASD"):
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:     
        

        st.image("images/Track1.png", width=400)


    st.write("""Various models aim to predict the visual attention features of images for healthy humans, but few address individuals with impairments like ASD. 
            The objective of this track is to develop machine learning models that accurately predict where individuals with ASD focus their visual attention. 
            Such models are crucial for creating customized Computer-Human Interfaces (CHIs) tailored for individuals with ASD, potentially enhancing accessibility and usability for this group.""")
         
st.text("")
    
with st.expander("**Track 2 - Capstone Project**: Given an image and the sequence of fixations of one observer, to classify if he/she is a person with ASD or TD."):
    left_co, middle_co, cent_co,last_co = st.columns(4)
    with middle_co:

        st.image("images/Track2.png", width=600)


    st.write("""Track 2 involves classifying individuals as either having Autism Spectrum Disorder (ASD) or being typically developing (TD) based on their gaze patterns.
            By this, learning the systematic VA patterns of children with ASD can help to identify those very early during the childs development, 
            providing early treatment, support and developmental therapy for affected families.
            WHile the scientific community tries to create a single model to improve the overall metrics of the models, our Capstone project focuses on 
            the deployment of  a diagnostic tool to screen for ASD
        
            """)
         
# st.header('abstract')
# st.write(""" Autism spectrum disorder (ASD) is developmental disability, where affected individuals exhibit difficulties in social situations and communication, 
#          and learning. Early diagnosis is crucial for affected to reach their potential in child development, but also very difficult to diagnose.
#          ASD affected individuals also differ by their gaze behaviour from typically developed (TD) children. 
#          Inspired by the Grand CHallenge Saliency4ASD, the Capstone Project "EyeTism" addressed the described problematic above and 
#          uses machine learning on available Eye Tracking data of high-functioning ASD and TD children to develop a diagnostic tool to screen for ASD indication 
#          by their respective gaze behaviour.
#          """)
st.markdown("---")
st.subheader("""Acknowledgements""")

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("images/neuefische.png")
st.text("")
st.text("")
st.write("""We would like to express our sincere appreciation to the Data Science Bootcamp team at neuefische GmbH 
         for their invaluable support and guidance throughout our journey.
          A special thank you goes to Lina Willing, Aljoscha Wilhelm, and Nico Steffen for their outstanding coaching and mentorship. We also extend our gratitude to all the other coaches who contributed their expertise and insights, enriching our learning experience. 
         Your dedication and support have been instrumental in our growth and success.""")