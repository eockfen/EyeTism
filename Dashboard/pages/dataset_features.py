import streamlit as st
import utils as ut
import pandas as pd
import numpy as np
import os
import random

# load default style settings
ut.default_style()

# sidebar menu
ut.create_menu()

st.title("Dataset & Features")
st.markdown("---")
## write everythiing about the dataset and the initial features. 


st.header('The Dataset and Features')

st.subheader('The Dataset')
st.write("""The dataset was provided by #### and is freely available for download.
        Contact information, datasets and further information are avaialble [here](https://saliency4asd.ls2n.fr/).""")

st.text('**The images**')
st.write(""" The dataset for consists of 300 images from the MIT1003 dataset. In several sessions, each image was displayed for 3 seconds followed by an grey image. For each picture the
          Gaze behaviour was recorderd for  14 typically developing children and 14 high-functioning ASD children in the age range of 5 to 12.
         The resulting scanpaths weree aggregateted per patient class and image, resulting in the following. 
            """)

st.text('Random selection of images')


# Function to load and display images from a folder
def display_images(folder_path, start_index, num_images=6):
    col1, col2, col3 = st.columns(3)
    for i in range(start_index, start_index + num_images):
        image_file = f"{i+1}.png"  # Assuming image filenames are 1.png, 2.png, etc.
        image_path = os.path.join(folder_path, image_file)
        if os.path.exists(image_path):
            if i % 3 == 0:
                col1.image(image_path, caption=image_file, use_column_width=True)
            elif i % 3 == 1:
                col2.image(image_path, caption=image_file, use_column_width=True)
            else:
                col3.image(image_path, caption=image_file, use_column_width=True)
        else:
            break  # Stop if the image file doesn't exist

# Path to the folder containing your images
folder_path = "../data/Saliency4ASD/TrainingData/Images"

# Slider to control the starting index of images
start_index = st.slider("Select starting index", 0, len(os.listdir(folder_path)) - 6, 0, step=6)

# Display the images
display_images(folder_path, start_index)







st.text("The scanpaths")
st.write(""" The scanpaths contain information about where, how long, and how often  the participant has looked 
         during the 3 second period, where the picture has been displayed. 
         The position of the eye gaze is depicted as x and y coordinates on the pixels of the image along with the duration of fixation in milliseconds, here as a fixation points
         The movement between two fixation points is saccades. 
         Along with these both metrics, a multitude of features can be calculated... 
         For more, see the section on 
         
          """ )
st.write(""" Depicted below is the workflow for data collection for one example image """)

with st.expander('Image'):
    st.image('../data/Saliency4ASD/TrainingData/Images/1.png')



col1, col2 = st.columns(2)
with col1:
    df_TD1 = pd.read_csv('../data/Saliency4ASD/TrainingData/TD/TD_scanpath_1.txt')
    st.dataframe(df_TD1.style.apply(lambda row: ['background-color: yellow' if row['Idx'] == 0 else '' for _ in row], axis=1))

    #st.dataframe(df_TD1.style.apply(lambda x: ['background-color: yellow' if x.name == 0 else '' for i in x], axis=1))


with col2:
    df_ASD1 = pd.read_csv('../data/Saliency4ASD/TrainingData/ASD/ASD_scanpath_1.txt')
    st.dataframe(df_ASD1.style.apply(lambda row: ['background-color: yellow' if row['Idx'] == 0 else '' for _ in row], axis=1))
st.write("""For this image are depicted the agreggated scanpaths for one image.
         Each row with '0' indicates the beginning of the gaze for a new participant.
         """)
    #st.dataframe(df_ASD1.style.apply(lambda x: ['background-color: yellow' if x.name == 0 else '' for i in x], axis=1))
    



st.write('''From the resulting scanpaths, the aggregated Fixation points on this specific image can be calculated.
         The points are not visible, so better zoom in on the picture.''')

with col1: 
    st.markdown("<h2 style='text-align: center;'>TD</h2>", unsafe_allow_html=True)
    st.image('../data/Saliency4ASD/AdditionalData/TD_FixPts/1_f.png')

with col2: 
    st.markdown("<h2 style='text-align: center;'>ASD</h2>", unsafe_allow_html=True)
    st.image('../data/Saliency4ASD/AdditionalData/ASD_FixPts/1_f.png')

st.write(""" and from the fixpoints can be heatmaps calculated inf """)

with col1: 
    st.markdown("<h2 style='text-align: center;'>TD</h2>", unsafe_allow_html=True)
    st.image('../data/Saliency4ASD/AdditionalData/TD_HeatMaps/1_h.png')

with col2: 
    st.markdown("<h2 style='text-align: center;'>ASD</h2>", unsafe_allow_html=True)
    st.image('../data/Saliency4ASD/AdditionalData/ASD_HeatMaps/1_h.png')


st.subheader('The features')

st.write("https://www.sciencedirect.com/science/article/abs/pii/S0923596520302150")
