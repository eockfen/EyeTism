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

st.subheader('About the Data')

st.write("""
The dataset has been provided by the organizing committee of the Grand Challenge Saliency4ASD from Shanghai Jiao Tong University, China, and Université de Nantes, France.
It is readily available for download. For contact information, access to the data, and further details, please visit the [official website](https://saliency4asd.ls2n.fr/).

The dataset consists of 300 images from the MIT1003 dataset. In multiple sessions, 14 typically developed (TD) children and 14 high-functioning ASD children, aged 5 to 12 years, were instructed to view each image. 
Each image was displayed for 3 seconds, followed by a 1-second grey image. The gaze behavior of the participants was recorded. 
The resulting scanpaths were aggregated per patient class and image, as depicted in the schema below.
""")



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

st.text("**The Scanpaths**")

st.write(""" 
The scanpaths contain information about where, how long, and how often the participant has looked during the 3-second period when the picture was displayed. 
The position of the eye gaze is depicted as x and y coordinates on the pixels of the image, along with the duration of fixation in milliseconds, representing **fixation points**.
The movement between two fixation points is called **saccades**. 
Along with these metrics, a multitude of features can be calculated. as it will be demonstrated below (add Link)
""")



import os

# Path to the folder containing images
image_folder = "../data/Saliency4ASD/TrainingData/Images/"
# Path to the folder containing scanpaths
scanpath_folder = "../data/Saliency4ASD/TrainingData/"
# Path to the folder containing fixation points
fixpts_folder = "../data/Saliency4ASD/AdditionalData/"
# Path to the folder containing heatmaps
heatmaps_folder = "../data/Saliency4ASD/AdditionalData/"

# Get a list of all image files
image_files = sorted(os.listdir(image_folder), key=lambda x: int(x.split(".")[0]))

# Function to get corresponding file paths
def get_file_paths(image_name):
    image_path = os.path.join(image_folder, image_name)
    scanpath_path = os.path.join(scanpath_folder, "TD/TD_scanpath_" + image_name[:-4] + ".txt")
    fixpts_path = os.path.join(fixpts_folder, "TD_FixPts/" + image_name[:-4] + "_f.png")
    heatmap_path = os.path.join(heatmaps_folder, "TD_HeatMaps/" + image_name[:-4] + "_h.png")
    return image_path, scanpath_path, fixpts_path, heatmap_path

# Select an image
selected_image = st.selectbox('Select an image:', image_files)

# Display the selected image
with st.expander('Show Image'):
    st.image(os.path.join(image_folder, selected_image))

# Display aggregated scanpaths
#col1, col2 = st.columns(2)

st.write("""
    This section presents the aggregated scanpaths for the selected image. 
    Each row corresponds to the gaze behavior of a participant, with a '0' denoting the initiation of a new gaze sequence.
    However, it's important to note that the order of scanpaths does not correspond to the order of individual participants in the experiments.
    As a result, it is not feasible to attribute gaze patterns to specific individuals.
    """)

with st.expander('Show Scanpaths'):
    
    col1, col2 = st.columns(2)
    with col1:
        df_TD = pd.read_csv(os.path.join(scanpath_folder, "TD/TD_scanpath_" + selected_image[:-4] + ".txt"))
        st.markdown("<h2 style='text-align: center;'>TD</h2>", unsafe_allow_html=True)
        st.dataframe(df_TD.style.apply(lambda row: ['background-color: yellow' if row['Idx'] == 0 else '' for _ in row], axis=1))
    with col2:
        df_ASD = pd.read_csv(os.path.join(scanpath_folder, "ASD/ASD_scanpath_" + selected_image[:-4] + ".txt"))
        st.markdown("<h2 style='text-align: center;'>ASD</h2>", unsafe_allow_html=True)
        st.dataframe(df_ASD.style.apply(lambda row: ['background-color: yellow' if row['Idx'] == 0 else '' for _ in row], axis=1))


# Display aggregated fixation points

st.write('''From the resulting scanpaths, aggregated fixation points for this specific image can be calculated. 
            However, these points are not visible at this scale, so it's recommended to zoom in on the picture for better visibility.''')
with st.expander('Show Fixation Points'):
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<h2 style='text-align: center;'>TD</h2>", unsafe_allow_html=True)
        st.image(os.path.join(fixpts_folder, "TD_FixPts/" + selected_image[:-4] + "_f.png"))
    with col4:
        st.markdown("<h2 style='text-align: center;'>ASD</h2>", unsafe_allow_html=True)
        st.image(os.path.join(fixpts_folder, "ASD_FixPts/" + selected_image[:-4] + "_f.png"))


# Display heatmaps
st.write("Heatmaps provide a better overview of where the gaze focuses during observation.")
with st.expander('Show Heatmaps'):
   
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("<h2 style='text-align: center;'>TD</h2>", unsafe_allow_html=True)
        st.image(os.path.join(heatmaps_folder, "TD_HeatMaps/" + selected_image[:-4] + "_h.png"))

    with col6:
        st.markdown("<h2 style='text-align: center;'>ASD</h2>", unsafe_allow_html=True)
        st.image(os.path.join(heatmaps_folder, "ASD_HeatMaps/" + selected_image[:-4] + "_h.png"))




st.subheader('The features')

st.write("https://www.sciencedirect.com/science/article/abs/pii/S0923596520302150")
