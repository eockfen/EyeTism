import streamlit as st
import utils as ut
import pandas as pd
import numpy as np
import os


# load default style settings
ut.default_style()

# sidebar menu
ut.create_menu()

st.title("Dataset & Features")
st.markdown("---")
## write everythiing about the dataset and the initial features. 


st.header('The Dataset and Features')

st.subheader('About the Data')
place = st.text("")

st.write("""
The dataset has been provided by the organizing committee of the Grand Challenge Saliency4ASD from Shanghai Jiao Tong University, China, and Université de Nantes, France. It is readily available for download. For contact information, 
         access to the data, and further details, please visit the [official website](https://saliency4asd.ls2n.fr/).

The dataset consists of 300 images from the MIT1003 dataset. 
         In multiple sessions, 14 typically developed (TD) children and 14 high-functioning ASD children, aged 5 to 12 years, were instructed to view each image. 
         Each image was displayed for 3 seconds, followed by a 1-second grey image.
         The gaze behavior of the participants was recorded. The resulting scanpaths were aggregated per patient class and image, as depicted in the schema below.
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
folder_path = "images/Saliency4ASD/TrainingData/Images"

# Slider to control the starting index of images
st.text("")
start_index = st.slider("Select starting index", 0, len(os.listdir(folder_path)) - 6, 0, step=6)
st.text("")
# Display the images
display_images(folder_path, start_index)

st.markdown("**Scanpaths**")

st.write(""" 
The scanpaths contain information about where, how long, and how often the participant has looked during the 3-second period when the picture was displayed. 
         The position of the eye gaze is depicted as x and y coordinates on the pixels of the image, along with the duration of fixation in milliseconds, 
         representing **fixation points**. The movement between two fixation points is called **saccades**. Along with these metrics, a multitude of features can be calculated,
          which are explained in detail in our (repository)[add Link].
""")


# Path to the folder containing images
image_folder = "images/Saliency4ASD/TrainingData/Images/"
# Path to the folder containing scanpaths
scanpath_folder = "images/Saliency4ASD/TrainingData/"
# Path to the folder containing fixation points
fixpts_folder = "images/Saliency4ASD/AdditionalData/"
# Path to the folder containing heatmaps
heatmaps_folder = "images/Saliency4ASD/AdditionalData/"
saliency_deepgaze_folder = "images/saliency_predictions/DeepGazeIIE/"
saliency_sam_resnet_folder = "images/saliency_predictions/sam_resnet/"
object_det_folder = "images/object_dt/"
object_sp_folder = "images/individual/"
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
st.text("")

st.markdown("**Heatmaps**")

st.write('''From the resulting scanpaths for each patient class, the aggregated heatmaps can be obtained for this specific image.
         They provide an idea where the gaze focuses during observation.
        ''')


# Display heatmaps

with st.expander('Show Heatmaps'):
   
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("<h2 style='text-align: center;'>TD</h2>", unsafe_allow_html=True)
        st.image(os.path.join(heatmaps_folder, "TD_HeatMaps/" + selected_image[:-4] + "_h.png"))

    with col6:
        st.markdown("<h2 style='text-align: center;'>ASD</h2>", unsafe_allow_html=True)
        st.image(os.path.join(heatmaps_folder, "ASD_HeatMaps/" + selected_image[:-4] + "_h.png"))
st.text("")
st.markdown('**Object and Face recognition**')
st.write("""
Visual attention is drawn to various elements within images, 
         such as expansive landscapes or highly concentrated objects like people, faces, animals, buildings, and everyday items.
         With [Mediapipe](https://developers.google.com/mediapipe/solutions/vision/object_detector), the objects can be attributed to the fixation points.
""")

with st.expander('Show images with object and face recognition'):
    col7, col8_1, col8_2 = st.columns(3)
    with col7:
        st.markdown("<h4 style='text-align: center;'>Object Detection </h4>", unsafe_allow_html=True)
        st.image(os.path.join(object_det_folder, selected_image[:-4] + ".png"))

    with col8_1: 
        st.markdown("<h4 style='text-align: center;'>TD</h4>", unsafe_allow_html=True)
        td_folder = os.path.join(object_sp_folder, "TD")
        td_image_number = int(selected_image[:-4])
        td_image_files = sorted([file for file in os.listdir(td_folder) if file.startswith(f"td_{td_image_number:03}")])
        td_scanpaths = [f"Scanpath {i.split('_')[2][:-4]}" for i in td_image_files]  # Remove the '.png' extension
        selected_td_image = st.selectbox('Select TD gaze sequence:', td_scanpaths)
        st.image(os.path.join(td_folder, td_image_files[td_scanpaths.index(selected_td_image)]))

    with col8_2: 
        st.markdown("<h4 style='text-align: center;'>ASD</h4>", unsafe_allow_html=True)
        asd_folder = os.path.join(object_sp_folder, "ASD")
        asd_image_number = int(selected_image[:-4])
        asd_image_files = sorted([file for file in os.listdir(asd_folder) if file.startswith(f"asd_{asd_image_number:03}")])
        asd_scanpaths = [f"Scanpath {i.split('_')[2][:-4]}" for i in asd_image_files]  # Remove the '.png' extension
        selected_asd_image = st.selectbox('Select ASD gaze sequence:', asd_scanpaths)
        st.image(os.path.join(asd_folder, asd_image_files[asd_scanpaths.index(selected_asd_image)]))



# with st.expander('Show images with object and face recognition'):
#     col7, col8_1, col8_2 = st.columns(3)
#     with col7:
#         st.markdown("<h4 style='text-align: center;'>Object Detection </h4>", unsafe_allow_html=True)
#         st.image(os.path.join(object_det_folder, selected_image[:-4] + ".png"))

#     with col8_1: 
#         st.markdown("<h4 style='text-align: center;'>TD</h4>", unsafe_allow_html=True)
#         td_folder = os.path.join(object_sp_folder, "TD")
#         td_image_number = int(selected_image[:-4])
#         td_image_files = sorted([file for file in os.listdir(td_folder) if file.startswith(f"td_{td_image_number:03}")])
#         selected_td_image = st.selectbox('Select TD gaze sequence:', td_image_files)
#         st.image(os.path.join(td_folder, selected_td_image))

#     with col8_2: 
#         st.markdown("<h4 style='text-align: center;'>ASD</h4>", unsafe_allow_html=True)
#         asd_folder = os.path.join(object_sp_folder, "ASD")
#         asd_image_number = int(selected_image[:-4])
#         asd_image_files = sorted([file for file in os.listdir(asd_folder) if file.startswith(f"asd_{asd_image_number:03}")])
#         selected_asd_image = st.selectbox('Select ASD gaze sequence:', asd_image_files)
#         st.image(os.path.join(asd_folder, selected_asd_image))



        

st.text("")

st.markdown("**Saliency**")
st.write('''Images can also be feed  to Saliency Models, which try to predict the features drawing the visual attention of humans.
         These in turn can be compared to our actual scanpaths and taken into account as additional features. For our tool, we decided on the most actual models available, [DeepGaze IIe](https://github.com/mpatacchiola/deepgaze) 
         and [SAM_ResNet](https://github.com/marcellacornia/sam). There are several models available, a selection 
         is provided [here](https://saliency.tuebingen.ai/datasets.html).
      
        ''')

with st.expander('Show Saliency Maps'):
    col9, col10, col11 = st.columns(3)
    
    with col9:
        st.markdown("<h4 style='text-align: center;'>Original Image</h4>", unsafe_allow_html=True)
        st.image(os.path.join(image_folder, selected_image))

    with col10:
        st.markdown("<h4 style='text-align: center;'>DeepGazeIIe</h4>", unsafe_allow_html=True)
        st.image(os.path.join(saliency_deepgaze_folder, selected_image))

    with col11: 
        st.markdown("<h4 style='text-align: center;'>SAM_ResNET</h4>", unsafe_allow_html=True)
        st.image(os.path.join(saliency_sam_resnet_folder, selected_image[:-4] + ".jpg"))  # Adjust file extension to JPG



st.text("")
st.markdown("---")
st.write("""All of these steps represent a part of the features, which will be feed into our model.
         For a detailed description of our feature engineering, please see our [repository](Placeholer)""")