import streamlit as st
import utils as ut
import pandas as pd
import os
import imageio.v3 as iio
import image_processing as ip
import matplotlib.pyplot as plt

# setup vars, menu, style, and so on --------------------
ut.init_vars()
ut.default_style()
ut.create_menu()

# variables & paths -----------------------------------------------------------
# Path to the folder containing images
path_images = os.path.join("content/images")
path_sp = os.path.join("content", "scanpaths")
path_sal_deepgaze = os.path.join("content", "sal_pred", "DeepGazeIIE")
path_sal_sam_resnet = os.path.join("content", "sal_pred", "sam_resnet")


# functions -------------------------------------------------------------------
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


# page ------------------------------------------------------------------------
st.title("Dataset & Features")
st.markdown("---")

# about the dataset -----------------------------------------------------------
st.markdown("## About the Dataset")
st.write(
    """The dataset has been provided by the organizing committee of the Grand
Challenge Saliency4ASD from Shanghai Jiao Tong University, China, and Université
de Nantes, France. It is readily available for download. For contact information,
access to the data, and further details, please visit the
[official website](https://saliency4asd.ls2n.fr/).

The dataset consists of 300 images from the MIT1003 dataset.  In multiple sessions,
14 typically developed (TD) children and 14 high-functioning ASD children,
aged 5 to 12 years, were instructed to view each image. Each image was displayed
for 3 seconds, followed by a 1-second grey image. The gaze behavior of the
participants was recorded. The resulting scanpaths were aggregated per patient
class and image, as depicted in the schema below."""
)

# Slider to control the starting index of images
ut.h_spacer(1)
start_index = st.slider(
    "Select starting image", 0, len(os.listdir(path_images)) - 6, 0, step=6
)

# Display the images
ut.h_spacer(1)
display_images(path_images, start_index)

# detailled overview of features per image ------------------------------------
st.divider()
st.markdown("## About the Features")

# Select an image
col_TD, col_ASD = st.columns([0.3, 0.7])
with col_TD:
    st.markdown(
        """For the selected image the following further details and
                features are shown below:

- individual gaze scanpaths
- Heatmaps, for ASD & TD
- detected objects & faces
- saliency predictions"""
    )

    image_files = sorted(os.listdir(path_images), key=lambda x: int(x.split(".")[0]))
    show_image = st.selectbox("Select image:", image_files)
    print(show_image)
with col_ASD:
    img = iio.imread(os.path.join(path_images, show_image))
    w, h = img.shape[1], img.shape[0]
    # Display the selected image
    st.image(img, width=int(287 * w / h))

# scanpaths -------------------------------------
ut.h_spacer(1)
st.markdown("### Scanpaths")
st.write(
    """ The scanpaths contain information about where, how long, and how often
the participant has looked during the 3-second period when the picture was
displayed. The position of the eye gaze is depicted as x and y coordinates on
the pixels of the image, along with the duration of fixation in milliseconds,
representing **fixation points**. The movement between two fixation points is
called **saccades**. Along with these metrics, a multitude of features can be
calculated, which are explained in detail in our
[Github repository](https://github.com/eockfen/EyeTism).

Furthernore, it's important to note that the order of scanpaths does not
correspond to the order of individual participants in the experiments. As a
result, it is not feasible to attribute gaze patterns to specific individuals."""
)

with st.container(border=True):
    # load df
    df_TD = pd.read_csv(os.path.join(path_sp, "TD_" + show_image[:-4] + ".txt"))
    df_TD.columns = map(str.strip, df_TD.columns)
    df_TD.columns = map(str.lower, df_TD.columns)
    df_TDi = df_TD[df_TD["idx"] == 0].index.tolist()

    df_ASD = pd.read_csv(os.path.join(path_sp, "ASD_" + show_image[:-4] + ".txt"))
    df_ASD.columns = map(str.strip, df_ASD.columns)
    df_ASD.columns = map(str.lower, df_ASD.columns)
    df_ASDi = df_ASD[df_ASD["idx"] == 0].index.tolist()

    col_TD, col_ASD = st.columns(2)
    with col_TD:
        st.markdown("#### TD")
        td_list = st.selectbox(
            "Select individual gaze scanpath:",
            [f"Individual  {i}" for i in range(len(df_TDi))],
        )
        td_idx = int(td_list.split(" ")[-1])

        col_df, col_sp = st.columns([0.35, 0.65])
        with col_df:
            # scanpath data
            st.dataframe(
                df_TD.iloc[df_TDi[td_idx]:df_TDi[td_idx + 1]][["x", "y", "duration"]],
                hide_index=True,
            )

        with col_sp:
            # create plt
            fig_td = plt.figure(
                figsize=(round(img.shape[1] * 0.01), round(img.shape[0] * 0.01)),
                frameon=False,
            )
            ax = plt.gca()
            ax.set_axis_off()
            ax.imshow(img)

            # add scanpath overlay
            fig_td = ip.overlay_scanpath(
                fig_td,
                df_TD.iloc[df_TDi[td_idx]:df_TDi[td_idx + 1]][
                    ["idx", "x", "y", "duration"]
                ],
            )

            # plot
            st.pyplot(fig_td)

    with col_ASD:
        st.markdown("#### ASD")
        asd_list = st.selectbox(
            "Select individual gaze scanpath:",
            [f"Individual  {i}" for i in range(len(df_ASDi))],
        )
        asd_idx = int(asd_list.split(" ")[-1])

        col_df, col_sp = st.columns([0.35, 0.65])
        with col_df:
            # scanpath data
            st.dataframe(
                df_ASD.iloc[df_ASDi[asd_idx]:df_ASDi[asd_idx + 1]][
                    ["x", "y", "duration"]
                ],
                hide_index=True,
            )

        with col_sp:
            # create plt
            fig_asd = plt.figure(
                figsize=(round(img.shape[1] * 0.01), round(img.shape[0] * 0.01)),
                frameon=False,
            )
            ax = plt.gca()
            ax.set_axis_off()
            ax.imshow(img)

            # add scanpath overlay
            fig_asd = ip.overlay_scanpath(
                fig_asd,
                df_ASD.iloc[df_ASDi[asd_idx]:df_ASDi[asd_idx + 1]][
                    ["idx", "x", "y", "duration"]
                ],
            )

            # plot
            st.pyplot(fig_asd)

# heatmaps --------------------------------------
ut.h_spacer(1)
st.markdown("### Visual Heatmaps")
st.write(
    """From the resulting scanpaths for each patient class, the aggregated
heatmaps can be obtained for this specific image.They provide an idea where
the gaze focuses during observation."""
)

# Display heatmaps
with st.container(border=True):

    col_TD, col_ASD = st.columns(2)
    with col_TD:
        st.markdown("#### TD")
        fig_hm_td = ip.create_heatmap(int(show_image.split(".")[0]))
        st.pyplot(fig_hm_td)

    with col_ASD:
        st.markdown("#### ASD")
        fig_hm_asd = ip.create_heatmap(int(show_image.split(".")[0]))
        st.pyplot(fig_hm_asd)

# saliency --------------------------------------
ut.h_spacer(1)
st.markdown("### Saliency Predictions")
st.write(
    """Images can also be feed to _Visual Attentive Models_, which try to predict
the saliency of of the image contents, which usually are drawing the visual
attention of humans. These in turn can be compared to our actual scanpaths and
taken into account as additional features. For our tool, we decided on the most
actual models available,
[DeepGaze IIe](https://github.com/mpatacchiola/deepgaze) and
[SAM_ResNet](https://github.com/marcellacornia/sam). There are several models
available and tested in a standardized way, a selection is provided
[here](https://saliency.tuebingen.ai/datasets.html)."""
)

with st.container(border=True):
    col_img, col_dg, col_sam = st.columns(3)

    with col_img:
        st.markdown("#### Original Image")
        st.image(os.path.join(path_images, show_image))

    with col_dg:
        st.markdown("#### DeepGazeIIE")
        st.image(os.path.join(path_sal_deepgaze, show_image))

    with col_sam:
        st.markdown("#### SAM ResNET")
        st.image(os.path.join(path_sal_sam_resnet, show_image[:-4] + ".jpg"))

# objects & faces -------------------------------
ut.h_spacer(1)
st.markdown("### Object and Face Recognition")
st.write(
    """Visual attention is drawn to various elements within images, such as
expansive landscapes or highly concentrated objects like people, faces, animals,
buildings, and everyday items. With
[Mediapipe](https://developers.google.com/mediapipe/solutions/vision/object_detector),
the objects can be attributed to the fixation points."""
)

with st.container(border=True):
    col_obj, col_face = st.columns(2)
    with col_obj:
        st.markdown("#### Objects")

        # create plt
        fig_obj = plt.figure(
            figsize=(round(img.shape[1] * 0.01), round(img.shape[0] * 0.01)),
            frameon=False,
        )
        ax = plt.gca()
        ax.set_axis_off()
        ax.imshow(img)

        # add scanpath overlay
        fig_obj = ip.overlay_objects(
            fig_obj, st.session_state.loaded_objects[int(show_image.split(".")[0])], lw=7, lbl=False
        )

        # plot
        st.pyplot(fig_obj)

    with col_face:
        st.markdown("#### Faces")

        # create plt
        fig_fcs = plt.figure(
            figsize=(round(img.shape[1] * 0.01), round(img.shape[0] * 0.01)),
            frameon=False,
        )
        ax = plt.gca()
        ax.set_axis_off()
        ax.imshow(img)

        # add scanpath overlay
        fig_fcs = ip.overlay_faces(
            fig_fcs, st.session_state.loaded_faces[int(show_image.split(".")[0])], lw=7
        )

        # plot
        st.pyplot(fig_fcs)

ut.h_spacer(1)
st.markdown("---")
st.write(
    """All of these steps represent a part of the features, which will be feed
into our model. For a detailed description of our feature engineering, please
see our [Github repository](https://github.com/eockfen/EyeTism)"""
)

# ------------------------------------------------------------
if st.session_state.debug:
    st.session_state
