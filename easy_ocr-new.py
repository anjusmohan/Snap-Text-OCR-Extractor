### Import necessary libraries

import easyocr as ocr  #OCR
import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 
import cv2
import math
from streamlit_extras.add_vertical_space import add_vertical_space

# Vertical sidebar contents
with st.sidebar: 
    st.title(":scroll:**TextSnap Extractor**:label::page_with_curl: :red[**'Pull Text from Pics'**]:lower_left_fountain_pen:  \n  :sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles:")
    st.markdown('''
    ## About
    This app is an :violet['OCR Extractor'] from images built using:
    - [Streamlit](https://streamlit.io/)
    - [EasyOCR](https://www.jaided.ai/easyocr/documentation/)
    - [Pillow](https://pillow.readthedocs.io/en/stable/)
     ''')
    add_vertical_space(0)
    st.write('❤️🤗   Made by Anju S Mohan   🤗❤️')
    
# Streamlit Title  
st.title(":books: Text Extraction from Images :bookmark:") #title

#subtitle
st.markdown("")

#image uploader
image = st.file_uploader(label = "Upload your image here", type=['png','jpg','jpeg','tiff','tif'])

# ORIENTATION CORRECTION/ADJUSTMENT
def rotate_image(image, angle):
    """Rotates the image in given angle

    Args:
        image: The image to be rotated
        angle: angle to be rotated clock-wise

    Returns:
        the rotated image
    """
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

# Function to calculate the angle of rotation
def calculate_rotation_angle(bounding_boxes):
    """Calculate angle of inclination 

    Args:
        bounding_boxes: coordinates of the identified rectangular region of text
        
    Returns:
        the average angle of inclination
    """
    angles = []
    for box in bounding_boxes:
        angle = math.atan2(box[2][1] - box[0][1], box[2][0] - box[0][0])
        angles.append(angle)
    average_angle = np.mean(angles)
    return math.degrees(average_angle)

@st.cache_data 
def load_model(): 
    reader = ocr.Reader(['en'], gpu=False, model_storage_directory='.')
    return reader 

# Define the function to detect text in an image
def detect_text(image, thre = 0):
    """Detects text in an image using EasyOCR.

    Args:
        image: The image to detect text in.
        thre: Threshold score to extract text

    Returns:
        A list of text boxes, each containing the text and its coordinates.
    """
    # instance text detector
    result_ = reader.readtext(image)
    max_box = []
    result = []
    # Get Coordinates of identified text regions with score > threshold
    for i, t in enumerate(result_):
        bbox, text, score = t
        if score > thre:
            max_box.append(bbox)
            result.append(text)   
            #print(text, bbox)     
    return result, max_box
    
reader = load_model() #load model

if image is not None:

    input_image = Image.open(image) #read image
    st.write("Original Image")
    st.image(input_image) #display image
    
    ## Image Pre-processing
    # normalization
    input_image = np.array(input_image)
    norm_img = np.zeros((input_image.shape[0], input_image.shape[1]))
    norm_img = cv2.normalize(input_image, norm_img, 0, 255, cv2.NORM_MINMAX)
    # convert to grayscale
    gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY) 

    with st.spinner("Text Extraction in progress!!!!...	:running: 	:running: "):

        # detect text on original image
        result, bbox = detect_text(gray, thre = 0)
        # Determine angle for orientation correction 
        max_angle = calculate_rotation_angle(bbox)
        print("Rotation angle: ", max_angle)
        # Rotate gray image for text extraction
        rotated_image = rotate_image(gray, -max_angle)
        # Rotate original image for text extraction
        rotated_ori = rotate_image(norm_img, -max_angle)
        

        # detect text on rotated & corrected image
        result2, bbox2 = detect_text(rotated_image, thre = 0.00001)
        
        st.write("The extracted texts: ")
        st.write(result) # display text
        
        # Write extracted text with bounding boxes in image
        for i in range(len(result2)):
            cv2.rectangle(rotated_ori, (int(np.array(bbox2[i][0])[0]), int(np.array(bbox2[i][0])[1])), (int(np.array(bbox2[i][2])[0]), int(np.array(bbox2[i][2])[1])), (239, 26, 255), 6)
            cv2.putText(rotated_ori, result2[i], (int(np.array(bbox2[i][0])[0]), int(np.array(bbox2[i][0])[1])-25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (184, 26, 255), 3)
        
        #display rotated image with identified text and bounding boxes
        st.write("Corrected Image with extracted text and bounding boxes")
        st.image(rotated_ori) 
       
    st.success("**Done!**  :100:")
    st.balloons()
else:
    st.write("Upload an Image")

st.caption("**GitHub:** [@anjusm](https://github.com/anjusm/Text-Snap-Extractor/tree/main) ❤️ ")

