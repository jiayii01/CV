# importing libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cvlib as cv

@st.cache_resource()
def load_models():
    model = tf.keras.models.load_model("emotion_recognition.model")
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    return model, genderNet

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

st.title("Computer Vision: Emotion and Gender Recognition")

model, genderNet = load_models()

img_file = st.file_uploader(
    "Upload image of a person", type=['jpeg', 'jpg', 'png', 'webp'])

if img_file:
    slot = st.empty()
    with st.spinner("Processing Image..."):
        image = Image.open(img_file)

        image = np.array(image.convert('RGB'))

        classes = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']

        #run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])

        # apply face detection
        face, confidence = cv.detect_face(image)

        # loop through detected faces
        for idx, f in enumerate(face):

            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

            # crop the detected face region
            face_crop = np.copy(image[startY:endY,startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (48,48))
            face_crop2 = cv2.resize(face_crop, (227,227))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = model.predict(face_crop)[0]

            # get label with max probability
            idx = np.argmax(conf)
            label = classes[idx]

            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10
            Y2 = startY - 25 if startY - 15 > 10 else startY + 25

            # get gender
            blob=cv2.dnn.blobFromImage(face_crop2, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            # get gender with max probability
            gender = genderList[genderPreds[0].argmax()]

            # write label and confidence above face rectangle
            cv2.putText(image, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.putText(image, f'{gender}', (startX, Y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
            FRAME_WINDOW.image(image)