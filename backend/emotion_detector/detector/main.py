from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def emotion_detector(base64Image):
    face_classifier = cv2.CascadeClassifier(r'C:/Users/default.LAPTOP-6IMDNNN0/Documents/CodePlayground/emotion_detection/Emotion-Detection/backend/emotion_detector/detector/haarcascade_frontalface_default.xml')
    classifier = load_model(
        r'C:/Users/default.LAPTOP-6IMDNNN0/Documents/CodePlayground/emotion_detection/Emotion-Detection/backend/emotion_detector/detector/final_model.h5')

    emotion_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    frame = parse_base64_image(base64Image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,  scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        # print("ROI GRAY: ")
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_NEAREST)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            return label
        else:
            return 'No faces'


import base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO


def parse_base64_image(base64_image):
    # Remove the "data:image/jpeg;base64," prefix from the base64 image string
    encoded_data = base64_image.split(',')[1]
    # Decode the base64 image string to bytes
    decoded_data = base64.b64decode(encoded_data)
    # Convert the bytes to a PIL Image object
    image = Image.open(BytesIO(decoded_data))
    # Convert the PIL Image to an OpenCV image (numpy array)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return frame
