import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained emotion detection model
emotion_model = load_model('path/to/emotion_detection_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read the image
img = cv2.imread('./Images/neutral.jpg')
gray_scale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_classifier.detectMultiScale(gray_scale_image, 1.3, 2)

for (x, y, w, h) in faces:
    # Extract the face region
    face = gray_scale_image[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype('float') / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    
    # Predict the emotion
    preds = emotion_model.predict(face)[0]
    emotion_probability = np.max(preds)
    label = emotion_labels[preds.argmax()]
    
    # Draw rectangle over the face
    cv2.rectangle(img, (x, y), (x+w, y+h), (100, 25, 70), 5, cv2.LINE_AA)
    
    # Display the label at the top of the bounding box
    label_position = (x, y-10)
    cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 25, 70), 2, cv2.LINE_AA)

# Display the output
cv2.imshow("Emotion Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
