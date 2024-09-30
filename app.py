import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import sqlite3
from datetime import datetime
import numpy as np
import os

# Load the pre-trained ResNet18 model and modify the last layer
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)  # Load ResNet18 without pre-training
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final fully connected layer
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the image and crop the face
def preprocess_image(image, face_cascade_path='haarcascade_frontalface_default.xml'):
    # Check if image is already a PIL image
    if isinstance(image, Image.Image):
        img_rgb = np.array(image)
    else:
        # Convert the image from BGR (OpenCV) to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Crop the first detected face
    x, y, w, h = faces[0]
    face = img_rgb[y:y+h, x:x+w]

    # Convert the cropped face to a PIL image
    face_pil = Image.fromarray(face)

    # Define the image preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations to the face image
    face_tensor = transform(face_pil)
    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)

    return face_tensor

# Perform classification
def classify_image(model, image_tensor, class_names):
    # Move the tensor to the same device as the model (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    model = model.to(device)

    # Make predictions
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class

# SQLite3 database initialization
def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT,
            datetime TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Store detection result in the SQLite3 database
def log_detection(person_name):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO detections (person_name, datetime) VALUES (?, ?)', (person_name, timestamp))
    conn.commit()
    conn.close()

# Streamlit app
def main():
    st.title("Real-Time Person Recognition with ResNet18")

    # Initialize the database
    init_db()

    # Sidebar: Load model and dataset information
    model_path = "person_classifier_resnet18.pth"  # Path to your trained ResNet18 model
    class_names = os.listdir("Data/train")  # Load class names from dataset
    num_classes = len(class_names)

    # Load the model
    model = load_model(model_path, num_classes)

    # Select a mode
    mode = st.sidebar.selectbox("Select Mode", ["Camera Feed", "Upload Image"])

    # Create two columns: left for the image, right for the prediction
    col1, col2 = st.columns([2, 1])

    # Initialize the variable for the current prediction
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = "No prediction yet"

    if mode == "Camera Feed":
        with col1:
            # Start the camera feed
            st.write("Using camera for real-time person recognition")
            run = st.checkbox('Run')
            FRAME_WINDOW = st.image([])

            cap = cv2.VideoCapture(0)

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.write("Unable to read camera feed.")
                    break

                # Try to classify the person
                try:
                    face_tensor = preprocess_image(frame)  # Preprocess the captured frame
                    predicted_class = classify_image(model, face_tensor, class_names)
                    # Log the detection into the database
                    log_detection(predicted_class)
                    st.session_state.current_prediction = predicted_class
                except ValueError as e:
                    st.session_state.current_prediction = str(e)  # Handle no face detected

                # Overlay the current prediction on the frame
                cv2.putText(frame, f"Prediction: {st.session_state.current_prediction}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Convert BGR to RGB and display the frame with the prediction overlay
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

            cap.release()

    elif mode == "Upload Image":
        with col1:
            # Upload an image for classification
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)

                # Display the uploaded image
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Perform classification on the uploaded image
                try:
                    face_tensor = preprocess_image(image)  # Preprocess the uploaded image
                    predicted_class = classify_image(model, face_tensor, class_names)
                    st.session_state.current_prediction = predicted_class
                except ValueError as e:
                    st.session_state.current_prediction = str(e)  # Handle no face detected

                # Display the current prediction in column 2
                with col2:
                    st.markdown("### Prediction:")
                    st.write(st.session_state.current_prediction)

if __name__ == "__main__":
    main()
