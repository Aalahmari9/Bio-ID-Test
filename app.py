import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# Define the LeNet-5 Model
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        
        # Layer 1: Convolutional Layer (input channels: 3, output channels: 6, kernel size: 5x5)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        # Layer 2: Max Pooling Layer (kernel size: 2x2, stride: 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Convolutional Layer (input channels: 6, output channels: 16, kernel size: 5x5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5)
        # Layer 4: Max Pooling Layer (kernel size: 2x2, stride: 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # Layer 5: Fully Connected Layer (input: 16*5*5, output: 120)
        self.fc1 = nn.Linear(46656, 264)
        # Layer 6: Fully Connected Layer (input: 120, output: 84)
        # self.fc2 = nn.Linear(264, 64)
        # Layer 7: Fully Connected Layer (input: 84, output: num_classes)
        self.fc3 = nn.Linear(264, num_classes)

    def forward(self, x):
        # Pass through first conv layer followed by max pooling
        x = self.pool1(torch.relu(self.conv1(x)))
        # Pass through second conv layer followed by max pooling
        x = self.pool2(torch.relu(self.conv2(x)))
        # Flatten the feature maps to pass through fully connected layers
        x = x.view(x.size(0), -1)  # Reshape to batch_size x (16*5*5)
        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation for the final layer since it's for classification)
        return x

# Load the model
def load_model(model_path, num_classes):
    model = LeNet5(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the image and crop the face
def preprocess_image(image, face_cascade_path='haarcascade_frontalface_default.xml'):
    # Convert the image from BGR (OpenCV) to RGB (PIL expects RGB)
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

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

# Streamlit app
def main():
    st.title("Real-Time Person Recognition")

    # Sidebar: Load model and dataset information
    model_path = "person_classifier_model.pth"  # Path to your trained model
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