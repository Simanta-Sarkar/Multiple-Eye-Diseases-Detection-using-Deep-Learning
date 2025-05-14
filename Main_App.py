import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F

# Function to preprocess an image for Keras models
def preprocess_image(image_path, target_size=None):
    img = Image.open(image_path)
    if target_size:
        img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to preprocess the uploaded image for PyTorch models
def preprocess_image_torch(image):
    # Resize image to match model input size
    image = image.resize((224, 224))  # Update to match the input size expected by your model
    # Convert image to numpy array
    image_array = np.array(image)
    # Normalize pixel values
    image_array = image_array / 255.0
    # Transpose the image to match PyTorch's expected input format (C, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))
    # Convert to torch tensor and add batch dimension
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
    return image_tensor

# Function to display project information
def about_project():
    st.title("About Our Project")
    st.write("A software for detecting different eye diseases from fundus images utilizing CNN Architecture")

# Define the ImprovedTinyVGGModel (replace this with your actual model class definition)
class ImprovedTinyVGGModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super(ImprovedTinyVGGModel, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(hidden_units),
            nn.Dropout(0.2)  # Dropout for regularization
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, 4 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * hidden_units, 4 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(4 * hidden_units),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(4 * hidden_units, 4 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * hidden_units, 2 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(2 * hidden_units),
            nn.Dropout(0.2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(2 * hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, output_shape, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(output_shape),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1176, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.classifier(x)
        return x

# Define the mapping of prediction indices to eye disease categories
class_mapping = {
    0: 'AMD',
    1: 'Cataract',
    2: 'Glaucoma',
    3: 'Myopia',
    4: 'Noneye',
    5: 'Normal'
}

# Function to make predictions
def predict(image):
    # Instantiate the model and load the state dictionary
    model = ImprovedTinyVGGModel(3, 48, len(class_mapping))  # Ensure this matches your actual model class
    model.load_state_dict(torch.load('./MultipleEyeDiseaseDetectModel.pth'))
    model.eval()  # Set the model to evaluation mode
    
    # Preprocess the image
    image_tensor = preprocess_image_torch(image)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
    prediction = F.softmax(output, dim=1).numpy()
    return prediction

# Function for Glaucoma detection
def glaucoma_detection(uploaded_image):
    model = load_model('my_model2.h5')
    image = Image.open(uploaded_image)
    image = ImageOps.fit(image, (100, 100), Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction[0][0]

# Function for Diabetic Retinopathy detection
def diabetic_retinopathy(uploaded_image):
    # Load the trained model
    model = load_model("retina_weights.hdf5")

    # Define labels mapping
    labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3: 'Proliferate_DR', 4: 'Severe'}

    # Preprocess and predict
    img_array = preprocess_image(uploaded_image, target_size=(256, 256))
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    return predicted_class

# Streamlit UI
def main():
    st.sidebar.title("Navigation")
    navigation = st.sidebar.radio("", ["About our project", "Multiple Eye Disease Detection", "Glaucoma detection using CNN", "Diabetic Retinopathy using Deep Learning"])

    if navigation == "About our project":
        about_project()
    elif navigation == "Multiple Eye Disease Detection":
        st.title("Multiple Eye Disease Detection")
        st.subheader("By:\n1.Jishantu Kripal Bordoloi\n\n2.Simanta Sarkar\n\n3.Amlan Jyoti Dutta\n\n4.Hriseekesh Kalita")
        st.subheader("Under the guidance of: Dr. Sanyukta Chetia\nAssistant Professor & HOD\n\nDepartment of Electronics and Telecommunication Engineering")

        st.write("\n")

        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

            # Make predictions and display result
            if st.button("Predict"):
                prediction = predict(Image.open(uploaded_image))
                predicted_class_index = np.argmax(prediction)
                predicted_class = class_mapping[predicted_class_index]
                st.success(f"Predicted Class: {predicted_class}")
    elif navigation == "Glaucoma detection using CNN":
        st.title("Glaucoma Detection")
        st.subheader("By:\n1.Jishantu Kripal Bordoloi\n\n2.Simanta Sarkar\n\n3.Amlan Jyoti Dutta\n\n4.Hriseekesh Kalita")
        st.subheader("Under the guidance of: Dr. Sanyukta Chetia\nAssistant Professor & HOD\n\nDepartment of Electronics and Telecommunication Engineering")

        st.write("\n")

        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

            # Make predictions and display result
            if st.button("Predict"):
                prediction = glaucoma_detection(uploaded_image)
                if prediction > 0.5:
                    st.success("Prediction: Your eye is Healthy. Great!!")
                    st.balloons()
                else:
                    st.error("Prediction: You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.")
    elif navigation == "Diabetic Retinopathy using Deep Learning":
        st.title("Diabetic Retinopathy Detection")
        st.subheader("By:\n1.Jishantu Kripal Bordoloi\n\n2.Simanta Sarkar\n\n3.Amlan Jyoti Dutta\n\n4.Hriseekesh Kalita")
        st.subheader("Under the guidance of: Dr. Sanyukta Chetia\nAssistant Professor & HOD\n\nDepartment of Electronics and Telecommunication Engineering")

        st.write("\n")

        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

            # Make predictions and display result
            if st.button("Predict"):
                prediction = diabetic_retinopathy(uploaded_image)
                st.success(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    main()
