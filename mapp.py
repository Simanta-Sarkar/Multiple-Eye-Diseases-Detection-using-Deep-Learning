import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # nn.Dropout(0.2)
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
            # nn.Dropout(0.2)
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

# Function to preprocess the uploaded image
def preprocess_image(image):
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

# Function to make predictions
def predict(image):
    # Instantiate the model and load the state dictionary
    model = ImprovedTinyVGGModel(3, 48, len(class_mapping))  # Ensure this matches your actual model class
    model.load_state_dict(torch.load('./MultipleEyeDiseaseDetectModel.pth'))
    model.eval()  # Set the model to evaluation mode
    
    # Preprocess the image
    image_tensor = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
    prediction = F.softmax(output, dim=1).numpy()
    return prediction

# Streamlit UI
def main():
    st.title("Eye Disease Detection")

    # File uploader
    uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded fundus image', use_column_width=True)

        # Predictions
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                # Make prediction
                prediction = predict(image)
                # Get predicted class and confidence
                predicted_class_index = np.argmax(prediction)
                predicted_class = class_mapping[predicted_class_index]
                confidence = prediction[0][predicted_class_index]
                # Display prediction result
                st.success(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

if __name__ == '__main__':
    main()
