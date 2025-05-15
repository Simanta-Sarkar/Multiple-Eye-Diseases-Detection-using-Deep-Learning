# Multiple-Eye-Diseases-Detection-using-Deep-Learning
This project aims to assist in the early and efficient detection of various eye diseases using deep learning techniques, particularly Convolutional Neural Networks (CNNs). The model is designed to work with retinal fundus images and can classify conditions such as Cataract, Glaucoma, Diabetic Retinopathy (DR), and Age-Related Macular Degeneration (AMD). This approach is intended to make eye disease screening more accessible, especially in resource-constrained environments.

1. Key Features
Multi-Disease Classification: Detects multiple eye diseases from fundus images.

Lightweight Architecture: Utilizes TinyVGG, a simplified CNN ideal for low-resource setups.

High Accuracy: Achieved up to 98% accuracy in multi-disease classification on the tested dataset.

Modular Workflow: Includes data preprocessing, training, validation, and testing stages.

2. Deep Learning Model
The model is based on a TinyVGG architecture:
2 Convolutional + MaxPooling blocks
1 Fully Connected Dense layer
Output layer with Softmax activation

This structure balances performance and computational efficiency, making it suitable for real-world applications.

3.Results
Model	Accuracy
Multi-Disease Model	98%
Diabetic Retinopathy	80%
Glaucoma	90% (Estimated from 18/20 correct)
