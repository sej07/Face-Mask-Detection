## Face Mask Detection using CNN

_This project aims to develop a Convolutional Neural Network (CNN) capable of detecting whether a person is wearing a face mask or not from an image. The model processes images, learns spatial patterns, and classifies them into two categories: Mask and No Mask._

#### Dataset Details
- Dataset: Face Mask
- Source: Kaggle (https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- Number of with mask images: 3725
- Number of without mask images: 3828
- Image Shape: Resized to 128 × 128 pixels  
- Labels: Mask or No Mask

#### ML Workflow: 
1. Importing Libraries
    1. `TensorFlow`, `Keras` for model building and training
    2. `NumPy`, `Pandas` for data handling
    3. `Matplotlib` for visualization
2. Data Exploration
    1. Verified class distribution to ensure no major imbalance between “Mask” and “No Mask” images.
    2. Sample images were plotted to visually confirm quality and diversity.
3. Image Preprocessing
    1. Resizing all images to 128×128×3.
    2. Normalizing pixel values to the range [0, 1].
4. Train-Test Split
    1. Dataset split into training and testing sets using an 80-20 ratio.
5. Model Architecture
    1. Conv2D Layer with 32 filters (3×3) and ReLU activation for feature extraction
    2. MaxPooling2D with pool size (2×2) to reduce spatial dimensions
    3. Conv2D Layer with 64 filters (3×3) and ReLU activation for deeper feature extraction
    4. MaxPooling2D with pool size (2×2) to further downsample feature maps
    5. Flatten to convert 2D feature maps into a 1D vector
    6. Dense Layer with 128 units and ReLU activation for high-level feature learning
    7. Dropout (0.5) to prevent overfitting
    8. Dense Layer with 64 units and ReLU activation for refined feature learning
    9. Dropout (0.5) to improve generalization
    10. Dense Output Layer with 2 units and Sigmoid activation for binary classification (mask vs. no mask) 
    > Input: 128×128×3 RGB image   
    > Conv2D: 32 filters, (3×3) kernel, ReLU   
    >MaxPooling2D: pool size (2×2)  
    >Conv2D: 64 filters, (3×3) kernel, ReLU activation   
    >MaxPooling2D: pool size (2×2)   
    >Flatten layer   
    >Dense: 128 units, ReLU activation   
    >Dropout: 0.5    
    >Dense: 64 units, ReLU activation  
    >Dropout: 0.5    
    >Dense: 2 units, Sigmoid activation 
    11. Loss Function: Binary cross-entropy
    12. Optimizer: Adam
    13. Metrics: Accuracy, Loss
6.  Model Training
    1. Compiled and trained the model for 10 epochs. 
7. Model Evaluation
    1. Metrics used: `Accuracy`, `Loss`

#### Results
Accuracy Score: 0.92

Loss: 0.19

Val Accuracy: 0.90

Val Loss: 0.28

Test Accuracy: 0.84


#### Improvements
1. Implement a deeper CNN or transfer learning (e.g., MobileNetV2) for better accuracy
2. Test with real-time webcam feed for live detection.
3. Add face detection preprocessing to handle images with multiple faces.

#### Visualizations
Model Summary
<img width="763" height="592" alt="Screenshot 2025-08-13 203238" src="https://github.com/user-attachments/assets/2a23315e-0b4b-4c58-95fe-b5e8fff2c143" />

Accuracy graph before Optimization 
<img width="763" height="592" alt="Screenshot 2025-08-13 203247" src="https://github.com/user-attachments/assets/71252459-6786-4757-aa3b-19c8a0073957" />

Accuracy graph after Optimization
<img width="763" height="592" alt="Screenshot 2025-08-13 203254" src="https://github.com/user-attachments/assets/cc9f6997-0d7b-4894-be6e-950c376939ab" />

Loss graph before Optimization
<img width="763" height="592" alt="Screenshot 2025-08-13 203301" src="https://github.com/user-attachments/assets/fccce209-21f5-42db-a217-de5d2376ffd0" />

Loss graph after Optimization
<img width="763" height="592" alt="Screenshot 2025-08-13 203308" src="https://github.com/user-attachments/assets/92268237-7088-4df7-bb8d-808f91f8668a" />

#### Assumptions
1. All images are correctly labeled and belong to the right class.
2. Image resizing to 128×128 does not remove critical facial details.  

#### Key Observation
Images in the “No Mask” category often have higher variation in facial expressions, while “Mask” images tend to have more uniform facial coverage and reduced visible features.

