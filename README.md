# AI-Powered Visual Defect Detection System

🔗 Live Demo: [https://ai-powered-visual-defect-detection-pryaxjddqh8uskqqgwzhlm.streamlit.app/](https://ai-powered-visual-defect-detection-pryaxjddqh8uskqqgwzhlm.streamlit.app/)

This project uses Deep Learning and Transfer Learning (ResNet18) to automatically detect and classify steel surface defects.

## Overview

This project focuses on the automated detection and classification of surface defects in steel sheets using deep learning techniques. The system is trained on the NEU Surface Defect Database and is capable of identifying six different categories of manufacturing defects.

The project was developed to demonstrate the application of computer vision and transfer learning in industrial quality inspection. A pre-trained ResNet18 model was fine-tuned on the dataset to achieve high classification accuracy. The final model was deployed through a Streamlit web application that allows users to upload an image and receive a defect prediction along with a confidence score.

---

## Dataset

The project uses the **NEU Surface Defect Database**, a publicly available dataset commonly used for steel surface defect classification tasks.

### Defect Classes

* Crazing
* Inclusion
* Patches
* Pitted Surface
* Rolled-in Scale
* Scratches

### Dataset Distribution

| Dataset Split  | Number of Images |
| -------------- | ---------------- |
| Training Set   | 1440             |
| Validation Set | 360              |
| Total Images   | 1800             |

Each image was resized and processed before being used for model training.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* PIL (Pillow)
* Scikit-learn
* Streamlit
* Jupyter Notebook

---

## Project Workflow

### 1. Data Exploration

The dataset was analyzed to understand:

* Available defect categories
* Number of images in each class
* Image dimensions
* Pixel value distribution
* Sample defect images

---

### 2. Data Preprocessing

The following preprocessing steps were performed:

* Image resizing to 224 × 224 pixels
* Conversion of images to tensors
* Creation of training and validation datasets
* Batch generation using DataLoader

---

### 3. Baseline CNN Model

A custom Convolutional Neural Network was initially developed to establish baseline performance.

Architecture used:

* Convolution Layer + ReLU + MaxPooling
* Convolution Layer + ReLU + MaxPooling
* Convolution Layer + ReLU + MaxPooling
* Fully Connected Layer
* Output Layer (6 Classes)

**Baseline Validation Accuracy:** 74.44%

---

### 4. Transfer Learning using ResNet18

To improve performance, transfer learning was implemented using a pre-trained ResNet18 model.

Steps performed:

* Loaded pre-trained ResNet18 weights
* Replaced the final classification layer
* Modified output layer to classify 6 defect categories
* Trained only the final layer initially
* Fine-tuned the model on the defect dataset

**Training Accuracy:** 98.96%

**Validation Accuracy:** 99.17%

---

### 5. Model Evaluation

The trained model was evaluated using:

* Accuracy Score
* Classification Report
* Precision
* Recall
* F1-Score

### Classification Results

The model achieved strong performance across all defect categories with an overall validation accuracy of **99.17%**.

---

### 6. Deployment

The trained model was deployed using **Streamlit**.

Features of the application:

* Upload steel surface images
* Automatic defect classification
* Confidence score display
* User-friendly interface
* Real-time prediction

---

## Project Structure

```text
AI_POWERED_VISUAL_DEFECT_DETECTION/

│
├── App/
│   └── app.py
│
├── Dataset/
│   ├── train/
│   └── validation/
│
├── models/
│   ├── cnn_baseline.pth
│   └── resnet18_final.pth
│
├── Notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_cnn_model.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_deployment.ipynb
│
├── Reports/
├── venv
├── images
├── README.md
└── requirements.txt
```

## Results

| Model                      | Validation Accuracy |
| -------------------------- | ------------------- |
| Custom CNN                 | 74.44%              |
| ResNet18 Transfer Learning | 99.17%              |

The transfer learning approach significantly improved performance and provided highly accurate defect classification results.

---

## How to Run the Project

### Clone the Repository

```bash
git clone <(https://github.com/palpriyadarshini-pixel/AI-Powered-Visual-Defect-Detection.git)>
cd AI_POWERED_VISUAL_DEFECT_DETECTION
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Streamlit Application

```bash
cd App
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## Future Improvements

* Support additional defect categories
* Deploy the application to the cloud
* Integrate real-time camera-based inspection
* Add defect localization using object detection models
* Improve model explainability using Grad-CAM visualizations

---

## Conclusion

This project demonstrates the use of deep learning and transfer learning for industrial defect classification. A pre-trained ResNet18 model was successfully adapted for steel surface defect detection and achieved a validation accuracy of 99.17%. The final Streamlit application provides an easy-to-use interface for testing and demonstrating the model in a practical setting.

---

## Author

**Priyadarshini Pal**

