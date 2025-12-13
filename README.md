🧠 Brain Tumor Classification from MRI

A Deep Learning–based Convolutional Neural Network (CNN) Project

This repository contains the implementation of a brain tumor classification system using MRI scan images. The project applies deep neural networks and advanced preprocessing techniques to classify MRI scans into:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

This work is based on 

📌 Project Highlights

CNN-based multi-class classification

Automated feature extraction from MRI images

Extensive preprocessing

Training and evaluation on benchmark MRI datasets

Achieved high model accuracy (85%–92%)

Includes analysis, challenges, and future enhancements

📁 Repository Structure
📦 Brain-Tumor-Classification
 ┣ 📜 brainTumorusemri.ipynb
 ┣ 📂 dataset/                      # (optional — if uploaded)
 ┣ 📜 README.md
 ┗ 📜 requirements.txt              # optional

🚀 Features
✔ Deep Learning Techniques Used

Convolutional Neural Networks (CNN)

MaxPooling layers

ReLU activation

Dropout for regularization

Adam optimizer

Softmax classifier

Data Augmentation (rotation, zoom, shifting, flips)

Normalization & noise reduction

✔ Preprocessing Steps

Skull stripping

Image resizing (224×224)

Normalization

Histogram equalization

Augmented training data

📊 Model Training & Evaluation
Epoch Range	Observation	Accuracy
1–10	Learns basic MRI patterns	50–60%
10–20	Extracting deeper features	70–80%
20–30	Model stabilizes	80–85%
30+	Strong classification	85–92%
Performance Metrics

Training Accuracy: ~90%

Validation Accuracy: ~84–87%

Loss curves converge smoothly

Confusion matrix shows strong separation between classes

Inference time: <0.1 sec/image

🧪 Datasets Used

Kaggle Brain Tumor MRI Dataset

Figshare MRI Dataset

Standard tumor datasets (Glioma, Meningioma, Pituitary, No Tumor)

Preprocessing Pipeline:

Resize → Normalize → Denoise → Augment

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/Mohanariprasath/Brain-tumor-classification-using-mri.git
cd Brain-tumor-classification-using-mri

2. Install dependencies
pip install -r requirements.txt

3. Open the Jupyter notebook
jupyter notebook brainTumorusemri.ipynb

🔍 Observations & Key Insights

CNN detects tumor boundaries with high clarity

Preprocessing significantly improves model accuracy

Data augmentation helps prevent overfitting

Higher resolution inputs → better performance

🛠 Challenges Faced
Challenge	Solution
Limited dataset	Data augmentation
Overfitting	Dropout + early stopping
Noisy MRI scans	Smoothing + normalization
Class imbalance	Oversampling techniques
Similar tumor shapes	Deeper CNN layers
🏥 Real-World Applications

Computer-aided Diagnosis (CAD) tools

Clinical MRI interpretation support

Early tumor screening

Telemedicine & remote diagnostics

AI-assisted radiology workflows

🔮 Future Enhancements

Transfer Learning (VGG16, ResNet50, EfficientNet)

3D MRI volumetric classification

Explainable AI (Grad-CAM)

Real-time API deployment (Flask / FastAPI)

Multi-GPU training

Multi-modal MRI + CT fusion
