**AI-for-Wheat-Disease**

During my time at the University, I built a Deep Learning model that can predict whether many wheat leaves, which is diseased and 
which is not with acceptable scores and assist the user, majorly the farmers to distinguish between the healthy and diseased wheat plants.


## 🚀 Introduction
This repository contains an implementation of a Deep Convolutional Neural Network (CNN) designed to detect wheat leaf diseases. By providing farmers with accurate and timely information about plant health, this tool aims to enhance agricultural productivity and sustainability.

---

## ✨ Features
- **Deep CNN Model:** Effectively classifies various wheat leaf diseases.
- **User-Friendly Interface:** Easy to train and test the model with simple commands.
- **Visualization Tools:** Graphical representations of training metrics and model performance.
- **Extensive Documentation:** Well-commented code and comprehensive user guides.

---

## 📦 Getting Started

### Prerequisites
To get started, ensure you have the following software installed:
- **Python 3.6 or higher**
- **TensorFlow 2.x** (or Keras)
- **OpenCV**
- **NumPy**
- **Matplotlib**

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/deep-cnn-wheat-disease-detection.git
   cd deep-cnn-wheat-disease-detection
Install the required packages:
bash
Copy code
pip install -r requirements.txt

**->🛠️ Usage**
Training the Model
Prepare your dataset and run the following command:
python train.py --data_dir path_to_your_dataset --epochs 50

**Making Predictions**

To classify an image of a wheat leaf:
python predict.py --image path_to_image

**📈 Training Process**
Data Augmentation: Enhances model generalization.
Epochs: Trained over a specified number of epochs.
Model Saving: Best-performing model weights are saved.
🎉 Results
The model's performance is evaluated through accuracy and loss metrics, which are logged during training. Sample results and visualizations are available in the results directory.

**->📊 Dataset**
In total, there are 4800 images in the dataset where

-> 1279 Healthy Wheat
-> 939 Wheat Loose Smut
-> 1622 Leaf Rust
-> 860 (after refining) Crown and Root Rot

**-> Model**
A new kind of technique has been used (by trying to browse through the many sources) of dividing every type of plant image into a binary array – four arrays have been used – to get a preferable and smooth way of classifying the images along with the help of Neural Networks.Even in a small cycle of 30 epochs only (as a check of the identifying model’s result), the accuracy score was a favourable 87.31%. Looking at the results, possible accuracy of 87.31% has been attained.

**->🏗️ Model Architecture**
The architecture consists of:

Convolutional Layers: For feature extraction.
Max Pooling Layers: To reduce dimensionality.
Dropout Layers: To prevent overfitting.
Fully Connected Layers: For final classification.
For a detailed model structure, refer to the model.py file.


# 🌾 Deep CNN for Wheat Leaf Disease Detection for Precise Agriculture 🌾

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**🤝 Contributing
Contributions are welcome! Please read CONTRIBUTING.md for guidelines on how to contribute to this project.**

**📜 License
This project is licensed under the MIT License. See the LICENSE file for details.**

**📫 Contact
For inquiries or feedback, please reach out:
-> ommsubrato.dey@gmail.com**

**Name: Om Subrato Dey
Email: your.email@example.com
GitHub: XOSDX
Thank you for your interest in this project! Happy coding! 🚀**
