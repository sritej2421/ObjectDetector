# Vehicle Image Classification with PyTorch and Transfer Learning 

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A deep learning project to classify vehicle images into 12 categories, with a focus on cleaning a real-world noisy dataset and deploying the model using the ONNX format.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

This project tackles the task of image classification for a dataset of 12 different vehicle types. The primary challenge was the quality of the dataset, which contained significant class imbalance, blurry images, and erroneous non-vehicle images (e.g., timestamp watermarks).

The solution employs a systematic pipeline:
1.  **Data Analysis & Cleaning:** Using `cleanvision` to identify and remove problematic images.
2.  **Modeling:** Implementing a transfer learning approach with a pre-trained **ResNet50** model.
3.  **Evaluation:** Assessing the model's performance using detailed metrics like Precision, Recall, F1-Score, and a confusion matrix.
4.  **Export:** Converting the final model to the **ONNX** format for interoperability and deployment.

---

## Key Features

- **Robust Data Cleaning:** Automated detection and removal of duplicates, low-quality images, and outliers.
- **Transfer Learning:** Leveraging a pre-trained ResNet50 on ImageNet to achieve strong performance with a limited dataset.
- **Detailed Evaluation:** In-depth performance analysis beyond simple accuracy.
- **Deployment-Ready:** The final model is exported to the standard ONNX format.

---

## Methodology

The project followed a structured workflow:

1.  **Initial Data Exploration:** Analyzed the class distribution to confirm a severe imbalance.
2.  **Automated Cleaning:** Used `cleanvision` to programmatically find issues. Identified and removed 107 useless or low-quality images.
3.  **Data Preprocessing & Augmentation:** Resized images to 224x224 and applied augmentations (random flips, crops) to the training set to improve model generalization.
4.  **Model Training:** Fine-tuned the final layer of a ResNet50 model for 10 epochs.
5.  **Evaluation:** Generated a classification report and confusion matrix to understand class-wise performance.
6.  **Model Export:** Converted the best-performing PyTorch model (`.pth`) to `vehicle_classifier.onnx`.

---

## Project Structure
.
├── final_submission/
│   ├── Image_Classification_Notebook.ipynb  # Main notebook with all code
│   ├── vehicle_classifier.onnx            # Final exported ONNX model
│   ├── classes.txt                        # List of vehicle classes
│   ├── Report.pdf                         # Detailed project report
│   └── README.md                          # This README file
└── dataset.zip                            # The original zipped dataset

---

## Setup and Installation

To set up the environment and run this project, follow these steps.

1.  **Clone the repository (optional):**
    ```bash
    git clone [your-repo-link]
    cd [your-repo-name]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file can be generated using `pip freeze > requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Key libraries include: `torch`, `torchvision`, `cleanvision[all]`, `scikit-learn`, `onnx`, `onnxruntime`, `seaborn`.

---

## How to Run

1.  **Place the Dataset:** Make sure the `dataset.zip` file is in the root directory.
2.  **Launch the Notebook:** Open and run the `Image_Classification_Notebook.ipynb` in a Jupyter environment (like Jupyter Lab or Google Colab).
3.  **Execute Cells:** Run the cells in sequential order. The notebook is structured to handle all steps from data unzipping to model export.
4.  **Check Outputs:** The trained model (`best_model.pth`), the ONNX model (`vehicle_classifier.onnx`), and the `classes.txt` file will be generated in the root directory.

---

## Results

The model achieved a **peak validation accuracy of 72.4%** after 10 epochs of training.

- **Classification Report:** The model showed strong performance on majority classes like trucks and buses.
- **Confusion Matrix:** Some confusion was noted between visually similar classes like "auto-rickshaw" and "e-rickshaw".

For a detailed breakdown of performance metrics and the confusion matrix plot, please refer to the `Report.pdf` or the evaluation section in the notebook.

---

## Future Work

- **Advanced Fine-Tuning:** Unfreeze more layers of the ResNet model and train with a lower learning rate to better adapt to the dataset.
- **Handle Class Imbalance:** Implement techniques like a weighted loss function or SMOTE (Synthetic Minority Over-sampling Technique).
- **Experiment with Architectures:** Test other models like EfficientNet or Vision Transformers (ViT) to potentially improve accuracy.
