# Vehicle Image Classification with PyTorch and Transfer Learning 🚗

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
