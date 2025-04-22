# 🚀 AI-Based Early Disease Detection with Visual Q&A for Healthcare Diagnostics

## 📌 Overview
This project develops an AI system to analyze medical images (X-rays, MRIs, dermatoscopic images) for early disease detection. The system incorporates image classification, segmentation, and an interactive Visual Question Answering (VQA) module to enable clinicians to ask targeted questions about the images and receive interpretable responses.

## 📌 Objectives
- **Classification:** Detect abnormalities using a lightweight CNN (e.g., MobileNetV2, DenseNet-121) with transfer learning.
- **Segmentation:** highligh abnormal regions using U-Net/UNet++ with modifications for CPU-efficient deployment.
- **Visual Q&A:** Enable interactive queries on the processed images using a transformer-based VQA module optimized for resource-limited environments.

## 📂 Dataset
- **Chest X-ray Dataset:** https://drive.google.com/file/d/1-SWJ_nIgotQ11ZHapb-uqWndvzeRs80d/view?usp=sharing
- **NIH Chest X-ray Dataset:** [Link to dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) 
- **VQA Datasets:** VQA-RAD and PMC-VQA (details provided in the documentation)
  
Download the required datasets and place them in the `datasets/` directory following the structure provided above. 
The dataset from the drive is preprocessed and augmented to improve model generalization.

## 🛠️ Technologies Used
- **Python** 🐍
- **PyTorch** (Deep Learning Framework)
- **Torchvision** (Image Processing & Augmentation)
- **Matplotlib** (Data Visualization)
- **PIL (Pillow)** (Image Manipulation)

  
## 📌 Features
1. **Training Classification Model:**  
   Run `python src/train_classification.py`.
2. **Training Segmentation Model:**  
   Run `python src/train_segmentation.py`.
3. **VQA Module Execution:**  
   Run `python src/vqa_module.py` for processing natural language queries.
4. **Integrated Inference:**  
   Run `python src/inference.py` to execute the complete pipeline and generate outputs.

## Project Contribution
- Integrated system combining classification, segmentation, and VQA for early disease detection.
- Resource-efficient model architectures optimized for CPU-only environments.
- Modular codebase for easy modifications and future improvements.

🔔 **If you found this project helpful, please ⭐ the repository!** ⭐
## Contact
For any queries or contributions, please contact Allen at allen.frederick@pau.edu.ng.
