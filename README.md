# AI-Based Early Disease Detection with Visual Q&A for Healthcare Diagnostics

## Overview
This project develops an AI system to analyze medical images (X-rays, MRIs, dermatoscopic images) for early disease detection. The system incorporates image classification, segmentation, and an interactive Visual Question Answering (VQA) module to enable clinicians to ask targeted questions about the images and receive interpretable responses.

## Objectives
- **Classification:** Detect abnormalities using a lightweight CNN (e.g., MobileNetV2, DenseNet-121) with transfer learning.
- **Segmentation:** highligh abnormal regions using U-Net/UNet++ with modifications for CPU-efficient deployment.
- **Visual Q&A:** Enable interactive queries on the processed images using a transformer-based VQA module optimized for resource-limited environments.

## Datasets
- **NIH Chest X-ray Dataset:** [Link to dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)
- **ISIC Skin Lesion Dataset:** [Link to dataset](https://www.isic-archive.com)
- **VQA Datasets:** VQA-RAD: [Link to dataset](https://osf.io/89kps) and PMC-VQA: [Link to dataset](https://huggingface.co/datasets/RadGenome/PMC-VQA)


## Setup Instructions

### Software Dependencies
Ensure you have Python 3.8+ installed. Install dependencies using:

*Dependencies include:* PyTorch, torchvision, OpenCV, Pillow, NumPy, SciPy, scikit-learn, transformers, matplotlib, and pandas.

### Datasets
Download the required datasets and place them in the `datasets/` directory following the structure provided above.

### Running the Project
1. **Preprocessing:**  
   Run `python src/preprocessing.py` to normalize, resize, and augment images.
2. **Training Classification Model:**  
   Run `python src/train_classification.py`.
3. **Training Segmentation Model:**  
   Run `python src/train_segmentation.py`.
4. **VQA Module Execution:**  
   Run `python src/vqa_module.py` for processing natural language queries.
5. **Integrated Inference:**  
   Run `python src/inference.py` to execute the complete pipeline and generate outputs.

## Project Contribution
- Integrated system combining classification, segmentation, and VQA for early disease detection.
- Resource-efficient model architectures optimized for CPU-only environments.
- Modular codebase for easy modifications and future improvements.

## Contact
For any queries or contributions, please contact Allen at allen.frederick@pau.edu.ng.
