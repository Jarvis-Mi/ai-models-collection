# ai-models-collection
"This repository contains my deep learning projects and models built using TensorFlow and PyTorch for various AI tasks like image classification, object detection, and audio recognition, and more model."

# Deep Learning Projects: Image Analysis and Text Processing

This repository contains various deep learning projects focused on image feature analysis and text processing. The projects primarily use **PyTorch** for image-related tasks and **TensorFlow** for time-series and text data processing. You will find projects related to computer vision, natural language processing (NLP), and time-series forecasting in this repository.

## Project Structure

Below is the folder structure used in this repository to maintain an organized and easy-to-navigate setup:


├── data/
# Raw and processed datasets
├── notebooks/
# Jupyter notebooks for experiments and exploratory analysis
├── src/
# Source code for model training, preprocessing, and utilities
│ ├── models/
# Deep learning model definitions (e.g., CNN, RNN, Transformers) 
│ ├── preprocess/
# Data preprocessing scripts (e.g., feature extraction, normalization)
│ ├── train.py
# Main script to train the models
│ ├── evaluate.py
# Script to evaluate trained models
│ └── utils.py # Helper functions
├── logs/
# Training logs and metrics
├── models/
# Saved trained models (.pth, .h5 files)
├── results/
# Results such as evaluation metrics and visualizations
├── requirements.txt
# Python dependencies
└── README.md
# This file


## File Naming Convention

To maintain consistency, the following naming conventions are used for files and scripts:
- **data/**: Contains datasets in subdirectories with descriptive names (e.g., `image_data/`, `text_data/`, `time_series_data/`).
- **notebooks/**: Jupyter notebooks are named by task and date (e.g., `image_analysis_2024_09.ipynb`, `text_classification_2024_10.ipynb`).
- **src/**:
  - Model scripts are named by the architecture and task (e.g., `cnn_image_classification.py`, `rnn_time_series.py`).
  - Preprocessing scripts are named based on the data type (e.g., `image_preprocess.py`, `text_preprocess.py`).
  - The `train.py` and `evaluate.py` scripts should handle the main training and evaluation pipelines.

## Installation Guide

To run these projects, you need to install the following dependencies using Anaconda. Follow these steps:

1. Create a new environment:
    ```bash
    conda create --name deep-learning-projects python=3.9
    conda activate deep-learning-projects
    ```

2. Install required libraries:
    ```bash
    # Install PyTorch
    conda install pytorch torchvision torchaudio -c pytorch
    
    # Install TensorFlow
    conda install -c conda-forge tensorflow
    
    # Install other libraries
    conda install scikit-learn matplotlib numpy pandas
    ```

3. To install additional requirements listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## How to Use the Projects

To train a model, simply run the corresponding `train.py` file in the `src/` folder. For example:
```bash
python src/train.py --model cnn --epochs 50 --batch_size 32

```


Feedback, suggestions, collaborations, and contributions are highly encouraged! Feel free to reach out to me via:

- **LinkedIn**: [My LinkedIn Profile](https://www.linkedin.com/in/mahdiajami/)
- **Email**: [My Gmail Address](gw2.fighter@gmail.com)
- **Instagram**: [My Instagram](@mjc.1400)

If you use these projects or find them helpful, please share your thoughts or suggestions on how they can be improved. Let's work together to enhance these deep learning models and projects!




