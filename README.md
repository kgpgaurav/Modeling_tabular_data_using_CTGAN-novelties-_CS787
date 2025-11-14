Project Report: https://drive.google.com/drive/folders/1tdsGj4DkYS6qMNza-xL6MuL4AJis4S5J?usp=sharing

# Modeling Tabular Data using CTGAN and TabDDPM

This repository contains the implementation and analysis of models for generating synthetic tabular data, with a focus on a Tabular Denoising Diffusion Probabilistic Model (TabDDPM). The project evaluates the performance of TabDDPM on various datasets and provides a framework for training the model and generating synthetic data. This work was developed as part of the CS787 course.

## Project Overview

The primary goal of this project is to explore and implement advanced generative models for tabular data. The core of this repository is the `TabDDPM` model, a diffusion-based model adapted for tabular data synthesis. The project also includes the well-known CTGAN (Conditional Tabular Generative Adversarial Network) for comparison.

The main analysis is conducted in the `analysis.ipynb` notebook, which covers the entire pipeline from data loading and preprocessing to model training, synthetic data generation, and evaluation.

## Repository Structure

```
.
├── analysis.ipynb
├── ctgan
│   ├── __init__.py
│   ├── __main__.py
│   ├── data.py
│   ├── data_sampler.py
│   ├── data_transformer.py
│   ├── demo.py
│   ├── errors.py
│   ├── normalizers.py
│   └── synthesizers
│       ├── __init__.py
│       ├── _utils.py
│       ├── base.py
│       ├── ctgan.py
│       ├── ctgan_OCLR.py
│       ├── hybrid_generator.py
│       ├── tabddpm.py
│       └── tvae.py
├── latest_requirements.txt
└── README.md
```

-   **`analysis.ipynb`**: A Jupyter Notebook that contains the main workflow for training and evaluating the TabDDPM model.
-   **`ctgan/`**: A directory containing the source code for the CTGAN and TabDDPM models.
    -   `synthesizers/tabddpm.py`: The core implementation of the Tabular Denoising Diffusion Probabilistic Model.
    -   `synthesizers/ctgan.py`: The implementation of the CTGAN model.
    -   `data_transformer.py`: Handles the preprocessing and transformation of tabular data for the models.
-   **`latest_requirements.txt`**: A list of Python dependencies required to run the project.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:

```bash
pip install -r latest_requirements.txt
```

### Usage

The main entry point for this project is the `analysis.ipynb` notebook. To run the analysis:

1.  **Place your datasets** in a `datasets/` directory in the root of the project. The notebook is configured to load datasets such as `adult.csv`, `covertype.csv`, etc.
2.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
3.  Open `analysis.ipynb` and run the cells sequentially to:
    -   Load and preprocess the data.
    -   Train the `TabDDPM` model.
    -   Generate synthetic data.
    -   Evaluate the quality of the generated data.

## Models

This repository includes implementations of the following models:

-   **TabDDPM (`ctgan/synthesizers/tabddpm.py`)**: A Denoising Diffusion Probabilistic Model adapted for tabular data. The architecture and training process are detailed in the `analysis.ipynb` notebook.
-   **CTGAN (`ctgan/synthesizers/ctgan.py`)**: A conditional GAN for tabular data, used as a baseline for comparison.

## Analysis and Evaluation

The `analysis.ipynb` notebook provides a comprehensive analysis of the TabDDPM model's performance. The key steps in the notebook are:

1.  **Data Loading**: Loads datasets and identifies discrete and continuous columns.
2.  **Data Preprocessing**: Splits the data and applies `StandardScaler` to normalize continuous features.
3.  **Model Training**: Initializes and trains the `TabDDPM` model. The training progress and loss are monitored.
4.  **Synthetic Data Generation**: Samples synthetic data from the trained model.
5.  **Evaluation**:
    -   Performs a diversity check to compare the number of unique values between real and synthetic data.
    -   Uses the `sdmetrics` library to generate a quality report, evaluating the statistical similarity and privacy of the synthetic data.

## Group Members

-   **Kshitij Pratap Singh** - 220554
-   **Kumar Gaurav Prakash** – 220560
-   **Sumit Neniwal** – 221103
-   **Akshit Shukrawal** - 220107
-   **M. K. S. Roshan** - 220633

## Acknowledgements

This project is inspired by the original CTGAN paper and the growing field of diffusion models for generative tasks. The code in the `ctgan` directory is based on the SDV (Synthetic Data Vault) library.
