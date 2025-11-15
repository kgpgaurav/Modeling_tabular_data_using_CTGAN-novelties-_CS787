# Modeling Tabular Data using CTGAN and TabDDPM

**Project Report:** [https://drive.google.com/drive/folders/1tdsGj4DkYS6qMNza-xL6MuL4AJis4S5J?usp=sharing](https://drive.google.com/drive/folders/1tdsGj4DkYS6qMNza-xL6MuL4AJis4S5J?usp=sharing)

This repository contains an implementation of CTGAN and an experimental Tabular Denoising Diffusion Probabilistic Model (TabDDPM) for generating synthetic tabular data. This work was developed as part of the CS787 course and builds upon the original CTGAN library.

## Project Overview

The primary goal of this project is to explore and implement advanced generative models for tabular data. This repository includes:
- A forked and modified version (with added novelties) of **CTGAN** (Conditional Tabular Generative Adversarial Network).
- A novel implementation of **TabDDPM**, a diffusion-based model for tabular data synthesis.
- An analysis notebook (`analysis.ipynb`) demonstrating the training and evaluation pipeline for these models.

## Novelties and Contributions

We have implemented five key novelties to enhance the CTGAN framework and explore alternative generative approaches:

1.  **Hybrid Generator (MLP + Transformer):** Replaced the standard MLP generator with a hybrid architecture using Transformer-based self-attention to better model contextual relationships between columns.
2.  **Adaptive Temperature for Gumbel-Softmax:** Designed a dynamic, per-column temperature schedule that adapts to each column's unique cardinality, importance (rarity), and the training time.
3.  **OneCycleLR Integration:** Integrated the OneCycleLR policy for both generator and discriminator optimizers to improve training stability and accelerate convergence.
4.  **Pluggable Normalization Framework:** Refactored the data transformer to support interchangeable continuous normalization strategies (VGM, KDE, DPM) instead of being hard-coded to only VGM.
5.  **Tabular Diffusion (TabDDPM) Model:** Designed and implemented a complete, alternative synthesizer based on a Denoising Diffusion Probabilistic Model (DDPM) to explore a non-adversarial approach to generation.

## Installation

You can install the package directly from this repository using `pip`:
```bash
pip install git+https://github.com/kgpgaurav/Modeling_tabular_data_using_CTGAN-novelties-_CS787.git
```
Alternatively, you can clone the repository and install it in editable mode:
```bash
git clone https://github.com/kgpgaurav/Modeling_tabular_data_using_CTGAN-novelties-_CS787.git
cd Modeling_tabular_data_using_CTGAN-novelties-_CS787
pip install -e .
```
You can find additional requirements in the `latest_requirements.txt` file.

## Quickstart

To get started, you can use the `CTGANSynthesizer` to learn a dataset and generate new synthetic data.

### 1. Prepare your data
First, load your data and identify the discrete columns.

```python
import pandas as pd
from ctgan import CTGANSynthesizer

# Example data
data = pd.read_csv('https://raw.githubusercontent.com/sdv-dev/CTGAN/main/examples/csv/student_placements_100.csv')
discrete_columns = ['student_degree', 'placement_status']
```

### 2. Train the Synthesizer
Train the `CTGANSynthesizer` model on your data.

```python
# Initialize and train the CTGAN model
ctgan = CTGANSynthesizer(epochs=10)
ctgan.fit(data, discrete_columns)

# Generate synthetic data
synthetic_data_ctgan = ctgan.sample(100)
```

## Repository Structure

-   **`analysis.ipynb`**: A Jupyter Notebook that contains a detailed workflow for training and evaluating the TabDDPM model.
-   **`ctgan/`**: The source code for the models.
    -   `synthesizers/ctgan.py`: The implementation of the CTGAN model with our novel additions.
    -   `synthesizers/tabddpm.py`: The core implementation of the Tabular Denoising Diffusion Probabilistic Model.
    -   `data_transformer.py`: Handles the preprocessing and transformation of tabular data.
-   **`latest_requirements.txt`**: A list of additional Python dependencies for analysis.

## Group Members

-   **Kshitij Pratap Singh** - 220554
-   **Kumar Gaurav Prakash** – 220560
-   **Sumit Neniwal** – 221103
-   **Akshit Shukrawal** - 220107
-   **M. K. S. Roshan** - 220633

## Acknowledgements

This project is inspired by and builds upon the original **CTGAN** paper and the **SDV (Synthetic Data Vault)** library. Our work extends it by implementing and evaluating a Tabular Denoising Diffusion Probabilistic Model (TabDDPM) and other enhancements.
