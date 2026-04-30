================================================================================
    Network Traffic Monitoring and Detection System
================================================================================

Module:     6Z0019 Synoptic Project
University: Manchester Metropolitan University (MMU)

--------------------------------------------------------------------------------
DESCRIPTION
--------------------------------------------------------------------------------

This system is a local web-based application built with Python and Streamlit
that analyses publicly available network traffic datasets to identify suspicious
or malicious behaviour. It uses the CIC-IDS-2017 dataset and applies machine
learning techniques (Random Forest and Isolation Forest) to detect anomalous
network traffic patterns.

Key capabilities:
  - Upload and preview network traffic datasets (CSV format)
  - Preprocess data (handle missing values, duplicates, type correction)
  - Extract relevant traffic features (flow duration, packet counts, byte
    counts, flow rates, and derived features)
  - Visualise traffic patterns through an interactive dashboard
  - Detect suspicious traffic using Random Forest (supervised) or Isolation
    Forest (unsupervised) detection methods
  - Evaluate detection performance with accuracy, precision, recall, F1-score,
    and confusion matrix
  - Export detection results as downloadable CSV files

--------------------------------------------------------------------------------
PREREQUISITES
--------------------------------------------------------------------------------

  - Python 3.8 or higher
  - pip (Python package manager)
  - A modern web browser (Chrome, Firefox, Edge, or Safari)

--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------

1. Open a terminal and navigate to the project directory:

       cd network_ids_project

2. Install the required dependencies:

       pip install -r requirements.txt

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------

From the project directory, run:

    streamlit run app.py

The application will open automatically in your default web browser at:

    http://localhost:8501

--------------------------------------------------------------------------------
HOW TO USE THE SYSTEM
--------------------------------------------------------------------------------

The application follows a step-by-step workflow accessible via the sidebar
navigation:

1. HOME
   - View the project overview and instructions.

2. DATA UPLOAD
   - Upload a CSV dataset using the file uploader, or click "Use Sample
     Dataset" to load the built-in sample data for quick testing.
   - After loading, view the dataset preview, row/column counts, column names,
     data types, and basic statistics.

3. PREPROCESSING
   - Click "Run Preprocessing" to clean the loaded dataset.
   - The system will remove duplicates, handle missing and infinite values,
     clean column names, correct data types, and remove constant columns.
   - A summary of all preprocessing actions is displayed.

4. FEATURE EXTRACTION
   - Click "Extract Features" to select and engineer key traffic features from
     the dataset.
   - View the list of selected features, summary statistics, and a correlation
     heatmap.

5. TRAFFIC DASHBOARD
   - Explore interactive visualisations of traffic patterns including:
     protocol distribution, label distribution (benign vs attack types),
     top source/destination IPs, and feature distribution plots.
   - Use the dropdown to select different features for distribution analysis.

6. DETECTION
   - Choose a detection method: Random Forest or Isolation Forest.
   - For Random Forest: toggle between binary and multi-class classification.
   - For Isolation Forest: adjust the contamination parameter using the slider.
   - Click "Run Detection" to train the model and generate predictions.
   - View the detection summary and results table.

7. EVALUATION
   - View performance metrics: accuracy, precision, recall, and F1-score.
   - Examine the confusion matrix heatmap.
   - For Random Forest: view the classification report and feature importance
     chart.
   - If both methods have been run: view a side-by-side comparison.

8. EXPORT
   - Download detection results as a CSV file.

--------------------------------------------------------------------------------
DATASET INFORMATION
--------------------------------------------------------------------------------

This system is designed to work with the CIC-IDS-2017 dataset:

  - Name:    Network Intrusion Dataset (CIC-IDS-2017)
  - Source:   Canadian Institute for Cybersecurity
  - Content: Labelled network traffic flows including both benign traffic and
             various attack types (DDoS, PortScan, Bot, Brute Force, etc.)
  - Format:  CSV with 78+ feature columns and a Label column

The dataset can be downloaded from Kaggle:
  https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/data

A small sample dataset (500 rows) is included in the sample_data/ folder for
quick testing without downloading the full dataset.

--------------------------------------------------------------------------------
FOLDER STRUCTURE
--------------------------------------------------------------------------------

network_ids_project/
    app.py                  Main Streamlit application entry point
    requirements.txt        Python package dependencies
    README.txt              This file - setup and usage instructions
    sample_data/
        sample_dataset.csv  Small dataset for quick testing
    modules/
        __init__.py         Package initialiser
        data_loader.py      Dataset upload and loading functions
        preprocessing.py    Data cleaning and preprocessing functions
        feature_extraction.py  Feature selection and engineering functions
        detection.py        Random Forest and Isolation Forest detection
        evaluation.py       Performance metrics and evaluation functions
        visualisation.py    Chart and plot generation functions
    assets/                 Optional branding assets
    exports/                Directory for exported result files

================================================================================
