"""
Data Loader Module
Handles dataset upload, loading, and initial validation.
"""

import pandas as pd
import os


def load_dataset(uploaded_file):
    """
    Load a CSV dataset from an uploaded file.
    """
    try:
        # Try UTF-8 first, then Latin-1
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1")

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Check file size and sample if needed
        file_size_mb = uploaded_file.size / (1024 * 1024) if hasattr(uploaded_file, 'size') else 0

        if len(df) > 100000 and file_size_mb > 500:
            df = df.sample(n=100000, random_state=42).reset_index(drop=True)
            return df, (
                f"Dataset loaded successfully. Due to large file size ({file_size_mb:.0f} MB), "
                f"a random sample of 100,000 rows has been selected."
            )

        return df, f"Dataset loaded successfully. {len(df):,} rows and {len(df.columns)} columns."

    except pd.errors.EmptyDataError:
        return None, "The uploaded file is empty. Please upload a valid CSV file."
    except pd.errors.ParserError:
        return None, "Failed to parse the CSV file. Please check the file format."
    except Exception as e:
        return None, f"An error occurred while loading the dataset: {str(e)}"


def load_sample_dataset():
    """
    Load the built-in sample dataset.
    """
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_dataset.csv")

    if not os.path.exists(sample_path):
        return None, "Sample dataset not found. Please ensure sample_data/sample_dataset.csv exists."

    try:
        df = pd.read_csv(sample_path)
        df.columns = df.columns.str.strip()
        return df, f"Sample dataset loaded successfully. {len(df):,} rows and {len(df.columns)} columns."
    except Exception as e:
        return None, f"An error occurred while loading the sample dataset: {str(e)}"
