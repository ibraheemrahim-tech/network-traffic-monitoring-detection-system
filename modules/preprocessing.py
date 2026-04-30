"""
Preprocessing Module
Handles data cleaning, missing value treatment, and column normalisation.
"""

import pandas as pd
import numpy as np


def preprocess_data(df):
    """
    Preprocess the raw dataset.
    """
    df = df.copy()
    summary = {
        "duplicates_removed": 0,
        "columns_dropped": 0,
        "missing_values_filled": 0,
        "infinite_values_replaced": 0,
        "constant_columns_removed": 0,
        "log": []
    }

    original_rows = len(df)
    original_cols = len(df.columns)

    # 1. Clean column names - strip whitespace, replace spaces with underscores, lowercase
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.lower()
    )
    summary["log"].append("Column names cleaned (stripped, lowercased, spaces replaced with underscores).")

    # 2. Remove duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        summary["duplicates_removed"] = int(dup_count)
        summary["log"].append(f"Removed {dup_count:,} duplicate rows.")

    # 3. Handle infinite values - replace with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        summary["infinite_values_replaced"] = int(inf_count)
        summary["log"].append(f"Replaced {inf_count:,} infinite values with NaN.")

    # 4. Drop columns with >50% missing values
    missing_pct = df.isna().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        summary["columns_dropped"] += len(cols_to_drop)
        summary["log"].append(
            f"Dropped {len(cols_to_drop)} column(s) with >50% missing values: {', '.join(cols_to_drop)}"
        )

    # 5. Convert numeric columns - coerce errors
    for col in df.columns:
        if df[col].dtype == object:
            # Try to convert string columns that look numeric
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only convert if majority of values are successfully converted
            if converted.notna().sum() > 0.5 * df[col].notna().sum():
                non_null_before = df[col].notna().sum()
                non_null_after = converted.notna().sum()
                # Only convert if we don't lose too many values
                if non_null_after >= 0.8 * non_null_before:
                    df[col] = converted

    # 6. Fill missing values
    fill_count = 0

    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            fill_count += null_count

    # Categorical columns: fill with mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
                fill_count += null_count

    if fill_count > 0:
        summary["missing_values_filled"] = int(fill_count)
        summary["log"].append(
            f"Filled {fill_count:,} missing values (numeric: median, categorical: mode)."
        )

    # 7. Remove constant (zero-variance) columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        df = df.drop(columns=constant_cols)
        summary["constant_columns_removed"] = len(constant_cols)
        summary["columns_dropped"] += len(constant_cols)
        summary["log"].append(
            f"Removed {len(constant_cols)} constant column(s): {', '.join(constant_cols)}"
        )

    summary["log"].append(
        f"Preprocessing complete: {original_rows:,} -> {len(df):,} rows, "
        f"{original_cols} -> {len(df.columns)} columns."
    )

    return df, summary
