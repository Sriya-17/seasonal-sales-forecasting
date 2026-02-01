"""CSV validation and processing utilities.

This module provides robust, user-friendly CSV validation for sales datasets.
It accepts flexible date formats (day-first and ISO), validates numeric sales
values, and standardizes the output DataFrame for downstream processing.
"""

import os
from dateutil import parser as date_parser
import pandas as pd


def _try_parse_dates(series):
    """Try to parse a pandas Series of date-like values flexibly.

    Strategy:
    - First use pandas.to_datetime with dayfirst=True.
    - For remaining unparsable values, fall back to dateutil.parser.parse
      trying dayfirst=True then dayfirst=False per value.

    Returns a tuple (parsed_series, num_unparsed, sample_unparsed_values).
    """
    parsed = pd.to_datetime(series, dayfirst=True, errors='coerce')
    if parsed.notna().all():
        return parsed, 0, []

    # Identify unparsable entries
    mask_unparsed = parsed.isna()
    unparsed_vals = series[mask_unparsed].astype(str)

    # Attempt per-value parsing with dateutil
    parsed_list = parsed.tolist()
    for idx, val in zip(unparsed_vals.index, unparsed_vals.values):
        parsed_val = None
        try:
            parsed_val = date_parser.parse(val, dayfirst=True)
        except Exception:
            try:
                parsed_val = date_parser.parse(val, dayfirst=False)
            except Exception:
                parsed_val = None
        parsed_list[idx] = parsed_val

    parsed_series = pd.to_datetime(pd.Series(parsed_list, index=series.index))
    remaining_unparsed = parsed_series.isna().sum()
    sample_unparsed = series[parsed_series.isna()].astype(str).unique().tolist()[:5]
    return parsed_series, remaining_unparsed, sample_unparsed


def validate_csv_structure(filepath):
    """
    Validate CSV file structure and required columns.

    Returns:
        tuple: (is_valid: bool, message: str, df: pd.DataFrame|None)
    """
    try:
        df = pd.read_csv(filepath)

        if df.empty:
            return False, "CSV file is empty.", None

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Detect required columns (flexible)
        date_col = next((col for col in df.columns if 'date' in col), None)
        sales_col = next((col for col in df.columns if 'sales' in col or 'weekly_sales' in col), None)

        if not date_col:
            return False, "Missing required column: Date.", None
        if not sales_col:
            return False, "Missing required column: Sales.", None

        # Parse dates flexibly
        parsed_dates, num_unparsed, sample_unparsed = _try_parse_dates(df[date_col].astype(str))
        if num_unparsed == len(df):
            sample_vals = df[date_col].astype(str).unique().tolist()[:5]
            return False, ("Unable to parse any dates from the Date column. "
                           f"Sample values: {sample_vals}. Please provide dates in common formats "
                           "(e.g. DD-MM-YYYY, YYYY-MM-DD, DD/MM/YYYY)."), None

        df[date_col] = parsed_dates

        # Convert sales to numeric, coerce errors
        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')

        # Drop rows with missing critical values
        before_count = len(df)
        df = df.dropna(subset=[date_col, sales_col]).copy()
        dropped_count = before_count - len(df)

        if df.empty:
            return False, "After removing invalid rows there is no data to process.", None

        # Rename columns to standardized names
        df = df.rename(columns={date_col: 'Date', sales_col: 'Weekly_Sales'})

        # Ensure Store column exists
        store_col = next((col for col in df.columns if 'store' in col), None)
        if store_col and store_col != 'store':
            df = df.rename(columns={store_col: 'Store'})
        if 'Store' not in df.columns:
            df['Store'] = 1

        # Standardize Date to timezone-naive datetimes
        try:
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_convert(None)
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Sort chronologically
        df = df.sort_values('Date').reset_index(drop=True)

        msg = f"CSV validated successfully. {len(df)} records available."
        if dropped_count > 0:
            msg += f" {dropped_count} invalid rows were removed (bad/missing dates or sales)."
        if num_unparsed > 0 and num_unparsed < before_count:
            msg += f" Note: {num_unparsed} date values could not be parsed and were removed."

        return True, msg, df

    except pd.errors.ParserError as e:
        return False, f"CSV parsing error: {e}", None
    except Exception as e:
        return False, f"Error reading file: {e}", None


def allowed_file(filename):
    """Check if file is a CSV by extension."""
    return isinstance(filename, str) and filename.lower().endswith('.csv')


def process_and_save_upload(uploaded_df, filepath):
    """Save processed DataFrame to `filepath` as CSV."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        uploaded_df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False
