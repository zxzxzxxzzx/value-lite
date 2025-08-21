# SOURITRA SAMANTA (3C)

import pandas as pd
import numpy as np

class HDBDataProcessor:
    def __init__(self):
        pass

    def clean_data(self, df):
        cleaned_df = df.copy() # souritra (watermark)

        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = initial_count - len(cleaned_df)

        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.dropna()
        missing_after = missing_before - cleaned_df.isnull().sum().sum()
        if missing_after > 0:
            print("[CLEANING] Removed {} records with missing values".format(missing_after))

        outliers_count = 0
        if 'resale_price' in cleaned_df.columns:
            price_col = cleaned_df['resale_price']
            Q1 = price_col.quantile(0.005)
            Q3 = price_col.quantile(0.995)

            outliers_mask = (price_col < Q1) | (price_col > Q3)
            outliers_count = outliers_mask.sum()

            cleaned_df = cleaned_df[~outliers_mask]

        if duplicates_removed > 0 or outliers_count > 0: # souritra (watermark)
            print("[CLEANING] Removed {} duplicates & {} price outliers".format(duplicates_removed, outliers_count))

        area_mask = (cleaned_df['floor_area_sqm'] >= 30) & (cleaned_df['floor_area_sqm'] <= 250)
        area_outliers = len(cleaned_df) - area_mask.sum()
        cleaned_df = cleaned_df[area_mask]
        if area_outliers > 0:
            print("[CLEANING] Removed {} unrealistic floor areas".format(area_outliers))

        categorical_columns = ['town', 'flat_type', 'storey_range', 'flat_model']
        for col in categorical_columns:
            if col in cleaned_df.columns:
                cleaned_df.loc[:, col] = cleaned_df[col].str.upper().str.strip()

        print("[SUCCESS] Final dataset: {} records".format(len(cleaned_df)))
        return cleaned_df # souritra (watermark)

    def validate_input_data(self, inputs):
        errors = []

        for key in ['floor_area_sqm', 'remaining_lease']:
            if key not in inputs:
                errors.append("[ERROR] Missing input: {}".format(key))

        if 'floor_area_sqm' in inputs:
            if inputs['floor_area_sqm'] < 30 or inputs['floor_area_sqm'] > 250:
                errors.append("[ERROR] Floor area must be 30-250 sqm")

        if 'remaining_lease' in inputs:
            if inputs['remaining_lease'] < 40 or inputs['remaining_lease'] > 99:
                errors.append("[ERROR] Remaining lease must be 40-99 yrs")

        return len(errors) == 0, errors

# SOURITRA SAMANTA (3C)
