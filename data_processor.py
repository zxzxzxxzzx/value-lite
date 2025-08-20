import pandas as pd
import numpy as np

class HDBDataProcessor:
    def __init__(self):
        pass
    
    def clean_data(self, df):
        cleaned_df = df.copy()
        
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = initial_count - len(cleaned_df)
        if duplicates_removed > 0:
            print("🧹 Removed {} duplicate records".format(duplicates_removed))
        
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.dropna()
        missing_after = missing_before - cleaned_df.isnull().sum().sum()
        if missing_after > 0:
            print("🧹 Removed {} records with missing values".format(missing_after))
        
        if 'resale_price' in cleaned_df.columns:
            price_col = cleaned_df['resale_price']
            Q1 = price_col.quantile(0.005)
            Q3 = price_col.quantile(0.995)
            
            outliers_mask = (price_col < Q1) | (price_col > Q3)
            outliers_count = outliers_mask.sum()
            
            cleaned_df = cleaned_df[~outliers_mask]
            if outliers_count > 0:
                print("🧹 Removed {} price outliers for polynomial regression".format(outliers_count))
        
        if 'floor_area_sqm' in cleaned_df.columns:
            area_mask = (cleaned_df['floor_area_sqm'] >= 30) & (cleaned_df['floor_area_sqm'] <= 250)
            area_outliers = len(cleaned_df) - area_mask.sum()
            cleaned_df = cleaned_df[area_mask]
            if area_outliers > 0:
                print("🧹 Removed {} unrealistic floor areas".format(area_outliers))
        
        categorical_columns = ['town', 'flat_type', 'storey_range', 'flat_model']
        for col in categorical_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].str.upper().str.strip()
        
        print("✅ Data cleaning completed. Final dataset: {} records".format(len(cleaned_df)))
        return cleaned_df
    
    def validate_input_data(self, inputs):
        errors = []
        
        required_fields = ['town', 'flat_type', 'floor_area_sqm', 
                          'storey_range', 'flat_model', 'remaining_lease']
        
        for field in required_fields:
            if field not in inputs or inputs[field] is None:
                errors.append("Missing required field: {}".format(field))
        
        if 'floor_area_sqm' in inputs:
            if inputs['floor_area_sqm'] < 30 or inputs['floor_area_sqm'] > 250:
                errors.append("Floor area must be between 30 and 250 sqm")
        
        if 'remaining_lease' in inputs:
            if inputs['remaining_lease'] < 40 or inputs['remaining_lease'] > 99:
                errors.append("Remaining lease must be between 40 and 99 years")
        
        return len(errors) == 0, errors