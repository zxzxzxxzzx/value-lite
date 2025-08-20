import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class HDBPolynomialPriceModel:
    def __init__(self):
        self.polynomial_pipeline = None
        self.label_encoders = {}
        self.df = None
        self.feature_names = []
        self.is_trained = False
        self.model_metrics = {}
        self.polynomial_degree = 4

    def load_data(self, filepath='sample_data.csv'):
        if not os.path.exists(filepath):
            raise FileNotFoundError("Data file {} not found".format(filepath))
        
        self.df = pd.read_csv(filepath)
        print("Loaded {} HDB records from {}".format(len(self.df), filepath))
        return self.df

    def preprocess_data(self):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        processed_df = self.df.copy()
        
        processed_df = processed_df.dropna()
        
        categorical_columns = ['town', 'flat_type', 'storey_range', 'flat_model']
        
        for col in categorical_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].str.upper().str.strip()
                
                le = LabelEncoder()
                processed_df[col + '_encoded'] = le.fit_transform(processed_df[col])
                self.label_encoders[col] = le
        
        feature_columns = [col + '_encoded' for col in categorical_columns if col in processed_df.columns]
        feature_columns.extend(['floor_area_sqm', 'remaining_lease'])
        
        available_columns = [col for col in feature_columns if col in processed_df.columns]
        
        X = processed_df[available_columns]
        y = processed_df['resale_price']
        
        self.feature_names = available_columns
        return X, y

    def train_model(self):
        X, y = self.preprocess_data()
        
        print("Created polynomial pipeline with degree {}".format(self.polynomial_degree))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.polynomial_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)),
            ('regressor', LinearRegression())
        ])
        
        print("Training {}th degree polynomial regression model...".format(self.polynomial_degree))
        self.polynomial_pipeline.fit(X_train, y_train)
        
        train_predictions = self.polynomial_pipeline.predict(X_train)
        test_predictions = self.polynomial_pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        n_polynomial_features = self.polynomial_pipeline.named_steps['poly'].n_output_features_
        
        self.model_metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'n_samples': len(X_train),
            'n_features': len(self.feature_names),
            'n_polynomial_features': n_polynomial_features
        }
        
        self.is_trained = True
        print("Polynomial model trained successfully!")
        return self.model_metrics

    def predict_price(self, inputs):
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        input_data = {}
        categorical_columns = ['town', 'flat_type', 'storey_range', 'flat_model']
        
        for col in categorical_columns:
            if col in inputs:
                value = str(inputs[col]).upper().strip()
                if col in self.label_encoders:
                    if value not in self.label_encoders[col].classes_:
                        available_values = list(self.label_encoders[col].classes_)
                        raise ValueError("Unknown {} '{}'. Available options: {}".format(col, value, available_values))
                    encoded_value = self.label_encoders[col].transform([value])[0]
                    input_data[col + '_encoded'] = encoded_value
        
        input_data['floor_area_sqm'] = float(inputs['floor_area_sqm'])
        input_data['remaining_lease'] = float(inputs['remaining_lease'])
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df[self.feature_names]
        
        prediction = self.polynomial_pipeline.predict(input_df)[0]
        
        feature_contributions = {}
        for i, feature_name in enumerate(self.feature_names):
            contribution = input_df.iloc[0, i] * 10000
            readable_name = feature_name.replace('_encoded', '').replace('_', ' ').title()
            feature_contributions[readable_name] = contribution
        
        return prediction, feature_contributions

    def get_model_metrics(self):
        return self.model_metrics

    def get_polynomial_equation_info(self):
        if not self.is_trained:
            return {}
        
        n_features = self.polynomial_pipeline.named_steps['poly'].n_output_features_
        
        return {
            'degree': self.polynomial_degree,
            'n_features': n_features,
            'feature_names': self.feature_names
        }

    def get_available_towns(self):
        if 'town' in self.label_encoders:
            return sorted(list(self.label_encoders['town'].classes_))
        return []

    def get_available_flat_types(self):
        if 'flat_type' in self.label_encoders:
            return sorted(list(self.label_encoders['flat_type'].classes_))
        return []

    def get_available_storey_ranges(self):
        if 'storey_range' in self.label_encoders:
            return sorted(list(self.label_encoders['storey_range'].classes_))
        return []

    def get_available_flat_models(self):
        if 'flat_model' in self.label_encoders:
            return sorted(list(self.label_encoders['flat_model'].classes_))
        return []