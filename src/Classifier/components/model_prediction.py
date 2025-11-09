"""
This file is having all the functions required for model prediction.

"""

import pandas as pd
import pickle
from src.Classifier.components.data_transformation import DataTransformation
from src.Classifier.config.configuration import ConfigurationManager
import numpy as np

class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.model_config = config.get_model_prediction_config()
        self.data_transformation = DataTransformation(config=config.get_data_transformation_config())
        self.model =  pickle.load(open(self.model_config.model_path, 'rb'))
        self.features = FEATURE_COLUMNS = [
            'Acres',
            'Distance to Substation (Miles) CAISO',
            'Distance to Substation (Miles) GTET 100 Max Voltage',
            'Distance to Substation (Miles) GTET 200 Max Voltage',
            'Shape__Length',
            'Shape__Area',
            'Percentile (GTET 100 Max Voltage)_25th to 50th',
            'Percentile (GTET 100 Max Voltage)_50th to 75th',
            'Percentile (GTET 100 Max Voltage)_75th to 100th',
            'Percentile (GTET 200 Max Voltage)_25th to 50th',
            'Percentile (GTET 200 Max Voltage)_50th to 75th',
            'Percentile (GTET 200 Max Voltage)_75th to 100th',
            'Percentile (CAISO)_25th to 50th',
            'Percentile (CAISO)_50th to 75th',
            'Percentile (CAISO)_75th to 100th',
            'Install Type_Parking',
            'Install Type_Rooftop',
            'Urban or Rural_Urban',
            'Region_Central California',
            'Region_Inland Deserts',
            'Region_North Central California',
            'Region_Northern California',
            'Region_South Coast'
        ]

            
    def scalarize_input(self, input_data: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
        scaler = pickle.load(open(self.model_config.scaler_path, 'rb'))
        scaled_data = scaler.transform(input_data[numerical_features])
        input_data = input_data.drop(columns=numerical_features).reset_index(drop=True)
        scaled_data = pd.concat([input_data, pd.DataFrame(scaled_data, columns=numerical_features)], axis=1)
        return pd.DataFrame(scaled_data)

    def encode_categorical_features(self, input_data: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
        encoder = pickle.load(open(self.model_config.encoder_path, 'rb'))
        # temp add solar to avoid error
        input_data['Solar Technoeconomic Intersection'] = 'Within'
        encoded_data = encoder.transform(input_data[categorical_features+['Solar Technoeconomic Intersection']])
        input_data = input_data.drop(columns=categorical_features).reset_index(drop=True)
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features+['Solar Technoeconomic Intersection']))
        input_data = pd.concat([input_data, encoded_df], axis=1)
        input_data.drop(columns=['Solar Technoeconomic Intersection_Within'], inplace=True)
        return pd.DataFrame(input_data)
    
    def log_transform(self, input_data: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
        for feature in numerical_features:
            if (input_data[feature] <= 0).any():
                input_data[feature] = input_data[feature] - input_data[feature].min() + 1
            input_data[feature] = input_data[feature].apply(lambda x: np.log(x))
        return input_data
        
    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([input_data])
        df = self.data_transformation.transform_county_to_region(df)
        df = df.drop(columns=['County'])
        # Apply the same transformations as during training
        numerical_features = [
            'Distance to Substation (Miles) GTET 100 Max Voltage',
            'Distance to Substation (Miles) GTET 200 Max Voltage',
            'Distance to Substation (Miles) CAISO',
            'Shape__Length',
            'Acres',
            'Shape__Area'
        ]
        categorical_features = [
            'Percentile (GTET 100 Max Voltage)',
            'Percentile (GTET 200 Max Voltage)',
            'Percentile (CAISO)',
            'Install Type',
            'Urban or Rural',
            'Region'
        ]

        df = self.log_transform(df, numerical_features)
        df = self.scalarize_input(df, numerical_features)
        df = self.encode_categorical_features(df, categorical_features)
        df = df.drop(columns=['Solar Technoeconomic Intersection'])
        return df
    
    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add missing columns
        for col in self.features:
            if col not in df.columns:
                print(f"Adding missing column: {col}")
                df[col] = 0  

        # Remove extra columns
        df = df[self.features]
        return df


    def predict(self, input_data: dict):
        df_preprocessed = self.preprocess_input(input_data)
        intput_final = self.validate_input(df_preprocessed)
        prediction = self.model.predict(intput_final)
        return prediction[0]
