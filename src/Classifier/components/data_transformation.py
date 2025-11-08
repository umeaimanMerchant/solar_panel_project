"""
This file is used for data transformation components
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.Classifier.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def columns_selector(self, data: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
        """
        Select specific columns from the dataframe
        """
        return data[selected_columns]
    
    def transform_county_to_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a county column into a region classification.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            county_col (str): Name of the column containing county names
            
        Returns:
            pd.DataFrame: DataFrame with an added 'Region' column
        """
        # Define region groups
        northern = ["Del Norte","Humboldt","Lassen","Mendocino","Modoc","Shasta","Siskiyou","Tehama","Trinity"]
        north_central = ["Alpine","Amador","Butte","Calaveras","Colusa","El Dorado","Glenn","Lake","Nevada",
                        "Placer","Plumas","Sacramento","San Joaquin","Sierra","Sutter","Yolo","Yuba"]
        bay_delta = ["Alameda","Contra Costa","Marin","Napa","San Mateo","San Francisco",
                    "Santa Clara","Santa Cruz","Solano","Sonoma"]
        central = ["Fresno","Kern","Kings","Madera","Mariposa","Merced","Monterey","San Benito",
                "San Luis Obispo","Stanislaus","Tulare","Tuolumne"]
        south_coast = ["Los Angeles","Orange","San Diego","Santa Barbara","Ventura"]
        inland_deserts = ["Imperial","Inyo","Mono","Riverside","San Bernardino"]

        # Helper function
        def categorize_county(county: str) -> str:
            county_clean = county.replace(" County", "").strip()
            if county_clean in northern:
                return 'Northern California'
            elif county_clean in north_central:
                return 'North Central California'
            elif county_clean in bay_delta:
                return 'Bay & Delta'
            elif county_clean in central:
                return 'Central California'
            elif county_clean in south_coast:
                return 'South Coast'
            elif county_clean in inland_deserts:
                return 'Inland Deserts'
            else:
                return 'Other'

        # Apply transformation
        df = df.copy()
        df["Region"] = df['County'].apply(categorize_county)

        return df

    def impute_missing_values(self, data: pd.DataFrame, numerical_features: list, categorical_features: list) -> pd.DataFrame:
        """
        Impute missing values in numerical and categorical features
        """
        for feature in numerical_features:
            imputer = SimpleImputer(strategy='mean')
            data[feature] = imputer.fit_transform(data[[feature]]).ravel()
        
        for feature in categorical_features:
            imputer = SimpleImputer(strategy='most_frequent')
            data[feature] = imputer.fit_transform(data[[feature]]).ravel()
        
        return data
    
    def log_transform(self, data: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
        """
        Apply log transformation to skewed numerical features
        """
        for feature in numerical_features:
            if (data[feature] <= 0).any():
                data[feature] = data[feature] - data[feature].min() + 1
            data[feature] = data[feature].apply(lambda x: np.log(x))
        return data
    
    def scale_features(self, data: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler
        """
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
        """
        One-hot encode categorical features
        """
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        encoded_data = encoder.fit_transform(data[categorical_features])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
        data = data.drop(columns=categorical_features).reset_index(drop=True)
        data = pd.concat([data, encoded_df], axis=1)
        return data
    
    def store_scaled_data(self, data: pd.DataFrame):
        """
        Store the scaled data to a CSV file
        """
        data.to_csv(self.config.transformed_data_file, index=False)

    