from src.Classifier.config.configuration import ConfigurationManager
from src.Classifier.components.data_transformation import DataTransformation
from src.Classifier import logger
from src.Classifier.utils.common import load_data

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline: 
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_loaded = load_data(data_transformation_config.source_data_file)
        print(f"Data loaded with shape: {data_loaded.shape}")
        print(data_loaded.head())
        # Region column creation
        data_loaded = data_transformation.transform_county_to_region(data_loaded)
        columns_to_select = data_transformation_config.selected_columns 

        data_loaded = data_transformation.columns_selector(data_loaded, columns_to_select)

        categorical_features = [
            'Percentile (GTET 100 Max Voltage)',
            'Percentile (GTET 200 Max Voltage)',
            'Percentile (CAISO)',
            'Install Type',
            'Urban or Rural',
            'Region',
            'Solar Technoeconomic Intersection'
        ]

        numerical_features = [
            'Distance to Substation (Miles) GTET 100 Max Voltage',
            'Distance to Substation (Miles) GTET 200 Max Voltage',
            'Distance to Substation (Miles) CAISO',
            'Shape__Length',
            'Acres',
            'Shape__Area'
        ]

        data_imputed = data_transformation.impute_missing_values(data_loaded, numerical_features, categorical_features)
        data_log_transformed = data_transformation.log_transform(data_imputed, numerical_features)
        data_scaled = data_transformation.scale_features(data_log_transformed, numerical_features)
        data_encoded = data_transformation.encode_categorical_features(data_scaled, categorical_features)
        data_transformation.store_scaled_data(data_encoded)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e