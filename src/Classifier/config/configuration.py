from src.Classifier.constants import *
import os
from src.Classifier.utils.common import read_yaml, create_directories
from src.Classifier.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self)-> DataTransformationConfig:
        config = self.config.data_transformation    

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            transformed_data_file=config.transformed_data_file,
            source_data_file=config.source_data_file,
            selected_columns=config.selected_columns
        )
        return data_transformation_config
    
    def get_model_trainer_config(self)-> ModelTrainerConfig:
        config = self.config.model_trainer
        model_trainer_config = ModelTrainerConfig(
            best_model_path=config.best_model_path,
            data_file_path=config.data_file_path,
            target_column=config.target_column,
            test_size=config.test_size,
            random_state=config.random_state
        )
        return model_trainer_config