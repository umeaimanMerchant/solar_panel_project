"""
This file will try to find the best model 
store the model in the file system
"""

from src.Classifier.config.configuration import ConfigurationManager
from src.Classifier.components.model_trainer import ModelTrainer
from src.Classifier import logger

STAGE_NAME = "Model Training Stage"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()  

        trainer = ModelTrainer(config=model_trainer_config)
        trainer.train_and_evaluate()

        best_model, score, reports = trainer.train_and_evaluate()

        logger.info(f"Best Model: {best_model} | F1 Score: {score}")
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e

