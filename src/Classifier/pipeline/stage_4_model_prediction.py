"""
This is the stage 4 model prediction module which will handle the model prediction tasks.
Apps will call this pipeline stage for making predictions using the trained model.
"""


from src.Classifier.components.model_prediction import PredictionPipeline
from src.Classifier import logger

STAGE_NAME = "Prediction Stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        input_sample = {
            "Acres": 25.0,
            "Distance to Substation (Miles) CAISO": 1.5,
            "Distance to Substation (Miles) GTET 100 Max Voltage": 2.3,
            "Distance to Substation (Miles) GTET 200 Max Voltage": 3.2,
            "Shape__Length": 1234.56,
            "Shape__Area": 1.2,
            "Percentile (GTET 100 Max Voltage)": "0 to 25th",
            "Percentile (GTET 200 Max Voltage)": "0 to 25th",
            "Percentile (CAISO)": "0 to 25th",
            "Install Type": "Rooftop",
            "Urban or Rural": "Rural",
            "County": "Del Norte County"
        }

        pipeline = PredictionPipeline()
        prediction = pipeline.predict(input_sample)

        logger.info(f"Prediction Output: {prediction}")
        print(f"Predicted Class: {prediction}")

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e
