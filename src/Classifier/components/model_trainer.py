"""
This File contains all the function which will help to train the model
and find the best model.

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.Classifier.entity.config_entity import ModelTrainerConfig
from src.Classifier import logger
from sklearn.metrics import f1_score, classification_report
import pickle
from pathlib import Path
from src.Classifier.utils.common import load_data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class ModelTrainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config = config
    

    def train_and_evaluate(self):
        data = load_data(self.config.data_file_path)

        # Load preprocessed train/test data
        X = data.drop(columns=[self.config.target_column])
        y = data[self.config.target_column]
        self.models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            ),
        }



        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size, random_state=self.config.random_state)

        f1_scores = {}
        reports = {}

        # Train and evaluate each model
        for name, model in self.models.items():
            logger.info(f"Training model: {name}")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="weighted")
            f1_scores[name] = f1
            reports[name] = classification_report(y_test, preds, output_dict=True)
            logger.info(f"{name} F1 Score: {f1:.4f}")

        print("This is the training data")
        print(X_train.columns)
        # print(y_train.columns)
        # Pick best model
        best_model_name = max(f1_scores, key=f1_scores.get)
        best_model = self.models[best_model_name]
        best_score = f1_scores[best_model_name]

        logger.info(f"Best model: {best_model_name} | F1 Score: {best_score:.4f}")

        # Save model and report
        # model_dir = Path(self.config.best_model_path) / "models"
        # model_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(best_model, open(self.config.best_model_path, 'wb'))
        logger.info(f"Saved best model at: {self.config.best_model_path}")

        # pd.DataFrame.from_dict(f1_scores, orient='index', columns=['F1_Score']).to_csv(
        #     model_dir / "model_scores.csv"
        # )

        return best_model_name, best_score, reports
        

