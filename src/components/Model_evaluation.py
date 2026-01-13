from src.components.Model_trainer import ModelTrainer
from src.logger.logger import logging
from src.exception import CustomException
from sklearn.metrics import davies_bouldin_score,silhouette_score,calinski_harabasz_score
import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
from src.utils.utils import load_obj
from urllib.parse import urlparse
from src.components.Data_ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
from src.components.Model_trainer import ModelTrainer

class ModelEvaluation:
    """
    Model Class for Evaluation    
    """
    def __init__(self,model_path):
        logging.info('Model Evaluation have been started')
        self.model_path=model_path
    
    # A method for evaluyating the metrics
    def eval_metrics(self,X:np.ndarray,labels):
        silhouette_scores=silhouette_score(X,labels)
        davies_bouldin_scores=davies_bouldin_score(X,labels)
        calinski_harabasz_scores=calinski_harabasz_score(X,labels)
        return silhouette_scores,davies_bouldin_scores,calinski_harabasz_scores
    
    def evaluate(self,X:np.ndarray):
        try:
            model=load_obj(self.model_path)
            logging.info('Model Loaded Successfully')
        except Exception as e:
            raise CustomException(e)
        labels = model.fit_predict(X)

        # Compute metrics
        sil_score, ch_score, db_score = self.eval_metrics(X, labels)
        logging.info(f"Silhouette Score: {sil_score:.4f}")
        logging.info(f"Calinski-Harabasz Index: {ch_score:.4f}")
        logging.info(f"Davies-Bouldin Index: {db_score:.4f}")

        # Log metrics to MLflow
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run(run_name="Clustering_Evaluation"):
            mlflow.log_metric("silhouette_score", sil_score)
            mlflow.log_metric(" davies_bouldin_score", db_score)
            mlflow.log_metric("calinski_harabasz_score", ch_score)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="Best_Clustering_Model")
            else:
                mlflow.sklearn.log_model(model, "model")

        return {
            "silhouette_score": sil_score,
            "calinski_harabasz_score": ch_score,
            "davies_bouldin_score": db_score
        }

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_path = data_ingestion.Instantiate_dataingestion()

    df = pd.read_csv('artifacts/raw_data.csv')

    data_transformation = DataTransformation()
    transformed_data = data_transformation.instantiate_data_transformer(df)

    # train=ModelTrainer()
    # train.train(transformed_data)
    model_evaluation=ModelEvaluation("artifacts/best_clustering_model.pkl")
    print(model_evaluation.evaluate(transformed_data))