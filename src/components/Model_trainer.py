import os
from dataclasses import dataclass
import numpy as np
import pickle
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from src.components.Data_ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
import pandas as pd

import mlflow
import mlflow.sklearn

import dagshub

from src.logger.logger import logging
from src.exception import CustomException
from src.utils.utils import save_obj

# ============================
# DAGsHub Init
# ============================
dagshub.init(
    repo_owner='aky-ds',
    repo_name='Spotify-Music-Recommendation-System',
    mlflow=True
)

# ============================
# Config
# ============================
@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join("artifacts", "clustering_models")
    best_model_path: str = os.path.join("artifacts", "best_clustering_model.pkl")


# ============================
# Clustering Trainer
# ============================
class ModelTrainer:
    """
    Class for training clustering models and selecting the best based on silhouette score
    """
    def __init__(self):
        self.modeltrainconfig = ModelTrainerConfig()
        logging.info("Clustering directory initialized")
        os.makedirs(self.modeltrainconfig.model_dir, exist_ok=True)
        
        # Set up MLflow experiment (DAGsHub)
        mlflow.set_experiment('Clustering Models')

    # ----------------------------
    # Evaluate model silhouette
    # ----------------------------
    def evaluate_model(self, model, X: np.ndarray):
        labels = model.fit_predict(X)
        unique_labels = set(labels)
        if len(unique_labels) <= 1:
            return -1
        return silhouette_score(X, labels)

    # ----------------------------
    # Global trainer for a model class and param grid
    # ----------------------------
    def _global_trainer(self, model_class, params: dict, X: np.ndarray, run_name: str):
        try:
            best_model = None
            best_score = -1
            best_params = None

            with mlflow.start_run(run_name=run_name):
                for param in ParameterGrid(params):
                    model = model_class(**param)
                    score = self.evaluate_model(model, X)
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = param

                if best_model:
                    # Log metrics & params
                    mlflow.log_params(best_params)
                    mlflow.log_metric("silhouette_score", best_score)
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=run_name)
                    logging.info(f"✅ {run_name} best silhouette: {best_score}")

            return best_model, best_score

        except Exception as e:
            raise CustomException(e)

    # ----------------------------
    # Train all models & pick the best
    # ----------------------------
    def train(self, X: np.ndarray):
        try:
            results = {}

            # KMeans
            kmeans_params = {"n_clusters": [3, 5, 8, 10, 12], "init": ["k-means++"], "n_init": [10], "random_state": [42]}
            results["KMeans"] = self._global_trainer(KMeans, kmeans_params, X, "KMeans")

            # DBSCAN
            dbscan_params = {"eps": [0.3, 0.5, 0.7, 1.0], "min_samples": [3, 5, 10]}
            results["DBSCAN"] = self._global_trainer(DBSCAN, dbscan_params, X, "DBSCAN")

            # Hierarchical
            hierarchical_params = {"n_clusters": [3, 5, 8, 10], "linkage": ["ward", "complete", "average"]}
            results["Hierarchical"] = self._global_trainer(AgglomerativeClustering, hierarchical_params, X, "Hierarchical")

            # Pick the best silhouette score
            best_model_name = max(results, key=lambda k: results[k][1])
            best_model, best_score = results[best_model_name]

            # Save only the best model as pickle
            with open(self.modeltrainconfig.best_model_path, "wb") as f:
                pickle.dump(best_model, f)

            logging.info(f"🎯 Best clustering model: {best_model_name} with silhouette: {best_score}")
            logging.info(f"✅ Pickle saved at: {self.modeltrainconfig.best_model_path}")

            logging.info('Model Training Completed')

            return best_model_name, best_score, self.modeltrainconfig.best_model_path

        except Exception as e:
            raise CustomException(e)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_path = data_ingestion.Instantiate_dataingestion()

    df = pd.read_csv('artifacts/raw_data.csv')

    data_transformation = DataTransformation()
    transformed_data = data_transformation.instantiate_data_transformer(df)

    train=ModelTrainer()
    train.train(transformed_data)
