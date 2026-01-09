import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sentence_transformers import SentenceTransformer

from src.components.Data_ingestion import DataIngestion
from src.logger.logger import logging
from src.exception import CustomException
from src.utils.utils import save_obj


# =====================================================
# Config
# =====================================================
@dataclass
class DataTransformationConfig:
    numeric_preprocessor_path: str = os.path.join(
        "artifacts", "numeric_preprocessor.pkl"
    )


# =====================================================
# Data Transformation
# =====================================================
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        logging.info("✅ Data Transformation Initialized")

    # -------------------------------------------------
    # Numeric Preprocessor (PICKLABLE)
    # -------------------------------------------------
    def get_numeric_preprocessor(self):
        try:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            return numeric_pipeline

        except Exception as e:
            raise CustomException(e)

    # -------------------------------------------------
    # Main Transformation
    # -------------------------------------------------
    def instantiate_data_transformer(self, data: pd.DataFrame):
        try:
            logging.info("🚀 Starting Data Transformation")

            df = data.copy()

            # -----------------------------
            # Drop unnecessary columns
            # -----------------------------
            drop_cols = ["_id", "track_id", "explicit"]
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

            # -----------------------------
            # Numeric cleanup
            # -----------------------------
            df["track_duration_min"] = (
                df["track_duration_min"]
                .astype(str)
                .str.replace(r"[^0-9.]", "", regex=True)
                .replace("", np.nan)
                .astype(float)
            )

            numeric_features = [
                "track_popularity",
                "artist_popularity",
                "artist_followers",
                "track_duration_min",
                "album_total_tracks"
            ]

            for col in numeric_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # -----------------------------
            # Text Feature Engineering
            # -----------------------------
            df["text_features"] = (
                df["track_name"].fillna("").astype(str)
                + " by "
                + df["artist_name"].fillna("").astype(str)
                + " genre "
                + df["artist_genres"].fillna("").astype(str)
                + " album "
                + df["album_name"].fillna("").astype(str)
            )

            # -----------------------------
            # Numeric Transformation
            # -----------------------------
            numeric_preprocessor = self.get_numeric_preprocessor()
            numeric_array = numeric_preprocessor.fit_transform(
                df[numeric_features]
            )

            # -----------------------------
            # Save ONLY numeric preprocessor
            # -----------------------------
            save_obj(
                file_path=self.config.numeric_preprocessor_path,
                obj=numeric_preprocessor
            )

            logging.info("✅ Numeric preprocessor saved successfully")

            # -----------------------------
            # Text Embeddings (NOT pickled)
            # -----------------------------
            text_model = SentenceTransformer("all-MiniLM-L6-v2")
            text_embeddings = text_model.encode(
                df["text_features"].astype(str).tolist(),
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # -----------------------------
            # Combine numeric + text
            # -----------------------------
            final_array = np.hstack([numeric_array, text_embeddings])

            logging.info(
                f"✅ Final transformed shape: {final_array.shape}"
            )

            return final_array

        except Exception as e:
            raise CustomException(e)


# =====================================================
# Main Execution
# =====================================================
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_path = data_ingestion.Instantiate_dataingestion()

    df = pd.read_csv('artifacts/raw_data.csv')

    data_transformation = DataTransformation()
    transformed_data = data_transformation.instantiate_data_transformer(df)

    print("✅ Transformed data shape:", transformed_data.shape)
