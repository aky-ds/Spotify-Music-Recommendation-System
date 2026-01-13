import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sentence_transformers import SentenceTransformer

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
    embedding_model_name: str = "all-MiniLM-L6-v2"


# =====================================================
# Data Transformation
# =====================================================
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        logging.info("✅ Data Transformation Initialized")

    # -------------------------------------------------
    # Numeric Preprocessor
    # -------------------------------------------------
    def get_numeric_preprocessor(self):
        try:
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
        except Exception as e:
            raise CustomException(e)

    # -------------------------------------------------
    # Main Transformation
    # -------------------------------------------------
    def instantiate_data_transformer(self, data: pd.DataFrame):
        try:
            logging.info(" Starting Data Transformation")

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
                "album_total_tracks",
            ]

            df[numeric_features] = df[numeric_features].apply(
                pd.to_numeric, errors="coerce"
            )

            # -----------------------------
            #  FIXED Text Feature Engineering
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
            # Numeric preprocessing
            # -----------------------------
            numeric_preprocessor = self.get_numeric_preprocessor()
            numeric_array = numeric_preprocessor.fit_transform(
                df[numeric_features]
            )

            save_obj(
                self.config.numeric_preprocessor_path,
                numeric_preprocessor
            )

            logging.info(" Numeric preprocessor saved")

            # -----------------------------
            # Text embeddings
            # -----------------------------
            text_model = SentenceTransformer(
                self.config.embedding_model_name
            )

            text_embeddings = text_model.encode(
                df["text_features"].tolist(),
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # -----------------------------
            # Combine numeric + text
            # -----------------------------
            final_array = np.hstack([numeric_array, text_embeddings])

            logging.info(f"Final training shape: {final_array.shape}")

            return final_array

        except Exception as e:
            raise CustomException(e)
