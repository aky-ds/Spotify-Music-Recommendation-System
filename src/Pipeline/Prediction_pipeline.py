import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from src.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_obj


class PredictionPipeline:
    def __init__(self):
        logging.info("✅ Prediction Pipeline Started")

        self.model_path = os.path.join(
            "artifacts", "best_clustering_model.pkl"
        )
        self.numeric_preprocessor_path = os.path.join(
            "artifacts", "numeric_preprocessor.pkl"
        )

        # MUST MATCH TRAINING
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.text_model = SentenceTransformer(self.embedding_model_name)

    def predict(self, df: pd.DataFrame):
        try:
            logging.info("🚀 Starting Prediction")

            model = load_obj(self.model_path)
            numeric_preprocessor = load_obj(self.numeric_preprocessor_path)

            # -----------------------------
            # Drop unused columns
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
            # Text feature
            # -----------------------------
            df["text_features"] = (
                df["track_name"].fillna("")
                + " by "
                + df["artist_name"].fillna("")
                + " genre "
                + df["artist_genres"].fillna("")
                + " album "
                + df["album_name"].fillna("")
            )

            # -----------------------------
            # Transform
            # -----------------------------
            numeric_array = numeric_preprocessor.transform(
                df[numeric_features]
            )

            text_embeddings = self.text_model.encode(
                df["text_features"].tolist(),
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            combined = np.hstack([numeric_array, text_embeddings])

            logging.info(f"✅ Prediction feature shape: {combined.shape}")

            cluster = model.predict(combined)

            return cluster

        except Exception as e:
            logging.error(f"❌ Prediction failed: {e}")
            raise CustomException(e)
