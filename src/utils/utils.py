import os
import sys
import dill
from sentence_transformers import SentenceTransformer
from src.logger.logging import logging
from src.exceptions.exception import CustomException


# -----------------------------
# Save Object Utility
# -----------------------------

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


# -----------------------------
# Load Object Utility
# -----------------------------

def load_obj(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)


