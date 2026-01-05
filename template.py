
import os

from pathlib import Path

list_of_files = [
    '.github/workflows/.gitkeep',
    'src/__init__.py',
    "src/components/__init__.py",
    "src/components/Data_ingestion.py",
    "src/components/Data_Transformation.py",
    "src/components/Model_trainer.py",
    "src/components/Model_evaluation.py",
    "src/Pipeline/__init__.py",
    "src/Pipeline/training_pipeline.py",
    "src/Pipeline/testing_pipeline.py",
    "src/logger/__init__.py",
    "src/logger/logger.py",
    "src/exceptions/__init__.py",
    "src/exception.py",
    "src/utils/__init__.py",
    "src/utils/utils.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "tox.ini",
    'requirements.txt',
    "requirements_dev.txt",
    "setup.py",
    "init_setup.sh",
    "setup.cfg",
    "pyproject.toml",
    "app.py",
    "docker",
    'templates/index.html',
    "experiments/experiments.ipynb"
]


for file in list_of_files:
    file_path=Path(file)
    file_dir,file_name=os.path.split(file_path)
    if file_dir !="":
        os.makedirs(file_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        with open(file_path,'w') as f:
            pass
        