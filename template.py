import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "ImageProcessingApp"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/processor.py",
    f"src/{project_name}/components/duplicate_detector.py",
    f"src/{project_name}/components/metadata_validator.py",
    "app.py",
    "main.py",
    "templates/index.html",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "tests/__init__.py",
    "tests/test_processor.py",
    "tests/test_pipeline.py",

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")