from setuptools import setup, find_packages

setup(
    name="image_processing_pipeline",
    version="0.1.0",
    package_dir={"": "src"},  # Use src-based layout
    packages=find_packages(where="src"),
    install_requires=[
        "flask",
        "flask-cors",
        "pillow",
        "werkzeug",
        "opencv-python-headless",
        "numpy",
        "imagehash",
    ],
    python_requires=">=3.7",
)
