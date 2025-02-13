from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import base64
import torch
import torchvision.models as models
from ImageProcessingApp.components.processor import ImageProcessor
from ImageProcessingApp.components.duplicate_detector import DuplicateDetector
from ImageProcessingApp.components.metadata_validator import MetadataValidator
from ImageProcessingApp.components.process_image import process_image_batch
from ImageProcessingApp.components.pipeline import ProcessingPipeline  # ✅ Import the pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['PROCESSED_FOLDER'] = 'processed'

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# # Load a pre-trained ResNet model
# teacher_model = models.resnet18(pretrained=True)

# # Modify final layer for binary classification
# teacher_model.fc = torch.nn.Linear(512, 1)
# torch.save(teacher_model.state_dict(), "../models/teacher_model.pth")
# print("Pre-trained teacher model downloaded and saved!")


# Initialize processors
processor = ImageProcessor()
detector = DuplicateDetector()
validator = MetadataValidator()
# pipeline = ProcessingPipeline("./config.json")  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pipeline')
def pipeline_page():
    """New page for teacher-student pipeline."""
    return render_template('pipeline.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    metadata = {
        'source': request.form.get('source', ''),
        'relevance': float(request.form.get('relevance', 0.8)),
        'timestamp': datetime.now().isoformat()
    }
    
    uploaded_files = []
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
    
    # Process images
    results = process_image_batch(uploaded_files, [metadata] * len(uploaded_files))
    
    # Convert processed images to base64 for preview
    processed_previews = {}
    for path in results['processed']:
        with open(path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            processed_previews[path] = img_data
    
    results['previews'] = processed_previews
    return jsonify(results)

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    """API endpoint to process images with teacher-student learning."""
    data = request.get_json()
    input_dir = data.get("input_dir")
    output_dir = data.get("output_dir")

    if not input_dir or not output_dir:
        return jsonify({"error": "Missing input_dir or output_dir"}), 400

    result = pipeline.run_batch_processing(input_dir, output_dir)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
