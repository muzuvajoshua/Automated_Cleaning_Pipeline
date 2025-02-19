import os
import logging
from flask import Flask, jsonify, send_file
from datetime import datetime
from pathlib import Path
from PIL import Image
from apscheduler.schedulers.background import BackgroundScheduler
from typing import List, Dict
import time
from ImageProcessingApp.components.process_image import ImageBatchProcessor

# Initialize Flask app
app = Flask(__name__)

# Configuration
class Config:
    STORAGE_PATH = '/mnt/st1/Scrapped_images'
    PROCESSED_PATH = '/mnt/st1/ProcessedImages'
    BATCH_SIZE = 50
    SCHEDULER_INTERVAL = 60  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    EMPTY_CHECKS_BEFORE_SHUTDOWN = 3  # Number of empty checks before shutdown

# Counter for tracking empty checks
empty_checks_counter = 0

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('image_processor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(Config.STORAGE_PATH, exist_ok=True)
os.makedirs(Config.PROCESSED_PATH, exist_ok=True)

# Initialize the ImageBatchProcessor
processor = ImageBatchProcessor(Config.PROCESSED_PATH)

def get_image_extensions() -> set:
    """Get set of supported image extensions."""
    return set(ext.lower() for ext in Image.registered_extensions().keys())

def get_unprocessed_images() -> List[str]:
    """
    Recursively scan STORAGE_PATH for unprocessed images with improved efficiency.
    """
    image_extensions = get_image_extensions()
    unprocessed = []

    for root, _, files in os.walk(Config.STORAGE_PATH):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                abs_path = os.path.abspath(full_path)
                
                # Check if image should be skipped
                should_skip, _ = processor.should_skip_image(abs_path)
                if not should_skip:
                    rel_path = os.path.relpath(full_path, Config.STORAGE_PATH)
                    unprocessed.append(rel_path)

    return sorted(unprocessed)

def create_metadata(image_paths: List[str], relative_paths: List[str]) -> List[Dict]:
    """Create metadata for a batch of images."""
    return [{
        'source': 'mounted_storage',
        'timestamp': datetime.now().isoformat(),
        'batch_processed': True,
        'original_filename': Path(path).name,
        'relative_path': rel_path
    } for path, rel_path in zip(image_paths, relative_paths)]

def process_new_images() -> None:
    """
    Process new images and shut down if none are found for several consecutive checks.
    """
    global empty_checks_counter
    
    try:
        unprocessed_images = get_unprocessed_images()
        
        if not unprocessed_images:
            empty_checks_counter += 1
            logger.info(f"No new unprocessed images found. Empty check count: {empty_checks_counter}")
            
            if empty_checks_counter >= Config.EMPTY_CHECKS_BEFORE_SHUTDOWN:
                logger.info("No new images found after multiple checks. Shutting down application...")
                scheduler.shutdown()
                os._exit(0)  # Force exit the application
            return
        
        # Reset counter if images are found
        empty_checks_counter = 0
        logger.info(f"Found {len(unprocessed_images)} unprocessed images.")
        
        # Process in batches
        for i in range(0, len(unprocessed_images), Config.BATCH_SIZE):
            batch = unprocessed_images[i:i + Config.BATCH_SIZE]
            image_paths = [os.path.join(Config.STORAGE_PATH, img) for img in batch]
            metadata_list = create_metadata(image_paths, batch)

            # Attempt processing with retries
            for attempt in range(Config.MAX_RETRIES):
                try:
                    results = processor.process_image_batch(
                        image_paths=image_paths,
                        metadata_list=metadata_list
                    )
                    logger.info(f"Processed batch {i//Config.BATCH_SIZE + 1}: {results}")
                    break
                except Exception as e:
                    if attempt < Config.MAX_RETRIES - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        time.sleep(Config.RETRY_DELAY)
                    else:
                        logger.error(f"Failed to process batch after {Config.MAX_RETRIES} attempts: {str(e)}")
                        raise

    except Exception as e:
        logger.error(f"Error in process_new_images: {str(e)}")

# API Endpoints
@app.route('/status')
def status():
    """Get application status."""
    try:
        return jsonify({
            'status': 'running',
            'empty_checks': empty_checks_counter,
            'checks_before_shutdown': Config.EMPTY_CHECKS_BEFORE_SHUTDOWN - empty_checks_counter,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/list_images')
def list_images():
    """List unprocessed images with error handling."""
    try:
        unprocessed = get_unprocessed_images()
        return jsonify({
            'status': 'success',
            'images': unprocessed,
            'total': len(unprocessed)
        })
    except Exception as e:
        logger.error(f"Error in list_images: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/processing_status')
def processing_status():
    """Get detailed processing statistics."""
    try:
        unprocessed = get_unprocessed_images()
        stats = processor.get_processing_stats()
        return jsonify({
            'status': 'success',
            'unprocessed_count': len(unprocessed),
            'empty_checks': empty_checks_counter,
            'stats': stats,
            'processing_rate': f"{Config.BATCH_SIZE} images every {Config.SCHEDULER_INTERVAL} seconds"
        })
    except Exception as e:
        logger.error(f"Error in processing_status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    try:
        # Initialize and start scheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=process_new_images,
            trigger="interval",
            seconds=Config.SCHEDULER_INTERVAL,
            max_instances=1  # Prevent overlapping jobs
        )
        scheduler.start()
        logger.info(f"Scheduler started. Processing up to {Config.BATCH_SIZE} images every {Config.SCHEDULER_INTERVAL} seconds.")
        logger.info(f"Application will shut down after {Config.EMPTY_CHECKS_BEFORE_SHUTDOWN} consecutive empty checks.")
        
        # Run the Flask application
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise