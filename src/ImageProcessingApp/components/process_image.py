from pathlib import Path
from typing import List, Dict
from PIL import Image
from .processor import ImageProcessor
from .duplicate_detector import DuplicateDetector
from .metadata_validator import MetadataValidator



def process_image_batch(image_paths: List[str],
                       metadata_list: List[Dict]) -> Dict[str, List]:
    """
    Process a batch of images with their metadata.
    Returns dictionary with processed images and error reports.
    """
    processor = ImageProcessor()
    detector = DuplicateDetector()
    validator = MetadataValidator()

    results = {
        'processed': [],
        'failed_quality': [],
        'duplicates': [],
        'invalid_metadata': [],
        'errors': []
    }

    for image_path, metadata in zip(image_paths, metadata_list):
        try:
            # Validate metadata first
            is_valid, metadata_errors = validator.validate_metadata(metadata)
            if not is_valid:
                results['invalid_metadata'].append({
                    'path': image_path,
                    'errors': metadata_errors
                })
                continue

            # Process image
            processed_img = processor.standardize_image(image_path)
            if processed_img is None:
                results['failed_quality'].append(image_path)
                continue

            # Check for duplicates
            if detector.is_duplicate(processed_img, image_path):
                results['duplicates'].append(image_path)
                continue

            # Save processed image
            save_path = f"processed_{Path(image_path).name}"
            processed_img.save(save_path, "JPEG", quality=95)
            results['processed'].append(save_path)

        except Exception as e:
            results['errors'].append({
                'path': image_path,
                'error': str(e)
            })

    return results