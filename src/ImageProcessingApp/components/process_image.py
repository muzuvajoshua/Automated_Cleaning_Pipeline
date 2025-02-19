from pathlib import Path
from typing import List, Dict, Set, Tuple
import os
import logging
import json
from datetime import datetime
from PIL import Image
from .processor import ImageProcessor
from .duplicate_detector import DuplicateDetector
from .metadata_validator import MetadataValidator

logger = logging.getLogger(__name__)

class ImageBatchProcessor:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.processor = ImageProcessor()
        self.detector = DuplicateDetector()
        self.validator = MetadataValidator()
        
        # Separate history files for different categories
        self.history_dir = os.path.join(output_dir, '.processing_history')
        os.makedirs(self.history_dir, exist_ok=True)
        
        self.processed_file = os.path.join(self.history_dir, 'processed_images.json')
        self.failed_quality_file = os.path.join(self.history_dir, 'failed_quality.json')
        self.duplicates_file = os.path.join(self.history_dir, 'duplicates.json')
        self.invalid_metadata_file = os.path.join(self.history_dir, 'invalid_metadata.json')
        
        # Load all history sets
        self.processed_images = self._load_history(self.processed_file)
        self.failed_quality_images = self._load_history(self.failed_quality_file)
        self.duplicate_images = self._load_history(self.duplicates_file)
        self.invalid_metadata_images = self._load_history(self.invalid_metadata_file)

    def _load_history(self, file_path: str) -> Set[str]:
        """Load a set of image paths from a history file."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    history = json.load(f)
                return set(history.get('images', []))
            except Exception as e:
                logger.error(f"Error loading history from {file_path}: {str(e)}")
                return set()
        return set()

    def _save_history(self, file_path: str, image_set: Set[str]):
        """Save a set of image paths to a history file."""
        try:
            history = {
                'images': list(image_set),
                'last_updated': datetime.now().isoformat()
            }
            with open(file_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history to {file_path}: {str(e)}")

    def should_skip_image(self, abs_path: str) -> Tuple[bool, str]:
        """
        Check if an image should be skipped and return the reason.
        Returns (should_skip, reason)
        """
        if abs_path in self.processed_images:
            return True, "already_processed"
        if abs_path in self.failed_quality_images:
            return True, "failed_quality"
        if abs_path in self.duplicate_images:
            return True, "duplicate"
        if abs_path in self.invalid_metadata_images:
            return True, "invalid_metadata"
        return False, ""

    def process_image_batch(self, image_paths: List[str],
                          metadata_list: List[Dict]) -> Dict[str, List]:
        """
        Process a batch of images, skipping any that have been processed or failed before.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        results = {
            'processed': [],
            'skipped': [],
            'failed_quality': [],
            'duplicates': [],
            'invalid_metadata': [],
            'errors': []
        }

        logger.debug(f"Starting process_image_batch for {len(image_paths)} images.")

        for image_path, metadata in zip(image_paths, metadata_list):
            abs_path = os.path.abspath(image_path)
            
            # Check if image should be skipped
            should_skip, skip_reason = self.should_skip_image(abs_path)
            if should_skip:
                results['skipped'].append({
                    'path': image_path,
                    'reason': skip_reason
                })
                continue

            logger.debug(f"Processing image: {image_path}")
            try:
                # Validate metadata
                is_valid, metadata_errors = self.validator.validate_metadata(metadata)
                if not is_valid:
                    logger.debug(f"Invalid metadata for {image_path}: {metadata_errors}")
                    self.invalid_metadata_images.add(abs_path)
                    self._save_history(self.invalid_metadata_file, self.invalid_metadata_images)
                    results['invalid_metadata'].append({
                        'path': image_path,
                        'errors': metadata_errors
                    })
                    continue

                # Process image
                processed_img = self.processor.standardize_image(image_path)
                if processed_img is None:
                    logger.debug(f"Image {image_path} failed quality check.")
                    self.failed_quality_images.add(abs_path)
                    self._save_history(self.failed_quality_file, self.failed_quality_images)
                    results['failed_quality'].append(image_path)
                    continue

                # Check for duplicates
                if self.detector.is_duplicate(processed_img, image_path):
                    logger.debug(f"Image {image_path} detected as duplicate.")
                    self.duplicate_images.add(abs_path)
                    self._save_history(self.duplicates_file, self.duplicate_images)
                    results['duplicates'].append(image_path)
                    continue

                # Save the processed image
                original_relative_path = metadata['relative_path']
                original_dir = os.path.dirname(original_relative_path)
                filename = f"processed_{os.path.basename(original_relative_path)}"
                save_dir = os.path.join(self.output_dir, original_dir)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, filename)
                processed_img.save(save_path)
                
                # Update processing history
                self.processed_images.add(abs_path)
                self._save_history(self.processed_file, self.processed_images)
                
                logger.debug(f"Saved processed image to {save_path}")
                results['processed'].append(save_path)

            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                results['errors'].append({
                    'path': image_path,
                    'error': str(e)
                })

        return results

    def get_processing_stats(self) -> Dict[str, int]:
        """Get statistics about processed and skipped images."""
        return {
            'total_processed': len(self.processed_images),
            'total_failed_quality': len(self.failed_quality_images),
            'total_duplicates': len(self.duplicate_images),
            'total_invalid_metadata': len(self.invalid_metadata_images)
        }