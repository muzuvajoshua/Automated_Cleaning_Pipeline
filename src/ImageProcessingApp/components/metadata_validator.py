from typing import List, Dict, Tuple
from datetime import datetime
import logging

class MetadataValidator:
    def __init__(self, required_fields: List[str] =
                 ['source', 'timestamp']):
        self.required_fields = required_fields

    def validate_metadata(self, metadata: Dict) -> Tuple[bool, List[str]]:
        """
        Validate metadata completeness and consistency.
        Returns (is_valid, list_of_errors)
        """
        logging.getLogger(__name__).debug(f"Validating metadata: {metadata}")
        errors = []

        # Check required fields
        for field in self.required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")

        # Validate timestamp format
        if 'timestamp' in metadata:
            try:
                datetime.fromisoformat(metadata['timestamp'])
            except ValueError:
                errors.append("Invalid timestamp format")

        # # Validate relevance score
        # if 'relevance' in metadata:
        #     relevance = metadata['relevance']
        #     if not isinstance(relevance, (int, float)) or \
        #        not (0 <= relevance <= 1):
        #         errors.append("Relevance must be float between 0 and 1")

        return len(errors) == 0, errors