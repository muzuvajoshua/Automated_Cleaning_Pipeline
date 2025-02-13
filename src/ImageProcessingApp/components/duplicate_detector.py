from PIL import Image
import imagehash

class DuplicateDetector:
    def __init__(self, hash_size: int = 8, threshold: int = 5):
        self.hash_size = hash_size
        self.threshold = threshold
        self.hash_dict = {}

    def compute_hash(self, image: Image.Image) -> str:
        """Compute perceptual hash of image"""
        return str(imagehash.average_hash(image, self.hash_size))

    def is_duplicate(self, image: Image.Image, image_id: str) -> bool:
        """
        Check if image is a duplicate using perceptual hashing.
        Returns True if duplicate is found.
        """
        current_hash = self.compute_hash(image)

        for stored_id, stored_hash in self.hash_dict.items():
            if stored_id != image_id:
                # Compare hash difference
                hash_diff = sum(c1 != c2 for c1, c2 in
                              zip(current_hash, stored_hash))
                if hash_diff <= self.threshold:
                    return True

        self.hash_dict[image_id] = current_hash
        return False