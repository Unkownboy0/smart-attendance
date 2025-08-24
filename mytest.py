# mytest.py
import cv2
import numpy as np

def test(image, model_dir, device_id):
    """
    Anti-spoofing placeholder.
    - image: frame from webcam
    - model_dir: directory where anti-spoofing models are located
    - device_id: webcam or processing device ID (not used in this dummy version)

    Returns:
        1 if face is considered real (pass)
        0 if face is considered fake (fail)
    """
    if image is None:
        return 0  # no image -> fail

    # Convert to grayscale to check image quality
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Simple heuristic: if variance is very low, treat as fake
    if np.var(gray) < 100:
        return 0

    # If all looks normal, treat as real
    return 1
