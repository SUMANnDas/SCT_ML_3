import cv2
import numpy as np
from PIL import Image
import io

def preprocess_uploaded_image(image_bytes, img_size=(64, 64)):
    """Preprocess uploaded image for prediction"""
    try:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Resize image
        img_resized = cv2.resize(img_array, img_size)
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        # Flatten
        img_flattened = img_gray.flatten()
        
        return img_flattened.reshape(1, -1)
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None