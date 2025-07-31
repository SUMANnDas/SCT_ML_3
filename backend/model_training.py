import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class CatDogSVMTrainer:
    def __init__(self, data_path, img_size=(64, 64)):
        self.data_path = data_path
        self.img_size = img_size
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self):
        """Load and preprocess images from the dataset"""
        X, y = [], []
        
        # Load cat images (label 0)
        cats_path = os.path.join(self.data_path, 'train', 'cats')
        if os.path.exists(cats_path):
            for filename in os.listdir(cats_path)[:12000]:  # Limit for faster training
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cats_path, filename)
                    img = self.preprocess_image(img_path)
                    if img is not None:
                        X.append(img)
                        y.append(0)  # Cat
        
        # Load dog images (label 1)
        dogs_path = os.path.join(self.data_path, 'train', 'dogs')
        if os.path.exists(dogs_path):
            for filename in os.listdir(dogs_path)[:12000]:  # Limit for faster training
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dogs_path, filename)
                    img = self.preprocess_image(img_path)
                    if img is not None:
                        X.append(img)
                        y.append(1)  # Dog
        
        return np.array(X), np.array(y)
    
    def preprocess_image(self, img_path):
        """Preprocess individual image"""
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                return None
                
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Convert to grayscale for SVM (reduces dimensionality)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Flatten the image
            img_flattened = img.flatten()
            
            return img_flattened
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def train_model(self):
        """Train the SVM model"""
        print("Loading data...")
        X, y = self.load_data()
        
        if len(X) == 0:
            print("No data loaded! Please check your dataset path.")
            return
        
        print(f"Loaded {len(X)} images")
        print(f"Cats: {np.sum(y == 0)}, Dogs: {np.sum(y == 1)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train SVM model
        print("Training SVM model...")
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Classification Report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
        
        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/svm_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        print("Model and scaler saved!")
        
        return accuracy, cm

def main():
    # Initialize trainer
    trainer = CatDogSVMTrainer(data_path=r'C:\Users\Suman\Desktop\CAT VS DOGS IMAGE CLASSIFIER USING SVM\data_path')
    
    # Train model
    accuracy, cm = trainer.train_model()
    
    if accuracy:
        print(f"Training completed! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()