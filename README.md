# 🐱🐶 Cat vs Dog Image Classifier using SVM

A complete machine learning project that classifies images of cats and dogs using Support Vector Machine (SVM) with a modern web interface.

## 🌟 Features

- **Machine Learning**: SVM classifier with RBF kernel for image classification
- **Web Interface**: Clean, modern HTML/CSS/JavaScript frontend
- **REST API**: FastAPI backend with image upload and prediction endpoints
- **Real-time Predictions**: Upload images and get instant cat/dog predictions
- **Confidence Scores**: Visual confidence indicators for predictions
- **Drag & Drop**: Intuitive file upload with drag-and-drop support
- **Responsive Design**: Works on desktop and mobile devices

## 🎯 Demo

Upload any cat or dog image and get instant predictions with confidence scores:

- 🐱 **Cat Detection**: "Predicted: Cat 🐱 - Confidence: 85.2%"
- 🐶 **Dog Detection**: "Predicted: Dog 🐶 - Confidence: 78.9%"

## 📁 Project Structure

```
cats-dogs-svm/
├── 📂 backend/
│   ├── 🐍 main.py                 # FastAPI application
│   ├── 🧠 model_training.py       # SVM training script
│   ├── 🔧 utils.py               # Helper functions
│   ├── 📋 requirements.txt       # Python dependencies
│   └── 📂 models/               # Saved models directory
│       ├── svm_model.joblib
│       └── scaler.joblib
├── 📂 frontend/
│   ├── 🌐 index.html            # Main HTML file
│   ├── 🎨 style.css            # Styling
│   └── ⚡ script.js            # JavaScript functionality
├── 📂 data/                    # Dataset directory
│   └── 📂 train/
│       ├── 📂 cats/            # Cat images
│       └── 📂 dogs/            # Dog images
└── 📖 README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [Kaggle Cats vs Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data) (or any cat/dog images)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd cats-dogs-svm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your images in this structure:
```
data/
└── train/
    ├── cats/     # Place cat images here (.jpg, .png, .jpeg)
    └── dogs/     # Place dog images here (.jpg, .png, .jpeg)
```

### 3. Train the Model

```bash
# From the backend directory
python model_training.py
```

Expected output:
```
Loading data...
Loaded 4000 images
Cats: 2000, Dogs: 2000
Training SVM model...
Accuracy: 0.7250
Model and scaler saved!
```

### 4. Start the Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the Frontend

**Option A: Direct File**
- Navigate to `frontend/` folder
- Open `index.html` in your browser

**Option B: HTTP Server**
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

### 6. Test the Application

1. Open the web interface
2. Upload a cat or dog image
3. Click "Predict"
4. See the results with confidence score!

## 🔧 API Documentation

### Endpoints

#### `GET /`
- **Description**: Root endpoint
- **Response**: `{"message": "Cat vs Dog Classifier API", "status": "running"}`

#### `GET /health`
- **Description**: Health check endpoint
- **Response**: `{"status": "healthy", "model_loaded": true}`

#### `POST /predict`
- **Description**: Predict if uploaded image is a cat or dog
- **Parameters**: 
  - `file`: Image file (JPG, PNG, JPEG)
- **Response**:
  ```json
  {
    "prediction": "Cat",
    "emoji": "🐱",
    "confidence": 85.2,
    "raw_prediction": 0
  }
  ```

### Example cURL Request

```bash
curl -X POST \
  -F "file=@cat_image.jpg" \
  http://localhost:8000/predict
```

## 🧠 Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**:
   - Images resized to 64x64 pixels
   - Converted to grayscale
   - Pixel values flattened and normalized

2. **Model Training**:
   - Support Vector Machine with RBF kernel
   - StandardScaler for feature normalization
   - 80/20 train-test split
   - Model persistence with joblib

3. **Performance**:
   - Typical accuracy: 70-75%
   - Training time: 5-15 minutes
   - Prediction time: <1 second

### Tech Stack

**Backend**:
- FastAPI for REST API
- scikit-learn for SVM
- OpenCV for image processing
- joblib for model persistence

**Frontend**:
- Vanilla HTML/CSS/JavaScript
- Modern CSS with gradients and animations
- Fetch API for backend communication
- Drag-and-drop file upload

## 📊 Model Performance

### Current Results
- **Accuracy**: ~72.5%
- **Precision**: Cat (66%), Dog (86%)
- **Recall**: Cat (91%), Dog (54%)



**Made with ❤️ for machine learning education and practical AI applications**

⭐ **Star this repo if you found it helpful!** ⭐
