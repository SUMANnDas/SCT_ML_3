from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from utils import preprocess_uploaded_image
import os

app = FastAPI(title="Cat vs Dog Classifier", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    """Load the trained model and scaler on startup"""
    global model, scaler
    
    try:
        model_path = "models/svm_model.joblib"
        scaler_path = "models/scaler.joblib"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully!")
        else:
            print("Model files not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Cat vs Dog Classifier API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and scaler is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict if uploaded image is a cat or dog"""
    global model, scaler
    
    # Check if model is loaded
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_uploaded_image(image_bytes)
        
        if processed_image is None:
            raise HTTPException(status_code=400, detail="Error processing image")
        
        # Scale the features
        scaled_features = scaler.transform(processed_image)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.decision_function(scaled_features)[0]
        
        # Convert prediction to label
        label = "Cat" if prediction == 0 else "Dog"
        emoji = "🐱" if prediction == 0 else "🐶"
        
        # Calculate confidence (convert decision function to probability-like score)
        confidence = abs(probability)
        
        return JSONResponse({
            "prediction": label,
            "emoji": emoji,
            "confidence": float(confidence),
            "raw_prediction": int(prediction)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)