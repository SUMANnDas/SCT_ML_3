class CatDogClassifier {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.initializeElements();
        this.attachEventListeners();
    }
    
    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.removeBtn = document.getElementById('removeBtn');
        this.predictBtn = document.getElementById('predictBtn');
        this.btnText = document.querySelector('.btn-text');
        this.btnLoader = document.querySelector('.btn-loader');
        this.resultSection = document.getElementById('resultSection');
        this.resultEmoji = document.getElementById('resultEmoji');
        this.resultText = document.getElementById('resultText');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.errorMessage = document.getElementById('errorMessage');
    }
    
    attachEventListeners() {
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });
        
        // File input change
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
        
        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.handleFileSelect(file);
            }
        });
        
        // Remove button
        this.removeBtn.addEventListener('click', () => {
            this.clearImage();
        });
        
        // Predict button
        this.predictBtn.addEventListener('click', () => {
            this.predictImage();
        });
    }
    
    handleFileSelect(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }
        
        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            this.showError('File size must be less than 5MB.');
            return;
        }
        
        this.selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.uploadArea.style.display = 'none';
            this.imagePreview.style.display = 'block';
            this.predictBtn.disabled = false;
            this.hideError();
            this.hideResult();
        };
        reader.readAsDataURL(file);
    }
    
    clearImage() {
        this.selectedFile = null;
        this.uploadArea.style.display = 'block';
        this.imagePreview.style.display = 'none';
        this.predictBtn.disabled = true;
        this.fileInput.value = '';
        this.hideResult();
        this.hideError();
    }
    
    async predictImage() {
        if (!this.selectedFile) return;
        
        // Show loading state
        this.setLoading(true);
        this.hideResult();
        this.hideError();
        
        try {
            // Create form data
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            
            // Make API request
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            this.showResult(result);
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Failed to predict image. Please make sure the backend server is running.');
        } finally {
            this.setLoading(false);
        }
    }
    
    setLoading(loading) {
        if (loading) {
            this.btnText.style.display = 'none';
            this.btnLoader.style.display = 'inline';
            this.predictBtn.disabled = true;
        } else {
            this.btnText.style.display = 'inline';
            this.btnLoader.style.display = 'none';
            this.predictBtn.disabled = false;
        }
    }
    
    showResult(result) {
        this.resultEmoji.textContent = result.emoji;
        this.resultText.textContent = `Predicted: ${result.prediction}`;
        
        // Show confidence
        const confidence = Math.min(result.confidence * 20, 100); // Scale confidence
        this.confidenceFill.style.width = `${confidence}%`;
        this.confidenceValue.textContent = `${confidence.toFixed(1)}%`;
        
        this.resultSection.style.display = 'block';
    }
    
    hideResult() {
        this.resultSection.style.display = 'none';
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.style.display = 'block';
    }
    
    hideError() {
        this.errorMessage.style.display = 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CatDogClassifier();
});