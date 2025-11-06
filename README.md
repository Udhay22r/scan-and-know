# Scan & Know 🍎

**AI-Powered Nutrition Analysis with Advanced OCR Technology**

A comprehensive web application that uses cutting-edge OCR technology and AI to analyze food products and provide personalized nutrition recommendations based on your dietary preferences and health conditions.

![Screenshot 1](images/Screenshot%202025-10-01%20120127.png)

![Screenshot 2](images/Screenshot%202025-10-01%20120248.png)

![Screenshot 3](images/Screenshot%202025-10-01%20120533.png)

![Screenshot 4](images/Screenshot%202025-10-01%20121129.png)

## 🚀 Features

### 🔍 **Advanced OCR Technology**
- **EasyOCR**: Primary OCR engine for superior text extraction accuracy
- **Tesseract OCR**: Robust fallback OCR engine for comprehensive coverage
- **Smart Text Processing**: Extracts nutrition data and allergens from product images

### 📱 **Multiple Scanning Methods**
- **Camera Barcode Scan**: Real-time barcode detection using device camera
- **Manual Input**: Direct barcode number entry
- **Image Upload**: Upload photos of barcodes or product labels
- **OCR Analysis**: Extract text from ingredients and nutritional labels

### 🤖 **AI-Powered Analysis**
- **Google Gemini AI**: Personalized nutrition advice and recommendations
- **Smart Chatbot**: Interactive AI assistant for product inquiries
- **Dietary Preferences**: Custom health conditions and allergen tracking
- **Real-time Analysis**: Instant nutrition insights and recommendations

### 🍎 **Smart Nutrition Logic**
- **Sugar Analysis**: Special algorithms for diabetic patients
- **Allergen Detection**: Comprehensive allergen identification
- **Ingredient Prioritization**: First 3-5 ingredients analysis
- **Consumption Recommendations**: Personalized intake guidelines

## 🛠️ **OCR Technologies Used**

### **EasyOCR**
- **Primary OCR Engine**: Superior accuracy for text extraction
- **Multi-language Support**: Handles various text formats
- **Real-time Processing**: Fast text recognition from images
- **High Precision**: Advanced neural network-based recognition

### **Tesseract OCR**
- **Fallback Engine**: Reliable backup for text extraction
- **Open Source**: Industry-standard OCR solution
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Robust Performance**: Handles challenging image conditions

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR (for text extraction from images)

### Install Tesseract OCR

**Windows:**
```bash
# Option 1: Using winget
winget install UB-Mannheim.TesseractOCR

# Option 2: Manual download
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Install and add to PATH
```

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

### Python Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd scan-and-know
```

2. **Create virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up API key:**
```bash
python setup_api_key.py
```

5. **Verify OCR installation:**
```bash
python setup_tesseract.py
```

6. **Run the application:**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 🎯 Usage

### Basic Scanning
1. Set your dietary preferences in the ⚙️ Preferences tab
2. Go to 📱 Scan Product tab
3. Choose your scanning method:
   - 📷 Camera Scan: Use your device camera
   - ⌨️ Manual Input: Type the barcode number
   - 📁 Upload Image: Upload a photo of the barcode

### OCR Analysis (Advanced Feature)
When a product is not found in the database:

1. **Upload Images Method:**
   - Upload a clear photo of the ingredients list
   - Upload a clear photo of the nutritional label
   - Click "Analyze Images" for OCR-based analysis

2. **Manual Input Method:**
   - Type the ingredients list directly
   - Click "Analyze Ingredients" for quick analysis

3. **Direct Analysis:**
   - Click "Analyze Using Ingredients & Nutritional Label" button
   - Skip scanning and go straight to manual analysis

### AI Chatbot
- After scanning or analyzing a product, use the AI assistant on the right
- Ask questions about nutrition, ingredients, health implications
- Get personalized advice based on your dietary preferences

## 🧠 Neural Network Prediction

The application includes a feed-forward neural network built with **PyTorch** that predicts consumption recommendations (can_consume, consume_half, or avoid) based on nutritional data and user preferences.

**To train the model:**
```bash
pip install torch scikit-learn pandas
python train_model.py
```

The model will be saved to `models/consumption_model.pth` and automatically used by the application. If the model is unavailable, the system falls back to rule-based analysis.

## 🔬 Technical Details

### OCR Processing Pipeline
1. **Image Preprocessing**: Optimize image quality for text extraction
2. **EasyOCR Processing**: Primary text recognition with high accuracy
3. **Tesseract Fallback**: Secondary processing for challenging images
4. **Text Post-processing**: Clean and structure extracted data
5. **Nutrition Analysis**: Parse nutritional information and allergens

### Sugar Analysis Logic
For diabetic patients:
- **First 3 ingredients contain sugar**: ❌ AVOID
- **First 5 ingredients contain sugar**: ⚠️ CONSUME IN MODERATION
- **No sugar in first 5**: ✅ SAFE TO CONSUME

For non-diabetic patients:
- **First 3 ingredients contain sugar**: ⚠️ CONSIDER IN MODERATION
- **First 5 ingredients contain sugar**: ✅ SAFE TO CONSUME
- **No sugar in first 5**: ✅ SAFE TO CONSUME

### API Integration
- **OpenFoodFacts**: Product database
- **Google Gemini AI**: Nutrition analysis and chatbot
- **Rate Limiting**: Built-in protection against API limits

## 🔧 Troubleshooting

### OCR Issues
- Ensure Tesseract is properly installed and in PATH
- If Tesseract is installed but not detected, add it to your system PATH:
  - Windows: Add `C:\Program Files\Tesseract-OCR` to PATH
  - macOS/Linux: Usually auto-added during installation
- Use clear, well-lit images for better text extraction
- Try different angles if text extraction fails
- Run `python setup_tesseract.py` to verify installation

### API Issues
- Check your Gemini API key in the .env file
- Monitor rate limits (10 requests per minute)
- Fallback responses are provided when API is unavailable

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
