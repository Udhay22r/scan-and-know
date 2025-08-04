# Scan & Know - AI Powered Nutrition Analysis

A comprehensive web application that uses AI to analyze food products and provide personalized nutrition recommendations based on user dietary preferences and health conditions.

## Features

### 🍎 Core Features
- **Barcode Scanning**: Scan product barcodes using camera, manual input, or image upload
- **AI-Powered Analysis**: Get personalized nutrition advice using Google's Gemini AI
- **Dietary Preferences**: Set your health conditions, allergens, and dietary restrictions
- **Real-time Chatbot**: Ask questions about products and get instant AI responses

### 🔍 New OCR Analysis Features
- **Product Not Found Handling**: When products aren't in the database, upload images for manual analysis
- **Image Text Extraction**: Upload ingredients list and nutritional label images for OCR analysis
- **Manual Ingredients Input**: Type ingredients directly for quick analysis
- **Smart Sugar Analysis**: Special logic for diabetic patients analyzing first 3-5 ingredients
- **Direct Analysis Button**: Skip scanning and go straight to manual analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR (for text extraction from images)

### Install Tesseract OCR

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH
3. Or use: `winget install UB-Mannheim.TesseractOCR`

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up API key:**
```bash
python setup_api_key.py
```

5. **Verify Tesseract OCR installation:**
```bash
python setup_tesseract.py
```

6. **Run the application:**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Basic Scanning
1. Set your dietary preferences in the ⚙️ Preferences tab
2. Go to 📱 Scan Product tab
3. Choose your scanning method:
   - 📷 Camera Scan: Use your device camera
   - ⌨️ Manual Input: Type the barcode number
   - 📁 Upload Image: Upload a photo of the barcode

### Manual Analysis (New Feature)
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

## Technical Details

### OCR Technology
- **EasyOCR**: Primary OCR engine for better accuracy
- **Pytesseract**: Fallback OCR engine
- **Text Processing**: Extracts nutrition data and allergens from images

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

## Troubleshooting

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

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License. 