from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import requests
import os
from datetime import datetime
import cv2
import numpy as np
from pyzbar import pyzbar
import google.generativeai as genai
from dotenv import load_dotenv
import pytesseract
import easyocr
from PIL import Image
import re
from nn_model import ConsumptionPredictor

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize extensions
CORS(app)

# Fix PIL.Image.ANTIALIAS compatibility issue
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Initialize EasyOCR reader for better text extraction
try:
    reader = easyocr.Reader(['en'])
    print("✅ EasyOCR initialized successfully")
except Exception as e:
    print(f"⚠️ EasyOCR initialization failed: {e}")
    reader = None

# Check Tesseract installation
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    print("✅ Tesseract OCR is available")
    TESSERACT_AVAILABLE = True
except ImportError:
    print("⚠️ pytesseract module not available")
    print("📝 Please install pytesseract: pip install pytesseract")
    TESSERACT_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Tesseract OCR not available: {e}")
    print("📝 Please install Tesseract OCR for better text extraction")
    print("   Windows: winget install UB-Mannheim.TesseractOCR")
    print("   macOS: brew install tesseract")
    print("   Ubuntu: sudo apt install tesseract-ocr")
    TESSERACT_AVAILABLE = False

# Initialize Gemini AI
def check_available_models():
    """Check which Gemini models are available"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("⚠️  GEMINI_API_KEY not found - skipping model check")
            return []
        
        genai.configure(api_key=api_key)
        models = genai.list_models()
        return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    except Exception as e:
        print(f"Error checking models: {e}")
        return []

def initialize_gemini():
    """Initialize Gemini AI with the first available model"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("⚠️  GEMINI_API_KEY not found in .env file")
            return None
        
        genai.configure(api_key=api_key)
        available_models = check_available_models()
        
        if not available_models:
            print("❌ No available models found")
            return None
        
        # Try to use gemini-1.5-flash first (lower rate limits), then fallback to others
        suitable_models = ['models/gemini-1.5-flash', 'models/gemini-pro', 'models/gemini-1.5-pro']
        
        for model_name in suitable_models:
            if model_name in available_models:
                print(f"✅ Using model: {model_name}")
                return genai.GenerativeModel(model_name)
        
        # If no suitable model found, use the first available one
        if available_models:
            print(f"✅ Using fallback model: {available_models[0]}")
            return genai.GenerativeModel(available_models[0])
        
        return None
    except Exception as e:
        print(f"❌ Error initializing Gemini: {e}")
        return None

# Initialize the model
model = None
try:
    model = initialize_gemini()
    if model:
        print("✅ Gemini AI model initialized successfully")
    else:
        print("⚠️  Gemini AI model not initialized - API key may be missing")
except Exception as e:
    print(f"⚠️  Error during Gemini initialization: {e}")
    model = None

# Global variables
user_preferences = {
    "diabetic": False,  # Default to non-diabetic
    "allergens": [],
    "age": 25,
    "weight": 70,
    "exercise_intensity": "medium",  # low, medium, high
    "dietary_restrictions": []
}

# Store current product data for chatbot
current_product_data = None

# Store extracted text data for manual analysis
extracted_text_data = None

# Initialize Neural Network Model
nn_predictor = None
try:
    nn_predictor = ConsumptionPredictor()
    if os.path.exists('models/consumption_model.pth'):
        nn_predictor.load_model()
        print("✅ Neural Network model loaded successfully")
    else:
        print("⚠️ Neural Network model not found. Run train_model.py to train the model.")
        print("   Falling back to rule-based analysis.")
except Exception as e:
    print(f"⚠️ Error initializing Neural Network: {e}")
    print("   Falling back to rule-based analysis.")
    nn_predictor = None

# Rate limiting variables
last_request_time = 0
request_count = 0
RATE_LIMIT_WINDOW = 60  # 1 minute window
MAX_REQUESTS_PER_WINDOW = 10  # Max requests per minute

# Simple fallback responses when API is rate limited
FALLBACK_RESPONSES = {
    "general": "**Rate Limit Notice**\n\nI'm experiencing high demand. Please try again in a few minutes.\n\n**Quick Tips:**\n• Check nutrition facts above\n• Review dietary conflicts\n• Wait 1-2 minutes before retrying",
    "diabetic": "**Diabetes Tips**\n\n**Key Guidelines:**\n• Low sugar content (<5g per 100g)\n• Avoid artificial sweeteners\n• Check glycemic index\n• Monitor carbs",
    "allergens": "**Allergen Alert**\n\n**Important:**\n• Check ingredients list\n• Look for allergen warnings\n• When in doubt, avoid",
    "nutrition": "**Quick Nutrition Check**\n\n**Key Nutrients:**\n• **Sugar** - aim for low amounts\n• **Fat** - check saturated vs unsaturated\n• **Protein** - for satiety\n• **Fiber** - for digestion"
}

# Dietary restrictions and their associated ingredients
DIETARY_RESTRICTIONS = {
    "diabetic": {
        "avoid": [
            "sugar", "glucose", "fructose", "sucrose", "high fructose corn syrup", "hfcs",
            "maltose", "dextrose", "aspartame", "acesulfame potassium", "acesulfame k",
            "sucralose", "saccharin", "neotame", "advantame", "cyclamate", "sorbitol",
            "xylitol", "mannitol", "erythritol", "maltitol", "isomalt", "corn syrup",
            "glucose syrup", "lactose", "maltodextrin"
        ],
        "limit": ["carbohydrates", "starch"]
    },
    "vegetarian": {
        "avoid": ["beef", "pork", "chicken", "fish", "meat", "gelatin", "rennet"]
    },
    "vegan": {
        "avoid": ["milk", "cheese", "butter", "cream", "eggs", "honey", "gelatin", "whey", "casein"]
    },
    "gluten_free": {
        "avoid": ["wheat", "gluten", "barley", "rye", "malt", "flour"]
    },
    "lactose_intolerant": {
        "avoid": ["milk", "lactose", "whey", "casein", "cream", "butter"]
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Simple health check endpoint for deployment verification"""
    return jsonify({
        "status": "healthy",
        "message": "Scan & Know application is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/api/scan_product', methods=['POST'])
def scan_product():
    """Scan product barcode and analyze nutrition"""
    global current_product_data
    
    try:
        data = request.get_json()
        barcode = data.get('barcode')
        
        if not barcode:
            return jsonify({"error": "Barcode is required"}), 400
        
        # Fetch product data from OpenFoodFacts API
        product_data = fetch_product_data(barcode)
        
        if not product_data:
            return jsonify({
                "error": "Product not found",
                "product_not_found": True,
                "message": "This product is not available in our database. You can upload the ingredients and nutritional label images for manual analysis."
            }), 404
        
        # Store product data for chatbot
        current_product_data = product_data
        
        # Analyze product against user preferences
        analysis_result = analyze_product(product_data, user_preferences)
        
        return jsonify({
            "product": product_data,
            "analysis": analysis_result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload_ingredients_analysis', methods=['POST'])
def upload_ingredients_analysis():
    """Upload ingredients image for analysis"""
    global extracted_text_data, current_product_data
    
    try:
        # Check if ingredients image is provided
        if 'ingredients_image' not in request.files:
            return jsonify({"error": "Ingredients image is required"}), 400
        
        ingredients_file = request.files['ingredients_image']
        
        if ingredients_file.filename == '':
            return jsonify({"error": "Ingredients image file must be selected"}), 400
        
        # Extract text from ingredients image
        ingredients_text = extract_text_from_image(ingredients_file)
        
        if not ingredients_text:
            return jsonify({"error": "Could not extract text from the uploaded ingredients image. Please ensure the image is clear and contains readable text."}), 400
        
        # Check if nutritional label image is provided
        nutritional_text = ""
        if 'nutritional_image' in request.files:
            nutritional_file = request.files['nutritional_image']
            if nutritional_file.filename != '':
                nutritional_text = extract_text_from_image(nutritional_file)
                # Note: We extract nutritional text but don't display it to the user
        
        # Create product data from ingredients text only (nutritional data is not used)
        product_data = create_product_from_ingredients_only(ingredients_text)
        
        # Store extracted text data for chatbot (only ingredients)
        extracted_text_data = {
            "ingredients_text": ingredients_text,
            "nutritional_text": nutritional_text,  # Store but don't display
            "product_data": product_data
        }
        
        # Store product data for chatbot
        current_product_data = product_data
        
        # Analyze product against user preferences
        analysis_result = analyze_product(product_data, user_preferences)
        
        return jsonify({
            "product": product_data,
            "analysis": analysis_result,
            "extracted_text": {
                "ingredients": ingredients_text,
                "nutritional": ""  # Don't send nutritional text to frontend
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/api/analyze_ingredients_only', methods=['POST'])
def analyze_ingredients_only():
    """Analyze ingredients text directly without nutritional label"""
    global extracted_text_data, current_product_data
    
    try:
        data = request.get_json()
        ingredients_text = data.get('ingredients_text', '').strip()
        
        if not ingredients_text:
            return jsonify({"error": "Ingredients text is required"}), 400
        
        # Create basic product data from ingredients only
        product_data = create_product_from_ingredients_only(ingredients_text)
        
        # Store extracted text data for chatbot
        extracted_text_data = {
            "ingredients_text": ingredients_text,
            "nutritional_text": "",
            "product_data": product_data
        }
        
        # Store product data for chatbot
        current_product_data = product_data
        
        # Analyze product against user preferences
        analysis_result = analyze_product(product_data, user_preferences)
        
        return jsonify({
            "product": product_data,
            "analysis": analysis_result,
            "extracted_text": {
                "ingredients": ingredients_text,
                "nutritional": ""
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Error analyzing ingredients: {str(e)}"}), 500

@app.route('/api/preferences', methods=['GET', 'POST'])
def manage_preferences():
    """Get or update user preferences"""
    global user_preferences
    
    if request.method == 'GET':
        return jsonify(user_preferences)
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            user_preferences.update(data)
            return jsonify({"message": "Preferences updated successfully", "preferences": user_preferences})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/scan_image', methods=['POST'])
def scan_image():
    """Scan barcode from uploaded image"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Read the image file
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Convert to grayscale for better barcode detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect barcodes in the image
        barcodes = pyzbar.decode(gray)
        
        if not barcodes:
            return jsonify({"error": "No barcode found in the image"}), 404
        
        # Get the first barcode found
        barcode = barcodes[0]
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        
        # Fetch product data using the detected barcode
        product_data = fetch_product_data(barcode_data)
        
        if not product_data:
            return jsonify({
                "error": "Product not found for barcode: " + barcode_data,
                "product_not_found": True,
                "message": "This product is not available in our database. You can upload the ingredients and nutritional label images for manual analysis."
            }), 404
        
        # Store product data for chatbot
        global current_product_data
        current_product_data = product_data
        
        # Analyze product against user preferences
        analysis_result = analyze_product(product_data, user_preferences)
        
        return jsonify({
            "barcode": barcode_data,
            "barcode_type": barcode_type,
            "product": product_data,
            "analysis": analysis_result
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """Handle chatbot queries"""
    global current_product_data, extracted_text_data, last_request_time, request_count
    
    try:
        if not model:
            return jsonify({"error": "Chatbot not available. Please check your API key."}), 500
        
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Simple rate limiting
        import time
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - last_request_time > RATE_LIMIT_WINDOW:
            request_count = 0
            last_request_time = current_time
        
        # Check if we're within rate limits
        if request_count >= MAX_REQUESTS_PER_WINDOW:
            fallback_response = get_fallback_response(user_message, current_product_data)
            return jsonify({
                "response": fallback_response,
                "rate_limited": True,
                "fallback": True,
                "suggestion": f"Rate limit reached ({MAX_REQUESTS_PER_WINDOW} requests per minute). Please wait a moment before trying again."
            })
        
        request_count += 1
        
        # Prepare context for the chatbot
        context = ""
        if current_product_data:
            product = current_product_data
            
            # Add extracted ingredients text if available (only for database products, not uploaded ones)
            extracted_text_info = ""
            if extracted_text_data and extracted_text_data.get('ingredients_text') and product.get('source') == 'database':
                extracted_text_info = f"""
**Extracted Ingredients Text:**
{extracted_text_data.get('ingredients_text', 'No ingredients extracted')}

**Important:** Please analyze the ingredients based on their order in the list. The first few ingredients are the most important as they make up the majority of the product.
"""
            
            context = f"""
**Product Information:**
- **Name:** {product.get('name', 'Unknown')}
- **Brand:** {product.get('brand', 'Unknown')}
- **Source:** {product.get('source', 'database')}
- **Ingredients:** {product.get('ingredients', 'No ingredients listed')}
- **Allergens:** {', '.join(product.get('allergens', [])) if product.get('allergens') else 'None detected'}
- **Nutrition Grade:** {product.get('nutrition_grade', 'Not graded')}

**Nutrients (per 100g):**
- **Sugar:** {product.get('nutrients', {}).get('sugar', 0)}g
- **Fat:** {product.get('nutrients', {}).get('fat', 0)}g
- **Protein:** {product.get('nutrients', {}).get('protein', 0)}g
- **Carbohydrates:** {product.get('nutrients', {}).get('carbohydrates', 0)}g
- **Salt:** {product.get('nutrients', {}).get('salt', 0)}g
- **Calories:** {product.get('nutrients', {}).get('calories', 0)}kcal

{extracted_text_info}

**User Dietary Preferences:**
- **Diabetic:** {user_preferences.get('diabetic', False)}
- **Allergens to avoid:** {', '.join(user_preferences.get('allergens', [])) if user_preferences.get('allergens') else 'None'}
- **Age:** {user_preferences.get('age', 'Not specified')}
- **Weight:** {user_preferences.get('weight', 'Not specified')}kg
- **Exercise Intensity:** {user_preferences.get('exercise_intensity', 'Not specified')}

**User Question:** {user_message}

Please provide **short, clear, and concise** advice about this product based on the user's dietary preferences and health conditions. Focus on the most important nutrition, ingredients, and health implications.
             
**Keep your response brief and to the point:**
- Use clear headings with **bold text**
- Use bullet points for key information
- Avoid lengthy explanations
- Focus on actionable advice
- Keep each section concise
- **IMPORTANT:** Only discuss ingredients and nutritional information that are actually available in the product data. Do not mention missing ingredients or nutritional values that are not provided. Focus your analysis on what is available.

If this is a manually uploaded product (source: manual_upload or ingredients_only), pay special attention to the extracted text data and provide analysis based on the actual ingredients and nutritional information provided. Only analyze what is available - do not speculate about missing information.
"""
        else:
            context = f"""
User Dietary Preferences:
- Diabetic: {user_preferences.get('diabetic', False)}
- Allergens to avoid: {', '.join(user_preferences.get('allergens', [])) if user_preferences.get('allergens') else 'None detected'}
- Age: {user_preferences.get('age', 'Not specified')}
- Weight: {user_preferences.get('weight', 'Not specified')}kg
- Exercise Intensity: {user_preferences.get('exercise_intensity', 'Not specified')}

User Question: {user_message}

Please provide helpful, accurate, and personalized advice about nutrition and dietary recommendations based on the user's preferences and health conditions. Focus on general dietary guidance and best practices.
"""
        
        # Generate response using Gemini with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(context)
                return jsonify({
                    "response": response.text,
                    "product_loaded": current_product_data is not None
                })
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_str and "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        print(f"⚠️ Rate limit hit, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        # Final attempt failed, provide fallback response
                        fallback_response = get_fallback_response(user_message, current_product_data)
                        return jsonify({
                            "response": fallback_response,
                            "rate_limited": True,
                            "fallback": True,
                            "suggestion": "For more detailed AI assistance, try again in a few minutes or consider upgrading your API plan at https://makersuite.google.com/app/apikey"
                        })
                else:
                    # Non-rate-limit error, don't retry
                    return jsonify({"error": f"Error generating response: {error_str}"}), 500
        
    except Exception as e:
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500

def fetch_product_data(barcode):
    """Fetch product data from OpenFoodFacts API"""
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 1:
            product = data.get('product', {})
            return {
                "name": product.get('product_name', 'Unknown Product'),
                "brand": product.get('brands', 'Unknown Brand'),
                "ingredients": product.get('ingredients_text', '') or product.get('ingredients_text_en', '') or product.get('ingredients_text_fr', '') or 'No ingredients listed',
                "allergens": product.get('allergens_tags', []),
                "nutrition_grade": product.get('nutrition_grade_fr', ''),
                "nutrients": {
                    "sugar": product.get('nutriments', {}).get('sugars_100g', 0),
                    "fat": product.get('nutriments', {}).get('fat_100g', 0),
                    "protein": product.get('nutriments', {}).get('proteins_100g', 0),
                    "carbohydrates": product.get('nutriments', {}).get('carbohydrates_100g', 0),
                    "salt": product.get('nutriments', {}).get('salt_100g', 0),
                    "calories": product.get('nutriments', {}).get('energy-kcal_100g', 0)
                },
                "image_url": product.get('image_front_url', ''),
                "barcode": barcode
            }
        else:
            return None
            
    except requests.RequestException:
        return None

def get_fallback_response(user_message, product_data=None):
    """Get a fallback response when API is rate limited"""
    message_lower = user_message.lower()
    
    if product_data:
        # Check if user is asking about specific aspects
        if any(word in message_lower for word in ['diabetic', 'diabetes', 'sugar', 'sweet']):
            return FALLBACK_RESPONSES["diabetic"]
        elif any(word in message_lower for word in ['allergen', 'allergy', 'ingredient']):
            return FALLBACK_RESPONSES["allergens"]
        elif any(word in message_lower for word in ['nutrition', 'calorie', 'fat', 'protein']):
            return FALLBACK_RESPONSES["nutrition"]
    
    return FALLBACK_RESPONSES["general"]

def analyze_product(product_data, preferences):
    """Analyze product against user preferences using Neural Network and rule-based logic"""
    ingredients = product_data.get('ingredients', '').lower()
    allergens = product_data.get('allergens', [])
    nutrients = product_data.get('nutrients', {})
    
    # Try Neural Network prediction first
    nn_recommendation = None
    nn_probabilities = None
    nn_reasons = []
    
    if nn_predictor and nn_predictor.model is not None:
        try:
            nn_recommendation, nn_probabilities = nn_predictor.predict(product_data, preferences)
            nn_reasons = nn_predictor.get_prediction_reasons(product_data, preferences, nn_recommendation, nn_probabilities)
        except Exception as e:
            print(f"⚠️ Neural Network prediction failed: {e}")
            nn_recommendation = None
    
    # Check for allergens
    user_allergens = preferences.get('allergens', [])
    allergen_conflicts = []
    if ingredients and ingredients != 'No ingredients listed':
        allergen_conflicts = [allergen for allergen in user_allergens if allergen.lower() in ingredients]
    
    # Check dietary restrictions
    dietary_conflicts = []
    consumption_recommendation = "can_consume"
    
    # Sugar content check for all users
    sugar_content = nutrients.get('sugar', 0)
    
    # Check for harmful artificial sweeteners and sugar substitutes
    harmful_sweeteners = [
        "aspartame", "acesulfame potassium", "acesulfame k", "sucralose", 
        "saccharin", "neotame", "advantame", "cyclamate", "sorbitol", 
        "xylitol", "mannitol", "erythritol", "maltitol", "isomalt",
        "high fructose corn syrup", "hfcs", "corn syrup", "fructose",
        "dextrose", "maltose", "lactose", "sucrose", "glucose syrup"
    ]
    
    # Check ingredients for harmful sweeteners
    found_harmful_sweeteners = []
    if ingredients and ingredients != 'No ingredients listed':
        for sweetener in harmful_sweeteners:
            if sweetener in ingredients:
                found_harmful_sweeteners.append(sweetener)
    
    # Also check for common brand names and variations
    brand_sweeteners = {
        "nutrasweet": "aspartame",
        "equal": "aspartame", 
        "sweet n low": "saccharin",
        "splenda": "sucralose",
        "sunett": "acesulfame potassium",
        "sweet one": "acesulfame potassium",
        "newtame": "neotame",
        "sweetex": "saccharin",
        "hermesetas": "saccharin"
    }
    
    if ingredients and ingredients != 'No ingredients listed':
        for brand, sweetener in brand_sweeteners.items():
            if brand in ingredients and sweetener not in found_harmful_sweeteners:
                found_harmful_sweeteners.append(sweetener)
    
    # Sugar analysis based on diabetic status
    if preferences.get('diabetic', False):
        # Diabetic patients
        if found_harmful_sweeteners and sugar_content > 15:
            # Both sweeteners and high sugar - avoid
            dietary_conflicts.append(f"Contains sweeteners: {', '.join(found_harmful_sweeteners)} AND high sugar content ({sugar_content}g per 100g) - avoid this")
            consumption_recommendation = "avoid"
        elif found_harmful_sweeteners:
            # Only sweeteners - consume in moderation
            dietary_conflicts.append(f"Contains sweeteners: {', '.join(found_harmful_sweeteners)} - consume in moderation")
            consumption_recommendation = "consume_half"
        elif sugar_content <= 15:
            # Moderate sugar - consume in moderation
            dietary_conflicts.append(f"Moderate sugar content ({sugar_content}g per 100g) - consider in moderation")
            consumption_recommendation = "consume_half"
        elif sugar_content > 15:
            # High sugar - avoid
            dietary_conflicts.append(f"High sugar content ({sugar_content}g per 100g) - avoid this")
            consumption_recommendation = "avoid"
        # Note: Low sugar (< 5g) is safe for diabetics, so no conflict is added
    else:
        # Non-diabetic patients
        if sugar_content > 25:
            # High sugar - consume in moderation
            dietary_conflicts.append(f"High sugar content ({sugar_content}g per 100g) - consume in moderation")
            consumption_recommendation = "consume_half"
        # Note: Low and moderate sugar (≤ 25g) is safe for non-diabetics, so no conflict is added
    
    # Check for allergen conflicts
    if allergen_conflicts:
        dietary_conflicts.extend([f"Contains {allergen}" for allergen in allergen_conflicts])
        consumption_recommendation = "avoid"
    
    # Exercise intensity adjustment
    exercise_intensity = preferences.get('exercise_intensity', 'medium')
    calories = nutrients.get('calories', 0)
    
    if exercise_intensity == 'low' and calories > 300:
        dietary_conflicts.append("High calorie content for low activity level")
        if consumption_recommendation == "can_consume":
            consumption_recommendation = "consume_half"
    
    # Use Neural Network recommendation if available, otherwise use rule-based
    if nn_recommendation:
        consumption_recommendation = nn_recommendation
        # Merge NN reasons with rule-based conflicts
        if nn_reasons:
            dietary_conflicts.extend(nn_reasons)
    
    # Note: We don't add warnings about missing ingredients to avoid cluttering the chatbot response
    # The chatbot will focus on available information only
    
    result = {
        "consumption_recommendation": consumption_recommendation,
        "dietary_conflicts": dietary_conflicts,
        "allergen_conflicts": allergen_conflicts,
        "nutrition_summary": {
            "calories": calories,
            "sugar": nutrients.get('sugar', 0),
            "fat": nutrients.get('fat', 0),
            "protein": nutrients.get('protein', 0),
            "carbohydrates": nutrients.get('carbohydrates', 0)
        }
    }
    
    # Add Neural Network information if available
    if nn_recommendation and nn_probabilities:
        result["nn_prediction"] = {
            "recommendation": nn_recommendation,
            "probabilities": nn_probabilities,
            "confidence": nn_probabilities[nn_recommendation],
            "method": "neural_network"
        }
    else:
        result["nn_prediction"] = {
            "method": "rule_based",
            "note": "Neural Network model not available, using rule-based analysis"
        }
    
    return result

def extract_text_from_image(image_file):
    """Extract text from uploaded image using OCR"""
    try:
        # Read image file
        file_bytes = image_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return ""
        
        # Convert to PIL Image for pytesseract
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Try EasyOCR first (better accuracy)
        if reader:
            try:
                results = reader.readtext(image)
                text = ' '.join([result[1] for result in results])
                if text.strip():
                    return text.strip()
            except Exception as e:
                print(f"EasyOCR failed: {e}")
        
        # Fallback to pytesseract if available
        if TESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(pil_image)
                return text.strip()
            except Exception as e:
                print(f"Pytesseract failed: {e}")
                return ""
        else:
            print("⚠️ Tesseract OCR not available. Please install it for better text extraction.")
            return ""
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def create_product_from_text(ingredients_text, nutritional_text):
    """Create product data from extracted text"""
    # Extract basic nutrition info from nutritional text
    nutrition_data = extract_nutrition_from_text(nutritional_text)
    
    return {
        "name": "Product (Manual Analysis)",
        "brand": "Unknown",
        "ingredients": ingredients_text,
        "allergens": extract_allergens_from_text(ingredients_text),
        "nutrition_grade": "Unknown",
        "nutrients": nutrition_data,
        "image_url": "",
        "barcode": "manual_analysis",
        "source": "manual_upload"
    }

def create_product_from_ingredients_only(ingredients_text):
    """Create product data from ingredients text only"""
    return {
        "name": "Product (Ingredients Analysis)",
        "brand": "Unknown",
        "ingredients": ingredients_text,
        "allergens": extract_allergens_from_text(ingredients_text),
        "nutrition_grade": "Unknown",
        "nutrients": {
            "sugar": 0,
            "fat": 0,
            "protein": 0,
            "carbohydrates": 0,
            "salt": 0,
            "calories": 0
        },
        "image_url": "",
        "barcode": "ingredients_only",
        "source": "ingredients_only"
    }

def extract_nutrition_from_text(text):
    """Extract nutrition information from text"""
    if not text:
        return {"sugar": 0, "fat": 0, "protein": 0, "carbohydrates": 0, "salt": 0, "calories": 0}
    
    text_lower = text.lower()
    
    # Extract calories
    calories_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:kcal|calories?)', text_lower)
    calories = float(calories_match.group(1)) if calories_match else 0
    
    # Extract sugar
    sugar_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s+)?sugar', text_lower)
    sugar = float(sugar_match.group(1)) if sugar_match else 0
    
    # Extract fat
    fat_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s+)?(?:total\s+)?fat', text_lower)
    fat = float(fat_match.group(1)) if fat_match else 0
    
    # Extract protein
    protein_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s+)?protein', text_lower)
    protein = float(protein_match.group(1)) if protein_match else 0
    
    # Extract carbohydrates
    carbs_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s+)?(?:total\s+)?carbohydrates?', text_lower)
    carbs = float(carbs_match.group(1)) if carbs_match else 0
    
    # Extract salt/sodium
    salt_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s+)?(?:salt|sodium)', text_lower)
    salt = float(salt_match.group(1)) if salt_match else 0
    
    return {
        "sugar": sugar,
        "fat": fat,
        "protein": protein,
        "carbohydrates": carbs,
        "salt": salt,
        "calories": calories
    }

def extract_allergens_from_text(text):
    """Extract allergens from ingredients text"""
    if not text:
        return []
    
    text_lower = text.lower()
    common_allergens = [
        "milk", "eggs", "fish", "shellfish", "tree nuts", "peanuts", "wheat", "soybeans",
        "gluten", "lactose", "casein", "whey", "albumin", "gelatin", "lecithin"
    ]
    
    found_allergens = []
    for allergen in common_allergens:
        if allergen in text_lower:
            found_allergens.append(allergen)
    
    return found_allergens

def analyze_ingredients_logic(ingredients_text, user_preferences):
    """Analyze ingredients using the specified logic"""
    if not ingredients_text:
        return "No ingredients available for analysis"
    
    # Split ingredients and clean them
    ingredients_list = [ingredient.strip().lower() for ingredient in ingredients_text.split(',')]
    ingredients_list = [ingredient for ingredient in ingredients_list if ingredient]
    
    if len(ingredients_list) < 3:
        return "Limited ingredients available. Analysis based on available ingredients only."
    
    # Check first 3 ingredients for sugar
    first_three = ingredients_list[:3]
    first_five = ingredients_list[:5] if len(ingredients_list) >= 5 else ingredients_list
    
    sugar_indicators = [
        "sugar", "glucose", "fructose", "sucrose", "high fructose corn syrup", "hfcs",
        "maltose", "dextrose", "corn syrup", "glucose syrup", "lactose", "maltodextrin"
    ]
    
    # Check if sugar appears in first 3 ingredients
    sugar_in_first_three = any(sugar in ' '.join(first_three) for sugar in sugar_indicators)
    
    # Check if sugar appears in first 5 ingredients
    sugar_in_first_five = any(sugar in ' '.join(first_five) for sugar in sugar_indicators)
    
    is_diabetic = user_preferences.get('diabetic', False)
    
    if is_diabetic:
        if sugar_in_first_three:
            return "❌ AVOID - Sugar appears in first 3 ingredients. This product is not suitable for diabetic patients."
        elif sugar_in_first_five:
            return "⚠️ CONSUME IN MODERATION - Sugar appears in first 5 ingredients. Limit consumption for diabetic patients."
        else:
            return "✅ SAFE TO CONSUME - No sugar detected in first 5 ingredients. Suitable for diabetic patients."
    else:
        if sugar_in_first_three:
            return "⚠️ CONSIDER IN MODERATION - Sugar appears in first 3 ingredients. Monitor your sugar intake."
        elif sugar_in_first_five:
            return "✅ SAFE TO CONSUME - Sugar appears in first 5 ingredients but not in top 3. Generally safe for non-diabetic patients."
        else:
            return "✅ SAFE TO CONSUME - No sugar detected in first 5 ingredients. Safe for consumption."



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port) 