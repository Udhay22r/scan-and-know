"""
Feed-forward Neural Network for Product Consumption Recommendation
Predicts: can_consume, consume_half, avoid
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import os
import pickle

class ConsumptionNet(nn.Module):
    """PyTorch Neural Network model for consumption recommendations"""
    
    def __init__(self, input_dim=18):
        super(ConsumptionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)  # 3 classes: can_consume, consume_half, avoid
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)

class ConsumptionPredictor:
    """Neural Network model for predicting product consumption recommendations"""
    
    def __init__(self, model_path='models/consumption_model.pth', scaler_path='models/scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
    def build_model(self, input_dim=18):
        """Build the feed-forward neural network"""
        model = ConsumptionNet(input_dim=input_dim)
        return model.to(self.device)
    
    def create_scaler(self):
        """Create a scaler for feature normalization"""
        return StandardScaler()
    
    def extract_features(self, product_data, preferences):
        """Extract features from product data and user preferences"""
        nutrients = product_data.get('nutrients', {})
        ingredients = product_data.get('ingredients', '').lower()
        allergens = product_data.get('allergens', [])
        user_allergens = preferences.get('allergens', [])
        
        # Nutrient features
        sugar = nutrients.get('sugar', 0)
        fat = nutrients.get('fat', 0)
        protein = nutrients.get('protein', 0)
        carbohydrates = nutrients.get('carbohydrates', 0)
        salt = nutrients.get('salt', 0)
        calories = nutrients.get('calories', 0)
        
        # User preference features
        is_diabetic = 1.0 if preferences.get('diabetic', False) else 0.0
        age = preferences.get('age', 25)
        weight = preferences.get('weight', 70)
        
        # Exercise intensity encoding (low=0, medium=1, high=2)
        exercise_intensity = preferences.get('exercise_intensity', 'medium')
        exercise_map = {'low': 0.0, 'medium': 1.0, 'high': 2.0}
        exercise_encoded = exercise_map.get(exercise_intensity, 1.0)
        
        # Check for allergens
        has_allergen = 1.0 if any(allergen.lower() in ingredients for allergen in user_allergens) else 0.0
        
        # Check for harmful sweeteners
        harmful_sweeteners = [
            "aspartame", "acesulfame potassium", "acesulfame k", "sucralose", 
            "saccharin", "neotame", "advantame", "cyclamate", "sorbitol", 
            "xylitol", "mannitol", "erythritol", "maltitol", "isomalt",
            "high fructose corn syrup", "hfcs", "corn syrup", "fructose",
            "dextrose", "maltose", "lactose", "sucrose", "glucose syrup"
        ]
        has_sweeteners = 1.0 if any(sweetener in ingredients for sweetener in harmful_sweeteners) else 0.0
        
        # Check sugar position in ingredients
        if ingredients and ingredients != 'no ingredients listed':
            ingredients_list = [ing.strip() for ing in ingredients.split(',')]
            first_three = ' '.join(ingredients_list[:3]).lower()
            first_five = ' '.join(ingredients_list[:5] if len(ingredients_list) >= 5 else ingredients_list).lower()
            
            sugar_indicators = ["sugar", "glucose", "fructose", "sucrose", "high fructose corn syrup", 
                              "hfcs", "maltose", "dextrose", "corn syrup", "glucose syrup", 
                              "lactose", "maltodextrin"]
            
            sugar_in_first_3 = 1.0 if any(sugar_ind in first_three for sugar_ind in sugar_indicators) else 0.0
            sugar_in_first_5 = 1.0 if any(sugar_ind in first_five for sugar_ind in sugar_indicators) else 0.0
        else:
            sugar_in_first_3 = 0.0
            sugar_in_first_5 = 0.0
        
        # Additional derived features
        sugar_per_calorie = sugar / max(calories, 1)
        fat_per_calorie = fat / max(calories, 1)
        protein_per_calorie = protein / max(calories, 1)
        
        # Feature vector: 18 features
        features = np.array([
            sugar, fat, protein, carbohydrates, salt, calories,
            is_diabetic, age, weight, exercise_encoded,
            has_allergen, has_sweeteners,
            sugar_in_first_3, sugar_in_first_5,
            sugar_per_calorie, fat_per_calorie, protein_per_calorie,
            len(ingredients.split(',')) if ingredients else 0  # ingredient count
        ], dtype=np.float32)
        
        return features
    
    def predict(self, product_data, preferences):
        """Predict consumption recommendation"""
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return None, None
        
        # Extract features
        features = self.extract_features(product_data, preferences)
        features = features.reshape(1, -1)
        
        # Normalize features
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Predict
        with torch.no_grad():
            predictions = self.model(features_tensor)
            predictions = predictions.cpu().numpy()[0]
        
        predicted_class = np.argmax(predictions)
        
        # Map to labels
        class_labels = ['can_consume', 'consume_half', 'avoid']
        recommendation = class_labels[predicted_class]
        
        # Get probabilities for all classes
        probabilities = {
            'can_consume': float(predictions[0]),
            'consume_half': float(predictions[1]),
            'avoid': float(predictions[2])
        }
        
        return recommendation, probabilities
    
    def get_prediction_reasons(self, product_data, preferences, recommendation, probabilities):
        """Generate reasons for the prediction"""
        reasons = []
        nutrients = product_data.get('nutrients', {})
        ingredients = product_data.get('ingredients', '').lower()
        user_allergens = preferences.get('allergens', [])
        is_diabetic = preferences.get('diabetic', False)
        
        sugar = nutrients.get('sugar', 0)
        calories = nutrients.get('calories', 0)
        fat = nutrients.get('fat', 0)
        protein = nutrients.get('protein', 0)
        
        # Check allergens
        if any(allergen.lower() in ingredients for allergen in user_allergens):
            reasons.append("Contains allergens that you need to avoid")
        
        # Check sugar content
        if is_diabetic:
            if sugar > 15:
                reasons.append(f"High sugar content ({sugar}g per 100g) - risky for diabetic patients")
            elif sugar > 5:
                reasons.append(f"Moderate sugar content ({sugar}g per 100g) - consume in moderation")
        else:
            if sugar > 25:
                reasons.append(f"Very high sugar content ({sugar}g per 100g)")
            elif sugar > 15:
                reasons.append(f"High sugar content ({sugar}g per 100g)")
        
        # Check sweeteners
        harmful_sweeteners = ["aspartame", "sucralose", "saccharin", "acesulfame"]
        if any(sweetener in ingredients for sweetener in harmful_sweeteners):
            reasons.append("Contains artificial sweeteners")
        
        # Check calories vs exercise
        exercise_intensity = preferences.get('exercise_intensity', 'medium')
        if exercise_intensity == 'low' and calories > 300:
            reasons.append(f"High calorie content ({calories}kcal) for low activity level")
        
        # Check protein content (positive indicator)
        if protein > 10 and recommendation == 'can_consume':
            reasons.append(f"Good protein content ({protein}g per 100g)")
        
        # Check fat content
        if fat > 30:
            reasons.append(f"High fat content ({fat}g per 100g)")
        
        # Add confidence-based reason
        if probabilities[recommendation] > 0.8:
            reasons.append(f"High confidence prediction ({probabilities[recommendation]*100:.1f}%)")
        elif probabilities[recommendation] < 0.6:
            reasons.append(f"Moderate confidence prediction ({probabilities[recommendation]*100:.1f}%)")
        
        return reasons
    
    def save_model(self, model, scaler):
        """Save the trained model and scaler"""
        torch.save(model.state_dict(), self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Model saved to {self.model_path}")
        print(f"Scaler saved to {self.scaler_path}")
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            if os.path.exists(self.model_path):
                # Determine input dimension (default 18)
                input_dim = 18
                self.model = ConsumptionNet(input_dim=input_dim)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model not found at {self.model_path}")
                return False
            
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Scaler loaded from {self.scaler_path}")
            else:
                print(f"⚠️ Scaler not found at {self.scaler_path}")
                return False
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
