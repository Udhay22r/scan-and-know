"""
Training script for the consumption recommendation neural network
Generates synthetic training data based on nutrition rules
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from nn_model import ConsumptionPredictor, ConsumptionNet

def generate_synthetic_data(n_samples=5000):
    """Generate synthetic training data based on nutrition rules"""
    print("Generating synthetic training data...")
    
    data = []
    
    for i in range(n_samples):
        # Generate random nutrient values
        sugar = np.random.uniform(0, 50)
        fat = np.random.uniform(0, 50)
        protein = np.random.uniform(0, 30)
        carbohydrates = np.random.uniform(0, 100)
        salt = np.random.uniform(0, 10)
        calories = np.random.uniform(50, 600)
        
        # Generate user preferences
        is_diabetic = np.random.choice([0, 1])
        age = np.random.uniform(18, 80)
        weight = np.random.uniform(45, 120)
        exercise_intensity = np.random.choice(['low', 'medium', 'high'])
        
        # Generate binary features
        has_allergen = np.random.choice([0, 1], p=[0.8, 0.2])
        has_sweeteners = np.random.choice([0, 1], p=[0.7, 0.3])
        sugar_in_first_3 = np.random.choice([0, 1], p=[0.7, 0.3])
        sugar_in_first_5 = np.random.choice([0, 1], p=[0.5, 0.5])
        
        # Derived features
        sugar_per_calorie = sugar / max(calories, 1)
        fat_per_calorie = fat / max(calories, 1)
        protein_per_calorie = protein / max(calories, 1)
        ingredient_count = np.random.randint(3, 20)
        
        # Determine label based on rules (simulating existing logic)
        label = determine_label(
            sugar, calories, is_diabetic, has_allergen, 
            has_sweeteners, sugar_in_first_3, exercise_intensity, fat
        )
        
        # Encode exercise intensity
        exercise_map = {'low': 0, 'medium': 1, 'high': 2}
        exercise_encoded = exercise_map[exercise_intensity]
        
        # Encode label
        label_map = {'can_consume': 0, 'consume_half': 1, 'avoid': 2}
        label_encoded = label_map[label]
        
        # Create feature vector
        features = [
            sugar, fat, protein, carbohydrates, salt, calories,
            is_diabetic, age, weight, exercise_encoded,
            has_allergen, has_sweeteners,
            sugar_in_first_3, sugar_in_first_5,
            sugar_per_calorie, fat_per_calorie, protein_per_calorie,
            ingredient_count
        ]
        
        data.append(features + [label_encoded])
    
    # Convert to DataFrame
    columns = [
        'sugar', 'fat', 'protein', 'carbohydrates', 'salt', 'calories',
        'is_diabetic', 'age', 'weight', 'exercise_intensity',
        'has_allergen', 'has_sweeteners',
        'sugar_in_first_3', 'sugar_in_first_5',
        'sugar_per_calorie', 'fat_per_calorie', 'protein_per_calorie',
        'ingredient_count', 'label'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    return df

def determine_label(sugar, calories, is_diabetic, has_allergen, 
                   has_sweeteners, sugar_in_first_3, exercise_intensity, fat):
    """Determine label based on nutrition rules"""
    
    # If has allergens, avoid
    if has_allergen:
        return 'avoid'
    
    # Diabetic patients
    if is_diabetic:
        if has_sweeteners and sugar > 15:
            return 'avoid'
        elif has_sweeteners or sugar > 15:
            return 'consume_half'
        elif sugar > 5:
            return 'consume_half'
        elif sugar_in_first_3:
            return 'avoid'
        elif sugar > 10:
            return 'consume_half'
        else:
            return 'can_consume'
    else:
        # Non-diabetic patients
        if sugar > 25:
            return 'consume_half'
        elif sugar_in_first_3 and sugar > 15:
            return 'consume_half'
        elif exercise_intensity == 'low' and calories > 300:
            return 'consume_half'
        elif fat > 40:
            return 'consume_half'
        else:
            return 'can_consume'

def train_model(dataset_path=None, save_dataset=False):
    """Train the neural network model
    
    Args:
        dataset_path: Path to load existing dataset CSV file (optional)
        save_dataset: If True, save generated dataset to CSV file
    """
    print("=" * 60)
    print("Training Consumption Recommendation Neural Network (PyTorch)")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or generate dataset
    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("Generating synthetic training data...")
        df = generate_synthetic_data(n_samples=5000)
        
        # Save dataset if requested
        if save_dataset:
            dataset_save_path = dataset_path or 'data/training_dataset.csv'
            os.makedirs(os.path.dirname(dataset_save_path) if os.path.dirname(dataset_save_path) else '.', exist_ok=True)
            df.to_csv(dataset_save_path, index=False)
            print(f"Dataset saved to: {dataset_save_path}")
    
    # Split features and labels
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values
    
    # Convert labels to one-hot encoding
    num_classes = 3
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_dim = X_train_scaled.shape[1]
    model = ConsumptionNet(input_dim=input_dim).to(device)
    
    print("\nModel Architecture:")
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("\nTraining model...")
    num_epochs = 50
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(batch_y.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(batch_y.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(test_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    # Final evaluation
    print("\nEvaluating model...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(batch_y.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    predictor = ConsumptionPredictor()
    predictor.save_model(model, scaler)
    
    # Print class distribution
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {predictor.model_path}")
    print(f"Scaler saved to: {predictor.scaler_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train consumption recommendation neural network')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset CSV file (optional)')
    parser.add_argument('--save-dataset', action='store_true', help='Save generated dataset to CSV file')
    args = parser.parse_args()
    
    train_model(dataset_path=args.dataset, save_dataset=args.save_dataset)
