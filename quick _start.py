import sys
import subprocess
import importlib

# List of required packages
REQUIRED_PACKAGES = [
    'tensorflow',
    'numpy',
    'matplotlib',
    'seaborn',
    'scikit-learn'
]

def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = importlib.import_module(package)

# Install required packages
for pkg in REQUIRED_PACKAGES:
    install_and_import(pkg)

import numpy as np
import tensorflow as tf
from model_trainer_scripts import EmotionModelTrainer  # Adjust import if needed

print("\n=== Quick Start: Emotion Detection Model ===\n")

# Create dummy data for demonstration
X = np.random.rand(500, 48, 48, 1)
y = np.random.randint(0, 7, (500,))
y = tf.keras.utils.to_categorical(y, num_classes=7)

# Split data (simple split for demo)
X_train, X_val, X_test = X[:350], X[350:425], X[425:]
y_train, y_val, y_test = y[:350], y[350:425], y[425:]

# Initialize trainer
trainer = EmotionModelTrainer()

# Train model (short run for demo)
print("Training model (demo run)...")
history = trainer.train_model(
    X_train, y_train, X_val, y_val,
    epochs=3,
    batch_size=32,
    model_type='cnn'
)

# Evaluate model
print("\nEvaluating model...")
test_accuracy, test_loss, predictions = trainer.evaluate_model(X_test, y_test)

print(f"\nDemo complete! Test accuracy: {test_accuracy:.4f}")
print("\nNext steps:")
print("- Prepare your real dataset and update the data loading section.")
print("- Explore model_trainer scripts.py for more options.")
print("- Use trainer.save_model('your_model.h5') to save your trained model.")
