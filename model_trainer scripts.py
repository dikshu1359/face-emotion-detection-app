import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from data_loader import EmotionDataLoader
import os

class EmotionModelTrainer:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def create_cnn_model(self):
        """Create CNN model for emotion detection"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_advanced_model(self):
        """Create a more advanced model with residual connections"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial conv
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Residual block 1
        residual = x
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Residual block 2
        residual = layers.Conv2D(64, (1, 1), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Residual block 3
        residual = layers.Conv2D(128, (1, 1), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_save_path='best_emotion_model.h5'):
        """Get training callbacks"""
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=50, batch_size=32, model_type='cnn'):
        """Train the emotion detection model"""
        # Create model
        if model_type == 'cnn':
            self.model = self.create_cnn_model()
        elif model_type == 'advanced':
            self.model = self.create_advanced_model()
        else:
            raise ValueError("Model type must be 'cnn' or 'advanced'")
        
        # Compile model
        self.model = self.compile_model(self.model)
        
        # Print model summary
        print("Model Summary:")
        self.model.summary()
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test data"""
        if self.model is None:
            print("No model available for evaluation")
            return None
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        return test_accuracy, test_loss, predictions
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Convert one-hot to class indices
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='emotion_model.h5'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='emotion_model.h5'):
        """Load a saved model"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict_emotion(self, image):
        """Predict emotion for a single image"""
        if self.model is None:
            print("No model available for prediction")
            return None
        
        # Preprocess image
        if len(image.shape) == 2:
            image = image.reshape(1, 48, 48, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, image.shape[0], image.shape[1], 1)
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Predict
        prediction = self.model.predict(image, verbose=0)
        
        # Get emotion
        emotion_idx = np.argmax(prediction[0])
        emotion = self.emotion_labels[emotion_idx]
        confidence = float(prediction[0][emotion_idx])
        
        return emotion, confidence, prediction[0]

# Example usage and training script
def main():
    # Initialize trainer
    trainer = EmotionModelTrainer()
    
    # Initialize data loader
    loader = EmotionDataLoader()
    
    # Load your data here
    # Option 1: Load from FER2013 CSV
    # X, y = loader.load_fer2013_dataset('fer2013.csv')
    
    # Option 2: Load from folder structure
    # X, y = loader.load_folder_dataset('emotion_dataset')
    
    # For demonstration, create dummy data
    print("Creating dummy data for demonstration...")
    X = np.random.rand(1000, 48, 48, 1)
    y = np.random.randint(0, 7, (1000,))
    y = tf.keras.utils.to_categorical(y, num_classes=7)
    
    if X is not None:
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Train model
        print("Training model...")
        history = trainer.train_model(
            X_train, y_train, X_val, y_val,
            epochs=20,  # Reduced for demo
            batch_size=32,
            model_type='cnn'
        )
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate model
        print("Evaluating model...")
        test_accuracy, test_loss, predictions = trainer.evaluate_model(X_test, y_test)
        
        # Plot confusion matrix
        trainer.plot_confusion_matrix(y_test, predictions)
        
        # Save model
        trainer.save_model('trained_emotion_model.h5')
        
        print("Training completed!")
    
    else:
        print("No data loaded. Please prepare your dataset first.")

if __name__ == "__main__":
    main()