import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class EmotionDataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.emotion_labels)
        
    def load_fer2013_dataset(self, csv_path):
        """Load FER2013 dataset from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            
            # Extract pixel data and labels
            X = []
            y = []
            
            for index, row in df.iterrows():
                # Convert pixel string to array
                pixels = np.array(row['pixels'].split(), dtype=np.float32)
                # Reshape to 48x48
                image = pixels.reshape(48, 48)
                # Normalize
                image = image / 255.0
                
                X.append(image)
                y.append(row['emotion'])
            
            X = np.array(X)
            y = np.array(y)
            
            # Reshape for CNN input
            X = X.reshape(X.shape[0], 48, 48, 1)
            
            # Convert labels to categorical
            y = to_categorical(y, num_classes=len(self.emotion_labels))
            
            return X, y
            
        except Exception as e:
            print(f"Error loading FER2013 dataset: {e}")
            return None, None
    
    def load_folder_dataset(self, folder_path):
        """Load dataset from folder structure: folder_path/emotion/image.jpg"""
        X = []
        y = []
        
        for emotion in self.emotion_labels:
            emotion_folder = os.path.join(folder_path, emotion)
            if not os.path.exists(emotion_folder):
                continue
                
            for filename in os.listdir(emotion_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(emotion_folder, filename)
                    
                    # Load and preprocess image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (48, 48))
                        img = img.astype('float32') / 255.0
                        
                        X.append(img)
                        y.append(emotion)
        
        X = np.array(X)
        X = X.reshape(X.shape[0], 48, 48, 1)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=len(self.emotion_labels))
        
        return X, y_categorical
    
    def augment_data(self, X, y):
        """Apply data augmentation"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        return datagen, X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets"""
        # First split into train and temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=42
        )
        
        # Then split temp into validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def visualize_data(self, X, y, num_samples=10):
        """Visualize sample data"""
        plt.figure(figsize=(15, 6))
        
        for i in range(num_samples):
            plt.subplot(2, 5, i+1)
            plt.imshow(X[i].reshape(48, 48), cmap='gray')
            emotion_idx = np.argmax(y[i])
            plt.title(self.emotion_labels[emotion_idx])
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self, y):
        """Get class distribution"""
        class_counts = np.sum(y, axis=0)
        distribution = {}
        
        for i, emotion in enumerate(self.emotion_labels):
            distribution[emotion] = int(class_counts[i])
        
        return distribution
    
    def plot_class_distribution(self, y):
        """Plot class distribution"""
        distribution = self.get_class_distribution(y)
        
        plt.figure(figsize=(10, 6))
        plt.bar(distribution.keys(), distribution.values())
        plt.title('Class Distribution')
        plt.xlabel('Emotions')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return distribution

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = EmotionDataLoader()
    
    # Example 1: Load from FER2013 CSV
    # X, y = loader.load_fer2013_dataset('fer2013.csv')
    
    # Example 2: Load from folder structure
    # X, y = loader.load_folder_dataset('emotion_dataset')
    
    # Example with dummy data
    X = np.random.rand(100, 48, 48, 1)
    y = np.random.randint(0, 7, (100,))
    y = to_categorical(y, num_classes=7)
    
    if X is not None:
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Visualize data
        # loader.visualize_data(X_train, y_train)
        
        # Show class distribution
        distribution = loader.get_class_distribution(y_train)
        print("Class distribution:", distribution)