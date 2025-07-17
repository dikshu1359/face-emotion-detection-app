import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import time
from datetime import datetime
import threading
import queue

class EmotionUtils:
    """Utility functions for emotion detection app"""
    
    def __init__(self):
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 255),    # Yellow
            'neutral': (128, 128, 128), # Gray
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 165, 255)  # Orange
        }
        
        self.emotion_emojis = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢',
            'surprise': '😮'
        }
    
    def preprocess_image(self, image, target_size=(48, 48)):
        """Preprocess image for model input"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize
        resized = cv2.resize(gray, target_size)
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        processed = normalized.reshape(1, target_size[0], target_size[1], 1)
        
        return processed
    
    def draw_emotion_box(self, image, x, y, w, h, emotion, confidence):
        """Draw emotion detection box on image"""
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        emoji = self.emotion_emojis.get(emotion, '🤔')
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion text
        text = f"{emoji} {emotion} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), color, -1)
        
        # Draw text
        cv2.putText(image, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
        
        return image
    
    def create_emotion_graph(self, predictions, emotion_labels):
        """Create emotion prediction graph"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [self.emotion_colors[label] for label in emotion_labels]
        colors = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]  # BGR to RGB
        
        bars = ax.bar(emotion_labels, predictions, color=colors)
        
        # Add value labels on bars
        for bar, pred in zip(bars, predictions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pred:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Confidence')
        ax.set_title('Emotion Predictions')
        ax.set_ylim(0, 1)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def save_detection_result(self, image, results, filename=None):
        """Save detection results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_detection_{timestamp}.json"
        
        # Prepare data
        data = {
            'timestamp': datetime.now().isoformat(),
            'detections': []
        }
        
        for result in results:
            detection = {
                'emotion': result['emotion'],
                'confidence': float(result['confidence']),
                'coordinates': result['coordinates'],
                'all_predictions': result['all_predictions']
            }
            data['detections'].append(detection)
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename
    
    def load_detection_results(self, filename):
        """Load detection results from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
    
    def calculate_emotion_statistics(self, results_list):
        """Calculate emotion statistics from multiple detections"""
        emotion_counts = {emotion: 0 for emotion in self.emotion_colors.keys()}
        total_detections = 0
        
        for results in results_list:
            if 'detections' in results:
                for detection in results['detections']:
                    emotion = detection['emotion']
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1
                        total_detections += 1
        
        # Calculate percentages
        emotion_percentages = {}
        if total_detections > 0:
            for emotion, count in emotion_counts.items():
                emotion_percentages[emotion] = (count / total_detections) * 100
        
        return emotion_counts, emotion_percentages
    
    def create_emotion_timeline(self, results_list):
        """Create emotion timeline from detection results"""
        timestamps = []
        emotions = []
        
        for results in results_list:
            if 'detections' in results and results['detections']:
                timestamp = datetime.fromisoformat(results['timestamp'])
                # Take the most confident detection
                best_detection = max(results['detections'], 
                                   key=lambda x: x['confidence'])
                
                timestamps.append(timestamp)
                emotions.append(best_detection['emotion'])
        
        return timestamps, emotions
    
    def plot_emotion_timeline(self, timestamps, emotions):
        """Plot emotion timeline"""
        if not timestamps or not emotions:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create numerical representation of emotions
        emotion_to_num = {emotion: i for i, emotion in enumerate(self.emotion_colors.keys())}
        y_values = [emotion_to_num[emotion] for emotion in emotions]
        
        # Plot timeline
        ax.plot(timestamps, y_values, 'o-', markersize=8, linewidth=2)
        
        # Customize plot
        ax.set_yticks(range(len(self.emotion_colors)))
        ax.set_yticklabels(list(self.emotion_colors.keys()))
        ax.set_xlabel('Time')
        ax.set_ylabel('Emotion')
        ax.set_title('Emotion Timeline')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def benchmark_model(self, model, test_images, num_runs=10):
        """Benchmark model performance"""
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            for image in test_images:
                processed = self.preprocess_image(image)
                _ = model.predict(processed, verbose=0)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = len(test_images) / avg_time
        
        return {
            'average_time': avg_time,
            'std_time': std_time,
            'fps': fps,
            'images_per_second': fps
        }
    
    def create_confidence_heatmap(self, predictions_list):
        """Create confidence heatmap for multiple predictions"""
        if not predictions_list:
            return None
        
        # Stack predictions
        stacked_predictions = np.array(predictions_list)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(stacked_predictions.T, cmap='hot', aspect='auto')
        
        # Customize
        ax.set_xlabel('Prediction Number')
        ax.set_ylabel('Emotions')
        ax.set_yticks(range(len(self.emotion_colors)))
        ax.set_yticklabels(list(self.emotion_colors.keys()))
        ax.set_title('Confidence Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Confidence')
        
        plt.tight_layout()
        return fig
    
    def filter_low_confidence_detections(self, results, min_confidence=0.5):
        """Filter out low confidence detections"""
        filtered_results = []
        
        for result in results:
            if result['confidence'] >= min_confidence:
                filtered_results.append(result)
        
        return filtered_results
    
    def smooth_predictions(self, predictions_queue, window_size=5):
        """Smooth predictions using moving average"""
        if len(predictions_queue) < window_size:
            return predictions_queue[-1] if predictions_queue else None
        
        # Get last window_size predictions
        recent_predictions = list(predictions_queue)[-window_size:]
        
        # Calculate average
        avg_predictions = np.mean(recent_predictions, axis=0)
        
        return avg_predictions
    
    def validate_image_format(self, image_path):
        """Validate if image format is supported"""
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        
        _, ext = os.path.splitext(image_path.lower())
        return ext in supported_formats
    
    def resize_image_for_display(self, image, max_size=500):
        """Resize image for display purposes"""
        height, width = image.shape[:2]
        
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(image, (new_width, new_height))
            return resized
        
        return image
    
    def create_emotion_report(self, results_list, output_path='emotion_report.html'):
        """Create HTML report of emotion analysis"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Emotion Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .result {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; }}
                .emotion {{ font-weight: bold; color: #333; }}
                .confidence {{ color: #666; }}
                .timestamp {{ color: #888; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Emotion Detection Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        for i, results in enumerate(results_list):
            if 'detections' in results:
                html_content += f"""
                <div class="result">
                    <h3>Detection {i+1}</h3>
                    <div class="timestamp">{results['timestamp']}</div>
                """
                
                for detection in results['detections']:
                    emotion = detection['emotion']
                    confidence = detection['confidence']
                    emoji = self.emotion_emojis.get(emotion, '🤔')
                    
                    html_content += f"""
                    <div class="emotion">{emoji} {emotion.title()}</div>
                    <div class="confidence">Confidence: {confidence:.3f}</div>
                    """
                
                html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path

# Example usage
if __name__ == "__main__":
    utils = EmotionUtils()
    
    # Test image preprocessing
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    processed = utils.preprocess_image(test_image)
    print(f"Processed image shape: {processed.shape}")
    
    # Test emotion colors
    print("Emotion colors:")
    for emotion, color in utils.emotion_colors.items():
        print(f"{emotion}: {color}")
    
    print("Utility functions ready!")