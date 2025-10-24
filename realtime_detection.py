import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time

class FacialExpressionDetector:
    def __init__(self, model_path='best_model.keras'):
        """Initialize the facial expression detector"""
        print("Loading model...")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Smoothing buffer for predictions
        self.prediction_buffer = deque(maxlen=10)
        
        # Colors for each emotion (BGR format)
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 0),    # Dark Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 255, 255), # Yellow
            'Neutral': (255, 255, 255) # White
        }
        
    def preprocess_face(self, face_img):
        """Preprocess face image for model prediction"""
        # Resize to 48x48
        face_img = cv2.resize(face_img, (48, 48))
        # Convert to grayscale
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Normalize
        face_img = face_img / 255.0
        # Reshape for model input
        face_img = face_img.reshape(1, 48, 48, 1)
        return face_img
    
    def get_smoothed_prediction(self, predictions):
        """Smooth predictions over time to reduce jitter"""
        self.prediction_buffer.append(predictions)
        if len(self.prediction_buffer) > 0:
            avg_predictions = np.mean(self.prediction_buffer, axis=0)
            return avg_predictions
        return predictions
    
    def detect_emotion(self, frame):
        """Detect faces and predict emotions in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(48, 48)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess
            processed_face = self.preprocess_face(face_roi)
            
            # Predict
            predictions = self.model.predict(processed_face, verbose=0)[0]
            
            # Smooth predictions
            smoothed_predictions = self.get_smoothed_prediction(predictions)
            
            # Get emotion
            emotion_idx = np.argmax(smoothed_predictions)
            emotion = self.emotion_labels[emotion_idx]
            confidence = smoothed_predictions[emotion_idx]
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': smoothed_predictions
            })
        
        return results
    
    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        for result in results:
            x, y, w, h = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']
            
            # Get color for emotion
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label with confidence
            label = f"{emotion}: {confidence*100:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(
                frame, 
                (x, y - label_size[1] - 10), 
                (x + label_size[0], y), 
                color, 
                -1
            )
            
            # Text
            cv2.putText(
                frame, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                2
            )
            
            # Draw emotion bars on the side
            self.draw_emotion_bars(frame, result['all_predictions'])
        
        return frame
    
    def draw_emotion_bars(self, frame, predictions):
        """Draw emotion probability bars"""
        h, w = frame.shape[:2]
        bar_height = 30
        bar_width = 200
        x_offset = w - bar_width - 20
        y_offset = 20
        
        for i, (emotion, prob) in enumerate(zip(self.emotion_labels, predictions)):
            y = y_offset + i * (bar_height + 5)
            
            # Background bar
            cv2.rectangle(
                frame, 
                (x_offset, y), 
                (x_offset + bar_width, y + bar_height), 
                (50, 50, 50), 
                -1
            )
            
            # Probability bar
            bar_length = int(bar_width * prob)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(
                frame, 
                (x_offset, y), 
                (x_offset + bar_length, y + bar_height), 
                color, 
                -1
            )
            
            # Text label
            text = f"{emotion}: {prob*100:.1f}%"
            cv2.putText(
                frame, 
                text, 
                (x_offset + 5, y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
    
    def run(self):
        """Run real-time detection"""
        print("\nStarting webcam...")
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect emotions
            results = self.detect_emotion(frame)
            
            # Draw results
            frame = self.draw_results(frame, results)
            
            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            cv2.putText(
                frame, 
                f"FPS: {fps}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Display instructions
            cv2.putText(
                frame, 
                "Press 'q' to quit, 's' to save", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # Show frame
            cv2.imshow('Facial Expression Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam stopped")

if __name__ == "__main__":
    print("=" * 60)
    print("Real-Time Facial Expression Recognition")
    print("=" * 60)
    
    detector = FacialExpressionDetector('best_model.keras')
    detector.run()
