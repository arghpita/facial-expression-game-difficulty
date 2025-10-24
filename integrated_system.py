import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time
from difficulty_controller import DifficultyController

class IntegratedExpressionSystem:
    def __init__(self, model_path='best_model.keras'):
        """Initialize integrated system with expression detection and difficulty control"""
        print("Initializing Integrated Expression Recognition System...")
        print("-" * 60)
        
        # Load model
        print("Loading model...")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded")
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ Face detector loaded")
        
        # Initialize difficulty controller
        self.difficulty_controller = DifficultyController(
            min_difficulty=1,
            max_difficulty=10,
            adjustment_rate=0.15,
            smoothing_window=30
        )
        print("✓ Difficulty controller initialized")
        
        # Emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Smoothing buffer
        self.prediction_buffer = deque(maxlen=10)
        
        # Colors
        self.emotion_colors = {
            'Angry': (0, 0, 255),
            'Disgust': (0, 128, 0),
            'Fear': (128, 0, 128),
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Surprise': (0, 255, 255),
            'Neutral': (255, 255, 255)
        }
        
        # Difficulty colors
        self.difficulty_colors = {
            "Very Easy": (0, 255, 0),      # Green
            "Easy": (0, 255, 128),         # Light Green
            "Medium": (0, 255, 255),       # Yellow
            "Hard": (0, 165, 255),         # Orange
            "Very Hard": (0, 0, 255)       # Red
        }
        
        print("✓ System ready!")
        print("-" * 60)
    
    def preprocess_face(self, face_img):
        """Preprocess face for model prediction"""
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img / 255.0
        face_img = face_img.reshape(1, 48, 48, 1)
        return face_img
    
    def get_smoothed_prediction(self, predictions):
        """Smooth predictions over time"""
        self.prediction_buffer.append(predictions)
        if len(self.prediction_buffer) > 0:
            return np.mean(self.prediction_buffer, axis=0)
        return predictions
    
    def detect_emotion(self, frame):
        """Detect faces and emotions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            processed_face = self.preprocess_face(face_roi)
            predictions = self.model.predict(processed_face, verbose=0)[0]
            smoothed_predictions = self.get_smoothed_prediction(predictions)
            
            emotion_idx = np.argmax(smoothed_predictions)
            emotion = self.emotion_labels[emotion_idx]
            confidence = smoothed_predictions[emotion_idx]
            
            # Update difficulty controller
            difficulty_info = self.difficulty_controller.update(emotion, confidence)
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': smoothed_predictions,
                'difficulty_info': difficulty_info
            })
        
        return results
    
    def draw_dashboard(self, frame, results):
        """Draw comprehensive dashboard"""
        h, w = frame.shape[:2]
        
        # Draw face detection results
        for result in results:
            x, y, w_box, h_box = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']
            
            # Face rectangle
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 3)
            
            # Emotion label
            label = f"{emotion}: {confidence*100:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw side panel
        self.draw_side_panel(frame, results)
        
        # Draw difficulty panel
        self.draw_difficulty_panel(frame, results)
        
        return frame
    
    def draw_side_panel(self, frame, results):
        """Draw emotion probability bars"""
        if len(results) == 0:
            return
        
        h, w = frame.shape[:2]
        predictions = results[0]['all_predictions']
        
        panel_width = 280
        panel_x = w - panel_width - 10
        bar_height = 35
        y_start = 10
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 5, y_start - 5), 
                     (w - 5, y_start + len(self.emotion_labels) * (bar_height + 5) + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Title
        cv2.putText(frame, "EMOTION ANALYSIS", (panel_x, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_start += 35
        
        # Emotion bars
        for i, (emotion, prob) in enumerate(zip(self.emotion_labels, predictions)):
            y = y_start + i * (bar_height + 5)
            bar_width = 200
            
            # Background
            cv2.rectangle(frame, (panel_x, y), (panel_x + bar_width, y + bar_height),
                         (50, 50, 50), -1)
            
            # Probability bar
            bar_length = int(bar_width * prob)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (panel_x, y), (panel_x + bar_length, y + bar_height),
                         color, -1)
            
            # Border
            cv2.rectangle(frame, (panel_x, y), (panel_x + bar_width, y + bar_height),
                         (100, 100, 100), 1)
            
            # Text
            text = f"{emotion}: {prob*100:.1f}%"
            cv2.putText(frame, text, (panel_x + 5, y + 23),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_difficulty_panel(self, frame, results):
        """Draw difficulty adjustment panel"""
        h, w = frame.shape[:2]
        
        if len(results) == 0:
            return
        
        difficulty_info = results[0]['difficulty_info']
        
        # Panel dimensions
        panel_height = 200
        panel_width = w - 20
        panel_x = 10
        panel_y = h - panel_height - 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Title
        cv2.putText(frame, "DIFFICULTY CONTROLLER", 
                   (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current difficulty
        difficulty = difficulty_info['difficulty']
        difficulty_level = difficulty_info['difficulty_level']
        difficulty_color = self.difficulty_colors.get(difficulty_level, (255, 255, 255))
        
        # Difficulty bar
        bar_y = panel_y + 50
        bar_width = panel_width - 40
        bar_height = 30
        
        cv2.rectangle(frame, (panel_x + 20, bar_y),
                     (panel_x + 20 + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Fill based on difficulty
        fill_width = int((difficulty / 10) * bar_width)
        cv2.rectangle(frame, (panel_x + 20, bar_y),
                     (panel_x + 20 + fill_width, bar_y + bar_height),
                     difficulty_color, -1)
        
        # Border
        cv2.rectangle(frame, (panel_x + 20, bar_y),
                     (panel_x + 20 + bar_width, bar_y + bar_height),
                     (200, 200, 200), 2)
        
        # Difficulty text
        diff_text = f"{difficulty:.1f}/10 - {difficulty_level}"
        cv2.putText(frame, diff_text,
                   (panel_x + 20 + bar_width // 2 - 80, bar_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Recommendation
        recommendation = difficulty_info['recommendation']
        
        # Wrap text if too long
        max_width = 80
        words = recommendation.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > max_width:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        y_text = bar_y + 55
        for line in lines:
            cv2.putText(frame, line,
                       (panel_x + 20, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_text += 25
        
        # Stats
        stats = self.difficulty_controller.get_stats()
        emotion_dist = stats.get('emotion_distribution', {})
        
        if emotion_dist:
            stats_y = bar_y + 120
            cv2.putText(frame, "Recent Emotions:",
                       (panel_x + 20, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            top_emotions = sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            stats_text = " | ".join([f"{em}: {pct:.0f}%" for em, pct in top_emotions])
            cv2.putText(frame, stats_text,
                       (panel_x + 20, stats_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    def run(self):
        """Run the integrated system"""
        print("\nStarting Integrated Expression Recognition System")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Reset difficulty")
        print("-" * 60)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect emotions and update difficulty
            results = self.detect_emotion(frame)
            
            # Draw dashboard
            frame = self.draw_dashboard(frame, results)
            
            # FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            cv2.putText(frame, f"FPS: {fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show
            cv2.imshow('Expression Recognition & Difficulty Control', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                self.difficulty_controller.reset()
                print("Difficulty reset to medium")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nSystem stopped")

if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATED EXPRESSION RECOGNITION SYSTEM")
    print("=" * 60)
    
    system = IntegratedExpressionSystem('best_model.keras')
    system.run()
