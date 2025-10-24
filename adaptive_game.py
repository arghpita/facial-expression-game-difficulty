import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time
import random
from difficulty_controller import DifficultyController

class Target:
    """Represents a clickable target in the game"""
    def __init__(self, x, y, radius, speed, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.color = color
        self.lifetime = 0
        self.max_lifetime = 3.0  # seconds before target expires
        
    def update(self, dt):
        """Update target state"""
        self.lifetime += dt
        # Pulse effect
        pulse = 1.0 + 0.2 * np.sin(self.lifetime * 5)
        return pulse
        
    def is_expired(self):
        """Check if target has expired"""
        return self.lifetime >= self.max_lifetime
        
    def is_clicked(self, mouse_x, mouse_y):
        """Check if target was clicked"""
        distance = np.sqrt((self.x - mouse_x)**2 + (self.y - mouse_y)**2)
        return distance <= self.radius

class AdaptiveReactionGame:
    """Reaction game with adaptive difficulty based on facial expressions"""
    
    def __init__(self, model_path='best_model.keras'):
        print("Initializing Adaptive Reaction Game...")
        print("-" * 60)
        
        # Load emotion recognition model
        print("Loading emotion model...")
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
            adjustment_rate=0.2,
            smoothing_window=20
        )
        print("✓ Difficulty controller initialized")
        
        # Emotion recognition setup
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.prediction_buffer = deque(maxlen=8)
        
        # Game state
        self.score = 0
        self.missed = 0
        self.targets = []
        self.spawn_timer = 0
        self.game_time = 0
        self.paused = False
        self.game_over = False
        
        # Difficulty parameters (adjusted by emotion)
        self.base_spawn_rate = 2.0  # seconds between spawns at medium difficulty
        self.base_target_speed = 1.0
        self.base_target_size = 40
        self.max_targets = 3
        
        # Game dimensions
        self.game_width = 800
        self.game_height = 600
        self.webcam_width = 320
        self.webcam_height = 240
        
        # Colors
        self.target_colors = [
            (0, 255, 0),    # Green
            (0, 255, 255),  # Yellow
            (255, 128, 0),  # Orange
            (255, 0, 255),  # Magenta
        ]
        
        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0
        self.click_effect = []
        
        print("✓ Game initialized!")
        print("-" * 60)
        
    def preprocess_face(self, face_img):
        """Preprocess face for emotion recognition"""
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img / 255.0
        face_img = face_img.reshape(1, 48, 48, 1)
        return face_img
    
    def detect_emotion(self, frame):
        """Detect emotion from webcam frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        
        if len(faces) == 0:
            return None, None, None
        
        # Use first detected face
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        processed_face = self.preprocess_face(face_roi)
        predictions = self.model.predict(processed_face, verbose=0)[0]
        
        # Smooth predictions
        self.prediction_buffer.append(predictions)
        smoothed = np.mean(self.prediction_buffer, axis=0)
        
        emotion_idx = np.argmax(smoothed)
        emotion = self.emotion_labels[emotion_idx]
        confidence = smoothed[emotion_idx]
        
        return emotion, confidence, (x, y, w, h)
    
    def get_difficulty_params(self, difficulty):
        """Calculate game parameters based on difficulty level"""
        # Normalize difficulty to 0-1 range
        norm_diff = (difficulty - 1) / 9
        
        # Spawn rate: easier = slower spawning
        spawn_rate = self.base_spawn_rate * (2.0 - norm_diff)  # 2.0s to 1.0s
        
        # Target speed: easier = slower movement
        speed = self.base_target_speed * (0.5 + norm_diff)  # 0.5x to 1.5x
        
        # Target size: easier = larger targets
        size = int(self.base_target_size * (1.5 - 0.5 * norm_diff))  # 60px to 40px
        
        # Max targets: easier = fewer targets
        max_targets = max(1, min(5, int(2 + norm_diff * 3)))  # 2 to 5
        
        return spawn_rate, speed, size, max_targets
    
    def spawn_target(self, size):
        """Spawn a new target"""
        margin = size + 10
        x = random.randint(margin, self.game_width - margin)
        y = random.randint(margin + 100, self.game_height - margin - 150)
        
        color = random.choice(self.target_colors)
        speed = 0  # Stationary targets
        
        target = Target(x, y, size, speed, color)
        self.targets.append(target)
    
    def update_game(self, dt, difficulty):
        """Update game state"""
        if self.paused or self.game_over:
            return
        
        self.game_time += dt
        
        # Get difficulty parameters
        spawn_rate, speed, size, max_targets = self.get_difficulty_params(difficulty)
        self.max_targets = max_targets
        
        # Spawn new targets
        self.spawn_timer += dt
        if self.spawn_timer >= spawn_rate and len(self.targets) < max_targets:
            self.spawn_target(size)
            self.spawn_timer = 0
        
        # Update targets
        expired_targets = []
        for target in self.targets:
            target.update(dt)
            if target.is_expired():
                expired_targets.append(target)
                self.missed += 1
        
        # Remove expired targets
        for target in expired_targets:
            self.targets.remove(target)
        
        # Update click effects
        self.click_effect = [(x, y, t - dt) for x, y, t in self.click_effect if t > dt]
        
        # Game over condition
        if self.missed >= 10:
            self.game_over = True
    
    def handle_click(self, x, y):
        """Handle mouse click"""
        if self.game_over or self.paused:
            return
        
        # Check if any target was clicked
        for target in self.targets[:]:
            if target.is_clicked(x, y):
                self.score += 10
                self.targets.remove(target)
                self.click_effect.append((x, y, 0.3))
                return True
        
        return False
    
    def draw_game(self, frame):
        """Draw game elements"""
        h, w = frame.shape[:2]
        
        # Draw background
        cv2.rectangle(frame, (0, 0), (self.game_width, self.game_height), (20, 20, 20), -1)
        
        # Draw targets
        for target in self.targets:
            pulse = target.update(0)
            radius = int(target.radius * pulse)
            
            # Target circle
            cv2.circle(frame, (target.x, target.y), radius, target.color, -1)
            cv2.circle(frame, (target.x, target.y), radius, (255, 255, 255), 2)
            
            # Lifetime indicator (shrinking circle)
            lifetime_pct = 1.0 - (target.lifetime / target.max_lifetime)
            inner_radius = int(radius * 0.7 * lifetime_pct)
            if inner_radius > 0:
                cv2.circle(frame, (target.x, target.y), inner_radius, (255, 255, 255), 2)
        
        # Draw click effects
        for cx, cy, time_left in self.click_effect:
            alpha = time_left / 0.3
            radius = int(30 * (1 - alpha))
            cv2.circle(frame, (int(cx), int(cy)), radius, (255, 255, 255), 2)
        
        # Draw header
        header_height = 80
        cv2.rectangle(frame, (0, 0), (self.game_width, header_height), (40, 40, 40), -1)
        
        # Score and stats
        cv2.putText(frame, f"SCORE: {self.score}", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"MISSED: {self.missed}/10", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        cv2.putText(frame, f"TIME: {int(self.game_time)}s", (300, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"ACTIVE: {len(self.targets)}/{self.max_targets}", (300, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw footer with instructions
        footer_y = self.game_height - 120
        cv2.rectangle(frame, (0, footer_y), (self.game_width, self.game_height), (40, 40, 40), -1)
        
        if self.game_over:
            cv2.putText(frame, "GAME OVER!", (self.game_width // 2 - 150, self.game_height // 2),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"Final Score: {self.score}", (self.game_width // 2 - 120, self.game_height // 2 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'R' to restart or 'Q' to quit", (self.game_width // 2 - 200, self.game_height // 2 + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        elif self.paused:
            cv2.putText(frame, "PAUSED", (self.game_width // 2 - 80, self.game_height // 2),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)
            cv2.putText(frame, "Press 'P' to continue", (self.game_width // 2 - 130, self.game_height // 2 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        else:
            cv2.putText(frame, "Click the targets before they disappear!", (20, footer_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Controls: P=Pause | R=Restart | Q=Quit", (20, footer_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def draw_emotion_panel(self, frame, emotion, confidence, difficulty_info, face_bbox):
        """Draw emotion and difficulty info panel"""
        panel_x = self.game_width + 10
        panel_y = 10
        panel_width = self.webcam_width
        
        # Webcam placeholder
        webcam_h = self.webcam_height
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + webcam_h),
                     (50, 50, 50), 2)
        
        # Emotion info
        info_y = panel_y + webcam_h + 20
        
        if emotion:
            cv2.putText(frame, f"Emotion: {emotion}", (panel_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (panel_x, info_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "No face detected", (panel_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        # Difficulty info
        if difficulty_info:
            diff_y = info_y + 60
            difficulty = difficulty_info['difficulty']
            level = difficulty_info['difficulty_level']
            
            cv2.putText(frame, "DIFFICULTY", (panel_x, diff_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
            
            # Difficulty bar
            bar_width = panel_width - 20
            bar_height = 25
            bar_y = diff_y + 10
            
            cv2.rectangle(frame, (panel_x, bar_y), 
                         (panel_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            fill_width = int((difficulty / 10) * bar_width)
            
            # Color based on difficulty
            if difficulty <= 3:
                color = (0, 255, 0)  # Green
            elif difficulty <= 6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 100, 255)  # Orange-Red
            
            cv2.rectangle(frame, (panel_x, bar_y),
                         (panel_x + fill_width, bar_y + bar_height),
                         color, -1)
            
            cv2.rectangle(frame, (panel_x, bar_y),
                         (panel_x + bar_width, bar_y + bar_height),
                         (200, 200, 200), 2)
            
            cv2.putText(frame, f"{difficulty:.1f} - {level}", (panel_x, bar_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def reset_game(self):
        """Reset game state"""
        self.score = 0
        self.missed = 0
        self.targets = []
        self.spawn_timer = 0
        self.game_time = 0
        self.game_over = False
        self.paused = False
        self.difficulty_controller.reset()
        print("Game reset!")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < self.game_width and y < self.game_height:
                self.handle_click(x, y)
    
    def run(self):
        """Run the game"""
        print("\nStarting Adaptive Reaction Game")
        print("The game difficulty adjusts based on your facial expressions!")
        print("-" * 60)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Create window
        window_name = 'Adaptive Reaction Game'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Create canvas
        canvas_width = self.game_width + self.webcam_width + 20
        canvas_height = self.game_height
        
        last_time = time.time()
        
        while True:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Read webcam
            ret, webcam_frame = cap.read()
            if not ret:
                break
            
            webcam_frame = cv2.flip(webcam_frame, 1)
            
            # Detect emotion
            emotion, confidence, face_bbox = self.detect_emotion(webcam_frame)
            
            # Update difficulty
            difficulty_info = None
            if emotion and confidence:
                difficulty_info = self.difficulty_controller.update(emotion, confidence)
                difficulty = difficulty_info['difficulty']
            else:
                difficulty = 5.0  # Default medium difficulty
            
            # Update game
            self.update_game(dt, difficulty)
            
            # Create canvas
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            # Draw game
            self.draw_game(canvas)
            
            # Draw webcam feed
            small_webcam = cv2.resize(webcam_frame, (self.webcam_width, self.webcam_height))
            if face_bbox:
                x, y, w, h = face_bbox
                # Scale to small webcam
                scale_x = self.webcam_width / webcam_frame.shape[1]
                scale_y = self.webcam_height / webcam_frame.shape[0]
                sx, sy = int(x * scale_x), int(y * scale_y)
                sw, sh = int(w * scale_x), int(h * scale_y)
                cv2.rectangle(small_webcam, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            
            canvas[10:10+self.webcam_height, self.game_width+10:self.game_width+10+self.webcam_width] = small_webcam
            
            # Draw emotion panel
            self.draw_emotion_panel(canvas, emotion, confidence, difficulty_info, face_bbox)
            
            # Show
            cv2.imshow(window_name, canvas)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
                print("Paused" if self.paused else "Resumed")
            elif key == ord('r'):
                self.reset_game()
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nGame ended!")
        print(f"Final Score: {self.score}")
        print(f"Total Time: {int(self.game_time)}s")

if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE REACTION GAME")
    print("=" * 60)
    
    game = AdaptiveReactionGame('best_model.keras')
    game.run()
