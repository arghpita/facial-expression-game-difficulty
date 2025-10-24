import numpy as np
from collections import deque
import time

class DifficultyController:
    """
    Controls game difficulty based on facial expressions.
    
    Difficulty Mapping Logic:
    - Frustrated/Angry/Fear → Decrease difficulty
    - Happy/Neutral → Maintain or slightly increase difficulty
    - Surprise → Slight increase (engagement indicator)
    - Disgust/Sad → Decrease difficulty
    """
    
    def __init__(self, 
                 min_difficulty=1, 
                 max_difficulty=10,
                 adjustment_rate=0.1,
                 smoothing_window=30):
        """
        Initialize difficulty controller
        
        Args:
            min_difficulty: Minimum difficulty level (default: 1)
            max_difficulty: Maximum difficulty level (default: 10)
            adjustment_rate: How fast difficulty changes (default: 0.1)
            smoothing_window: Number of frames to smooth over (default: 30)
        """
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = (min_difficulty + max_difficulty) / 2  # Start at medium
        self.adjustment_rate = adjustment_rate
        
        # Emotion history buffer
        self.emotion_history = deque(maxlen=smoothing_window)
        
        # Emotion weights for difficulty adjustment
        # Positive = increase difficulty, Negative = decrease difficulty
        self.emotion_weights = {
            'Angry': -0.8,      # Strong decrease
            'Disgust': -0.5,    # Moderate decrease
            'Fear': -0.7,       # Strong decrease
            'Happy': 0.3,       # Slight increase
            'Sad': -0.6,        # Moderate decrease
            'Surprise': 0.2,    # Slight increase (engagement)
            'Neutral': 0.1      # Maintain/slight increase
        }
        
        # Performance tracking
        self.adjustment_history = []
        self.last_adjustment_time = time.time()
        
    def update(self, emotion, confidence):
        """
        Update difficulty based on detected emotion
        
        Args:
            emotion: Detected emotion string
            confidence: Confidence score (0-1)
            
        Returns:
            dict: Contains current difficulty and adjustment info
        """
        # Add to history
        self.emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Calculate adjustment
        adjustment = self._calculate_adjustment()
        
        # Apply adjustment
        old_difficulty = self.current_difficulty
        self.current_difficulty += adjustment
        
        # Clamp to min/max
        self.current_difficulty = np.clip(
            self.current_difficulty, 
            self.min_difficulty, 
            self.max_difficulty
        )
        
        # Track adjustment
        self.adjustment_history.append({
            'timestamp': time.time(),
            'emotion': emotion,
            'confidence': confidence,
            'old_difficulty': old_difficulty,
            'new_difficulty': self.current_difficulty,
            'adjustment': adjustment
        })
        
        return {
            'difficulty': self.current_difficulty,
            'difficulty_level': self.get_difficulty_level(),
            'adjustment': adjustment,
            'emotion': emotion,
            'confidence': confidence,
            'recommendation': self._get_recommendation()
        }
    
    def _calculate_adjustment(self):
        """Calculate difficulty adjustment based on emotion history"""
        if len(self.emotion_history) == 0:
            return 0
        
        # Weight recent emotions more heavily
        total_weight = 0
        weighted_sum = 0
        
        for i, record in enumerate(self.emotion_history):
            # Recency weight (more recent = higher weight)
            recency_weight = (i + 1) / len(self.emotion_history)
            
            # Emotion weight
            emotion = record['emotion']
            confidence = record['confidence']
            emotion_weight = self.emotion_weights.get(emotion, 0)
            
            # Combined weight
            weight = recency_weight * confidence
            weighted_sum += emotion_weight * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        # Calculate adjustment
        avg_weight = weighted_sum / total_weight
        adjustment = avg_weight * self.adjustment_rate
        
        return adjustment
    
    def get_difficulty_level(self):
        """Get difficulty as categorical level"""
        if self.current_difficulty <= 2:
            return "Very Easy"
        elif self.current_difficulty <= 4:
            return "Easy"
        elif self.current_difficulty <= 6:
            return "Medium"
        elif self.current_difficulty <= 8:
            return "Hard"
        else:
            return "Very Hard"
    
    def _get_recommendation(self):
        """Get recommendation for game adjustment"""
        if len(self.emotion_history) == 0:
            return "Monitoring player state..."
        
        recent_emotions = [e['emotion'] for e in list(self.emotion_history)[-10:]]
        
        # Count negative emotions
        negative_emotions = ['Angry', 'Fear', 'Sad', 'Disgust']
        negative_count = sum(1 for e in recent_emotions if e in negative_emotions)
        
        if negative_count >= 6:
            return "Player showing frustration - reduce difficulty"
        elif negative_count >= 3:
            return "Player may be struggling - consider reducing difficulty"
        elif 'Happy' in recent_emotions[-3:]:
            return "Player engaged and performing well"
        else:
            return "Player state neutral - maintain current difficulty"
    
    def get_emotion_distribution(self):
        """Get distribution of emotions over history"""
        if len(self.emotion_history) == 0:
            return {}
        
        emotion_counts = {}
        for record in self.emotion_history:
            emotion = record['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Convert to percentages
        total = len(self.emotion_history)
        emotion_percentages = {
            emotion: (count / total) * 100 
            for emotion, count in emotion_counts.items()
        }
        
        return emotion_percentages
    
    def reset(self):
        """Reset difficulty to default"""
        self.current_difficulty = (self.min_difficulty + self.max_difficulty) / 2
        self.emotion_history.clear()
        self.adjustment_history.clear()
        print("Difficulty controller reset")
    
    def get_stats(self):
        """Get statistics about difficulty adjustments"""
        if len(self.adjustment_history) == 0:
            return "No adjustments yet"
        
        stats = {
            'current_difficulty': self.current_difficulty,
            'difficulty_level': self.get_difficulty_level(),
            'total_adjustments': len(self.adjustment_history),
            'emotion_distribution': self.get_emotion_distribution(),
            'recommendation': self._get_recommendation()
        }
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Difficulty Controller Test")
    print("=" * 60)
    
    controller = DifficultyController()
    
    # Simulate emotion sequence
    test_emotions = [
        ('Neutral', 0.8),
        ('Happy', 0.9),
        ('Happy', 0.85),
        ('Angry', 0.7),
        ('Angry', 0.8),
        ('Fear', 0.75),
        ('Sad', 0.7),
        ('Neutral', 0.8),
        ('Happy', 0.9),
    ]
    
    print("\nSimulating emotion sequence...")
    print("-" * 60)
    
    for emotion, confidence in test_emotions:
        result = controller.update(emotion, confidence)
        print(f"\nEmotion: {emotion} ({confidence*100:.1f}% confidence)")
        print(f"Difficulty: {result['difficulty']:.2f} ({result['difficulty_level']})")
        print(f"Adjustment: {result['adjustment']:+.3f}")
        print(f"Recommendation: {result['recommendation']}")
    
    print("\n" + "=" * 60)
    print("Final Statistics:")
    print("=" * 60)
    stats = controller.get_stats()
    print(f"Current Difficulty: {stats['current_difficulty']:.2f}")
    print(f"Difficulty Level: {stats['difficulty_level']}")
    print(f"Total Adjustments: {stats['total_adjustments']}")
    print("\nEmotion Distribution:")
    for emotion, percentage in stats['emotion_distribution'].items():
        print(f"  {emotion}: {percentage:.1f}%")
    print(f"\nRecommendation: {stats['recommendation']}")
