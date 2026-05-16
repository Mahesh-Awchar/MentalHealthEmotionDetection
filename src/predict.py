import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from MentalHealthEmotionDetection.src.utils import logger

class EmotionPredictor:
    """
    Handles loading the trained model and making real-time emotion predictions.
    """
    def __init__(self, model_path="models/roberta_goemotions_model", tokenizer_path="roberta-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(model_path):
            logger.error(f"Model path {model_path} does not exist. Please train the model first.")
            raise FileNotFoundError(f"Model directory {model_path} not found.")
            
        logger.info(f"Loading model from {model_path} and tokenizer from {tokenizer_path} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_emotion(self, text, threshold=0.3):
        """
        Predicts emotions for a given text.
        Returns a sorted dictionary of emotions and their confidence scores.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Map probabilities to labels
        id2label = self.model.config.id2label
        results = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        
        # Sort results by confidence
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        
        # Filter by threshold for active emotions
        active_emotions = {k: v for k, v in sorted_results.items() if v >= threshold}
        
        return sorted_results, active_emotions

    def generate_mental_health_insight(self, active_emotions):
        """
        Provides mental health insights based on the detected dominant emotions.
        """
        if not active_emotions:
            return "No strong emotions detected. You seem to be in a neutral state."
            
        insights = []
        emotions = list(active_emotions.keys())
        
        # Risk of Depression/Sadness
        if any(e in emotions for e in ['sadness', 'grief', 'disappointment', 'loneliness', 'remorse']):
            insights.append("Your text indicates feelings of sadness or grief. If these feelings persist and interfere with your daily life, it might be helpful to reach out to a mental health professional.")
            
        # Risk of Anxiety/Stress
        if any(e in emotions for e in ['fear', 'nervousness', 'stress']):
            insights.append("Signs of anxiety or fear are present. Deep breathing exercises, mindfulness, or talking to someone you trust can help manage these feelings.")
            
        # Anger issues
        if any(e in emotions for e in ['anger', 'annoyance', 'disapproval']):
            insights.append("You seem to be experiencing frustration or anger. Taking a step back and practicing relaxation techniques might help you process these emotions constructively.")
            
        # Positive indicators
        if any(e in emotions for e in ['joy', 'optimism', 'caring', 'gratitude', 'love']):
            insights.append("Positive emotions detected! Acknowledging and savoring these moments contributes significantly to overall mental well-being.")
            
        if not insights:
            top_emotion = emotions[0]
            insights.append(f"Your primary detected emotion is {top_emotion}. Acknowledging your emotional state is a great first step in emotional awareness.")
            
        return " ".join(insights)
