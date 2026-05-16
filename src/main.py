import argparse
from MentalHealthEmotionDetection.src.train import train_model
from MentalHealthEmotionDetection.src.predict import EmotionPredictor
from MentalHealthEmotionDetection.src.utils import logger

def main():
    parser = argparse.ArgumentParser(description="Mental Health Emotion Detection CLI")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"],
                        help="Mode to run the script: 'train' to train the model, 'predict' to test inference.")
    parser.add_argument("--text", type=str, default="I am feeling a bit stressed and anxious today.",
                        help="Text to analyze (used only in 'predict' mode).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        logger.info("--- Starting Training Pipeline ---")
        train_model(epochs=args.epochs, batch_size=args.batch_size)
        
    elif args.mode == "predict":
        logger.info("--- Starting Prediction Pipeline ---")
        try:
            predictor = EmotionPredictor()
            all_emotions, active_emotions = predictor.predict_emotion(args.text)
            
            print(f"\n[Input Text]: {args.text}\n")
            print("[Top Detected Emotions]:")
            for emotion, score in list(all_emotions.items())[:5]:
                print(f"  - {emotion.capitalize()}: {score:.4f}")
                
            print(f"\n[Mental Health Insight]:")
            insight = predictor.generate_mental_health_insight(active_emotions)
            print(f"  {insight}\n")
            
        except FileNotFoundError:
            print("Error: Trained model not found. Please run 'python main.py --mode train' first.")

if __name__ == "__main__":
    main()
