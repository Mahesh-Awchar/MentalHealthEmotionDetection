import logging
import os
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a configured logger for the project.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger('emotion_detection', 'logs/project.log')

def compute_metrics(eval_pred):
    """
    Compute metrics for multi-label classification.
    """
    logits, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    # Note: logits might already be sigmoid-ed depending on how Trainer handles it,
    # but typically they are raw logits. We use a threshold of 0.5.
    probs = expit(logits)
    predictions = (probs >= 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def plot_training_history(history, output_dir="outputs"):
    """
    Plot training and validation loss/metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from history (which is typically trainer.state.log_history)
    train_loss = [x['loss'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    steps = [x['step'] for x in history if 'loss' in x]
    eval_steps = [x['step'] for x in history if 'eval_loss' in x]

    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(steps, train_loss, label='Training Loss')
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label='Validation Loss')
        
    plt.title('Training and Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    logger.info(f"Saved training history plot to {output_dir}/training_loss.png")
