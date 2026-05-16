import torch
from transformers import AutoModelForSequenceClassification
from MentalHealthEmotionDetection.src.utils import logger

def get_model(model_name="roberta-base", num_labels=28, id2label=None, label2id=None):
    """
    Loads and configures the RoBERTa model for multi-label sequence classification.
    """
    logger.info(f"Loading {model_name} model for multi-label classification with {num_labels} labels...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    return model
