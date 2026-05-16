import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import os
from MentalHealthEmotionDetection.src.utils import logger

class EmotionDatasetManager:
    """
    Manager for loading and preprocessing the GoEmotions dataset.
    Optimized for speed with dynamic padding.
    """
    def __init__(self, model_name="roberta-base", max_length=64): # Reduced max_length for speed
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = None
        self.labels = None
        self.num_labels = 0
        
    def load_data(self):
        """
        Loads the GoEmotions dataset from Hugging Face.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        dataset_cache_dir = os.path.join(project_root, "dataset")
        
        logger.info(f"Loading GoEmotions dataset into {dataset_cache_dir}...")
        self.dataset = load_dataset("go_emotions", "simplified", cache_dir=dataset_cache_dir)
        
        self.labels = self.dataset['train'].features['labels'].feature.names
        self.num_labels = len(self.labels)
        logger.info(f"Loaded dataset with {self.num_labels} labels.")
        
        return self.dataset
    
    def preprocess_function(self, examples):
        """
        Tokenizes the text and converts labels to multi-hot encoding.
        Dynamic padding is used (padding=False here, DataCollator handles it later).
        """
        # Tokenize the text without padding, truncating to max_length
        batch = self.tokenizer(
            examples["text"], 
            padding=False, # Dynamic padding later 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Convert list of label indices to multi-hot encoded vectors
        batch_labels = []
        for label_indices in examples["labels"]:
            multi_hot = np.zeros(self.num_labels, dtype=np.float32)
            for idx in label_indices:
                multi_hot[idx] = 1.0
            batch_labels.append(multi_hot)
            
        batch["labels"] = batch_labels
        return batch

    def get_tokenized_datasets(self):
        """
        Applies preprocessing to all splits and formats them for PyTorch.
        """
        if self.dataset is None:
            self.load_data()
            
        logger.info("Tokenizing and formatting dataset...")
        tokenized_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        
        tokenized_datasets.set_format("torch")
        logger.info("Dataset formatting complete.")
        
        return tokenized_datasets
