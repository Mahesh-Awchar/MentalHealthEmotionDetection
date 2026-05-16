import os
import torch
from transformers import Trainer, TrainingArguments
from MentalHealthEmotionDetection.src.utils import logger, compute_metrics, plot_training_history
from MentalHealthEmotionDetection.src.dataset import EmotionDatasetManager
from MentalHealthEmotionDetection.src.model import get_model

def train_model(output_dir="models", epochs=1, batch_size=16):
    """
    Full training pipeline: loads data, model, and runs the CustomTrainer.
    """
    # 1. Prepare data
    dataset_manager = EmotionDatasetManager()
    tokenized_datasets = dataset_manager.get_tokenized_datasets()
    labels = dataset_manager.labels
    
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    
    # 2. Prepare model
    model = get_model(num_labels=dataset_manager.num_labels, id2label=id2label, label2id=label2id)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 3. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(), # Enable mixed precision if GPU is available
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="outputs/logs",
        logging_steps=10,
    )
    
    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=dataset_manager.tokenizer,
        compute_metrics=compute_metrics
    )
    
    # 5. Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # 6. Save final model and tokenizer
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    # 7. Evaluate on test set
    logger.info("Evaluating on test dataset...")
    test_metrics = trainer.evaluate(tokenized_datasets["test"])
    logger.info(f"Test metrics: {test_metrics}")
    
    # 8. Plot training history
    plot_training_history(trainer.state.log_history, output_dir="outputs")
    
    logger.info("Training complete!")
