# Mental Health Emotion Detection Using Textual Data

## Project Overview
This project builds a fully functional end-to-end transformer NLP system that detects emotions from textual data, focusing on mental health insights. It uses the `RoBERTa` transformer model fine-tuned on the `GoEmotions` dataset to perform multi-label emotion classification (predicting multiple emotions simultaneously). 

The application includes a training pipeline using the Hugging Face Trainer API and a modern web interface built with Streamlit for real-time predictions and mental health insights.

## Architecture
1. **Data Pipeline**: Uses Hugging Face `datasets` to download and preprocess the GoEmotions dataset. Applies tokenization and multi-hot encoding for the 28 emotion labels.
2. **Model**: Fine-tunes `roberta-base` using PyTorch and Hugging Face `transformers`.
3. **Training**: Uses `Trainer` API with `BCEWithLogitsLoss` customized for multi-label classification. Supports GPU acceleration (CUDA) and mixed precision (fp16).
4. **Inference & UI**: A Streamlit web application that takes user text, predicts the top emotions, visualizes confidence scores using Plotly/Matplotlib, and maps these emotions to actionable mental health insights.

## Project Structure
```text
MentalHealthEmotionDetection/
│
├── dataset/             # (Created upon execution) Local dataset cache
├── models/              # (Created upon execution) Saved fine-tuned models
├── notebooks/           # Jupyter notebooks for experimentation
├── outputs/             # Training logs, plots, and evaluation results
├── app/                 # Streamlit web application
│   └── app.py
├── src/                 # Core Python source code
│   ├── dataset.py       # Data loading and preprocessing
│   ├── model.py         # Model configuration and loading
│   ├── train.py         # Training pipeline
│   ├── predict.py       # Inference and insight generation
│   └── utils.py         # Logging, metrics, and visualization
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── main.py              # CLI entry point for training and testing
```

## Installation Steps

1. **Clone the repository / Navigate to the folder:**
   ```bash
   cd MentalHealthEmotionDetection
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   
   # On Windows (VS Code Command Prompt/PowerShell):
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *Note: If you have a CUDA-enabled GPU, you might need to install the CUDA-specific version of PyTorch. Check [pytorch.org](https://pytorch.org/) for the specific command.*

## Usage Instructions

### 1. Training the Model
To train the model from scratch on the GoEmotions dataset:
```bash
python main.py --mode train
```
*This will download the dataset, tokenize it, train the RoBERTa model, and save the final model inside the `models/` directory. It also saves training graphs in `outputs/`.*

### 2. Testing via CLI
To quickly test the model in the terminal (ensure you have trained or downloaded the model first):
```bash
python main.py --mode predict --text "I feel so overwhelmed and anxious about the future."
```

### 3. Running the Streamlit App
To launch the interactive web application:
```bash
streamlit run app/app.py
```
This will open a local web server (typically `http://localhost:8501`) where you can enter text, see the emotion predictions, view confidence graphs, and read mental health insights.

## Technologies Used
* **Python 3.8+**
* **PyTorch** (Deep Learning framework)
* **Hugging Face Transformers & Datasets** (NLP tools)
* **RoBERTa** (Pre-trained Transformer)
* **Streamlit** (Web UI)
* **Scikit-learn, Numpy, Pandas** (Metrics and Data handling)
* **Plotly, Matplotlib, Seaborn** (Data Visualization)

## Future Scope
* Incorporating more specific clinical datasets for depression and anxiety detection.
* Fine-tuning larger models (e.g., RoBERTa-large, LLaMA) for better context understanding.
* Adding a database (like SQLite/PostgreSQL) to track user emotional trends over time.
* Deploying the Streamlit app to AWS, Hugging Face Spaces, or Streamlit Cloud.
