import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px
import pandas as pd
import os

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    # Safely resolve model path relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "roberta_goemotions_model")
    
    if not os.path.exists(model_path):
        st.error(f"Model directory not found at {model_path}. Please ensure the model is downloaded.")
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Move model to CPU explicitly if running on Streamlit Cloud (no GPU)
    model.eval() # Set model to evaluation mode
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Define emotion labels (must be in the same order as during training)
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance",
    "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude",
    "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # No need to move to model.device if model is already on CPU as set above
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0]

    predictions = []
    for i, prob in enumerate(probs):
        if prob > 0.5:
            predictions.append(
                (emotion_labels[i], round(prob.item(), 4))
            )
    # Sort predictions by probability in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

def generate_mental_health_insight(predictions):
    if not predictions:
        return "No strong emotions detected. You seem to be in a neutral state."
        
    insights = []
    emotions = [p[0] for p in predictions]

    # Group emotions for easier combination checks
    has_sadness = any(e in emotions for e in ['sadness', 'grief', 'disappointment', 'remorse'])
    has_anxiety = any(e in emotions for e in ['fear', 'nervousness', 'embarrassment'])
    has_anger = any(e in emotions for e in ['anger', 'annoyance', 'disapproval', 'disgust'])
    has_positive = any(e in emotions for e in ['joy', 'optimism', 'caring', 'gratitude', 'love', 'pride', 'relief', 'amusement', 'excitement'])
    
    # Check for specific combinations
    if has_sadness and has_anxiety:
        insights.append("Experiencing both sadness and anxiety can be particularly overwhelming. Remember to be gentle with yourself, and consider speaking to a supportive friend or professional.")
    elif has_sadness:
        insights.append("Your text indicates feelings of sadness or grief. If these feelings persist and interfere with your daily life, it might be helpful to reach out to a mental health professional.")
    elif has_anxiety:
        insights.append("Signs of anxiety or fear are present. Deep breathing exercises, mindfulness, or talking to someone you trust can help manage these feelings.")
        
    if has_anger and has_sadness:
        insights.append("A mix of anger and sadness often indicates deep frustration or feeling misunderstood. Taking a moment to write down your thoughts might help process this complex mix.")
    elif has_anger and not (has_sadness and has_anxiety): # Avoid overwhelming with too many insights
        insights.append("You seem to be experiencing frustration or anger. Taking a step back and practicing relaxation techniques might help you process these emotions constructively.")
        
    if has_positive and (has_sadness or has_anxiety or has_anger):
        insights.append("It's interesting that you're expressing both positive and challenging emotions. Mixed feelings are completely normal and show emotional depth and resilience.")
    elif has_positive and not (has_sadness or has_anxiety or has_anger):
        insights.append("Positive emotions detected! Acknowledging and savoring these moments contributes significantly to overall mental well-being.")
        
    if not insights:
        top_emotion = emotions[0]
        insights.append(f"Your primary detected emotion is {top_emotion.capitalize()}. Acknowledging your emotional state is a great first step in emotional awareness.")
        
    return " ".join(insights)

st.set_page_config(page_title="Mental Health Emotion Detector", page_icon=":brain:")

st.title("🧠 Mental Health Emotion Detector")
st.markdown(
    "This application detects emotions in text using a fine-tuned RoBERTa model."
    "Enter some text below to see the predicted emotions and their confidence scores."
)

user_input = st.text_area("Enter text here:", "I am so happy and excited about this project!")

if st.button("Analyze Emotion"):
    if user_input:
        with st.spinner("Analyzing your text... Please wait."):
            prediction = predict_emotion(user_input)
        if prediction:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Predicted Emotions:")
                for emotion, score in prediction:
                    st.write(f"- **{emotion.capitalize()}**: {score:.4f}")
                
                st.subheader("Mental Health Insight:")
                st.info(generate_mental_health_insight(prediction))
                
            with col2:
                # Create a bar chart with Plotly
                df = pd.DataFrame(prediction, columns=["Emotion", "Confidence"])
                fig = px.bar(df, x="Confidence", y="Emotion", orientation='h', 
                             title="Emotion Confidence Scores", color="Emotion")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No strong emotions detected (all probabilities below 0.5).")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---<br>Developed by Mahesh Awchar", unsafe_allow_html=True)
