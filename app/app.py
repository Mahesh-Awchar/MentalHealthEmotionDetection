import sys
import os
import base64
import time
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
import streamlit.components.v1 as components

# Ensure the root project directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import EmotionPredictor

HF_MODEL = "awcharm034/mhed-roberta-goemotions"

# --- Page Configuration ---
st.set_page_config(
    page_title="Mental Health Emotion AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Futuristic AI UI) ---
st.markdown("""
<style>
    /* Animations */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes float {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(20px, 40px) scale(1.1); }
    }
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    @keyframes blink {
        50% { border-color: transparent }
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }

    /* Dark AI Theme Base */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Animated Background Blobs */
    .bg-blobs {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        overflow: hidden;
    }
    .blob {
        position: absolute;
        filter: blur(80px);
        opacity: 0.2;
        border-radius: 50%;
        animation: float 10s infinite alternate ease-in-out;
    }
    .blob1 { top: -10%; left: -10%; width: 40vw; height: 40vw; background: rgba(0, 201, 255, 0.4); animation-delay: 0s; }
    .blob2 { bottom: -10%; right: -10%; width: 50vw; height: 50vw; background: rgba(146, 254, 157, 0.4); animation-delay: -5s; }

    /* Gradient Text Header */
    .gradient-text {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: clamp(2rem, 5vw, 4rem);
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    
    /* Subtitle */
    .typing-container {
        display: inline-block;
        overflow: hidden;
        white-space: nowrap;
        border-right: 2px solid #00C9FF;
        animation: typing 3.5s steps(40, end), blink 0.75s step-end infinite;
        text-align: center;
        color: #8b949e;
        font-size: clamp(1rem, 2vw, 1.2rem);
        font-weight: 300;
        margin: 0 auto;
    }
    
    /* Badges */
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-top: 15px;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        backdrop-filter: blur(5px);
        display: flex;
        align-items: center;
        gap: 8px;
        color: #c9d1d9;
    }
    .live-indicator {
        width: 10px;
        height: 10px;
        background: #4caf50;
        border-radius: 50%;
        box-shadow: 0 0 10px #4caf50;
        animation: pulse 1.5s infinite;
    }

    /* Glassmorphism Containers */
    .glass-box {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        animation: fadeUp 0.6s ease-out forwards;
        transition: all 0.3s ease-in-out;
    }
    .glass-box:hover {
        border: 1px solid rgba(0, 201, 255, 0.4);
        box-shadow: 0 4px 30px rgba(0, 201, 255, 0.15);
        transform: translateY(-2px);
    }
    
    /* Metrics */
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00C9FF;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
    }

    /* Glowing Download Button */
    .glow-button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        border: none;
        color: #000 !important;
        padding: 12px 20px;
        text-align: center;
        text-decoration: none;
        display: block;
        font-size: 14px;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.3s;
        box-shadow: 0 0 10px rgba(0, 201, 255, 0.4);
        width: 100%;
        margin-top: 15px;
    }
    .glow-button:hover {
        box-shadow: 0 0 20px rgba(146, 254, 157, 0.8);
        transform: scale(1.02);
    }
    
    /* Chat Message adjustments */
    div[data-testid="stChatMessage"] {
        background-color: transparent;
        animation: fadeUp 0.4s ease-out forwards;
        padding: 10px;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
    }
    
    /* Hero Spacing */
    .hero-section {
        padding: 2rem 1rem 1rem 1rem;
        text-align: center;
    }
</style>

<div class="bg-blobs">
    <div class="blob blob1"></div>
    <div class="blob blob2"></div>
</div>
""", unsafe_allow_html=True)

# --- Emoji & Tips Engine ---
EMOJI_MAP = {
    "admiration": "🤩", "amusement": "😂", "anger": "😡", "annoyance": "😒",
    "approval": "👍", "caring": "🥰", "confusion": "😕", "curiosity": "🤔",
    "desire": "😏", "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
    "embarrassment": "😳", "excitement": "🎉", "fear": "😨", "gratitude": "🙏",
    "grief": "😢", "joy": "😊", "love": "❤️", "nervousness": "😬",
    "optimism": "🌟", "pride": "😌", "realization": "💡", "relief": "😮‍💨",
    "remorse": "😔", "sadness": "😭", "surprise": "😲", "neutral": "😐"
}

# --- Emotion Color Mapping ---
EMOTION_COLORS = {
    "sadness": "#1f77b4", "joy": "#ff7f0e", "love": "#e377c2", "anger": "#d62728",
    "fear": "#9467bd", "surprise": "#17becf", "neutral": "#7f7f7f", "amusement": "#ffbb78",
    "admiration": "#98df8a", "approval": "#2ca02c", "caring": "#f7b6d2",
    "excitement": "#ff9896", "gratitude": "#c5b0d5", "optimism": "#c49c94",
    "pride": "#dbdb8d", "relief": "#9edae5", "annoyance": "#c5b0d5",
    "confusion": "#c49c94", "curiosity": "#f7b6d2", "desire": "#ff9896",
    "disappointment": "#1f77b4", "disapproval": "#d62728", "disgust": "#8c564b",
    "embarrassment": "#e377c2", "grief": "#7f7f7f", "nervousness": "#9467bd",
    "realization": "#17becf", "remorse": "#8c564b"
}

def generate_wellness_tips(active_emotions):
    tips = []
    high_risk = ['sadness', 'grief', 'fear', 'nervousness', 'stress', 'depression']
    if any(e in active_emotions for e in high_risk):
        tips.append("🌬️ **Breathe:** Try the 4-7-8 breathing technique to calm your nervous system.")
        tips.append("🫂 **Connect:** Reach out to a friend, family member, or professional support hotline.")
        tips.append("🛑 **Grounding:** Try the 5-4-3-2-1 exercise to bring yourself back to the present.")
    
    if any(e in active_emotions for e in ['anger', 'annoyance', 'disapproval', 'disgust', 'frustration']):
        tips.append("🚶 **Step Away:** Take a 10-minute walk to physically distance yourself from the stressor.")
        tips.append("✍️ **Journal:** Write down exactly what's frustrating you to process the feeling constructively.")
        
    if any(e in active_emotions for e in ['joy', 'optimism', 'caring', 'gratitude', 'love', 'excitement']):
        tips.append("📝 **Gratitude:** Write down 3 things you're grateful for to lock in this positive energy!")
        tips.append("🎉 **Share the Positivity:** Compliment someone or share your good mood with a loved one.")
        
    if not tips:
        tips.append("💧 **Hydrate & Rest:** Basic self-care (water and sleep) is the foundation of emotional balance.")
        tips.append("🧘 **Mindfulness:** Take a moment to stretch and observe your surroundings without judgment.")
        
    return tips

def calculate_wellness_metrics(top_emotions):
    stress_emotions = ['fear', 'nervousness', 'stress', 'anger', 'annoyance', 'disapproval', 'disgust', 'sadness', 'grief']
    positive_emotions = ['joy', 'optimism', 'caring', 'gratitude', 'love', 'excitement', 'admiration', 'approval', 'amusement', 'pride', 'relief']
    
    stress_score = sum(score for emo, score in top_emotions.items() if emo in stress_emotions)
    positive_score = sum(score for emo, score in top_emotions.items() if emo in positive_emotions)
    
    total = sum(top_emotions.values())
    if total == 0: total = 1
    
    wellness_index = min(100, max(0, int(((positive_score - stress_score) / total + 1) * 50)))
    
    if stress_score > 0.5: stress_level = "High 🔴"
    elif stress_score > 0.2: stress_level = "Medium 🟡"
    else: stress_level = "Low 🟢"
        
    return wellness_index, stress_level

def create_download_link(text_content, filename="wellness_report.txt"):
    b64 = base64.b64encode(text_content.encode('utf-8')).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="glow-button">📥 Download Analysis Report</a>'

# --- Model Loader ---
@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        model_dir = "models/roberta_goemotions_model"
        if os.path.exists(model_dir):
            return EmotionPredictor(model_path=model_dir)
        if os.path.exists("models/config.json"):
            return EmotionPredictor(model_path="models")
        return EmotionPredictor(model_path=HF_MODEL)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar Dashboard ---
with st.sidebar:
    st.markdown("## 🧠 System Settings")
    st.markdown("---")
    st.markdown("### ⚙️ Diagnostics")
    device_status = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🟡 CPU (Standard)"
    st.markdown(f"**Compute Node:** {device_status}")
    st.markdown(f"**Streamlit UI:** v{st.__version__}")
    st.markdown("**Core Model:** RoBERTa-base")
    st.markdown("**Dataset:** GoEmotions (28 Classes)")
    
    st.markdown("---")
    st.markdown("### 📊 Session Tracking")
    st.markdown(f"**Total Analyses:** {len(st.session_state.history)}")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
        
    if len(st.session_state.history) > 0:
        all_history_text = "FULL SESSION HISTORY\n\n"
        for i, h in enumerate(st.session_state.history):
            all_history_text += f"--- Analysis {i+1} ---\n"
            all_history_text += f"Time: {h['timestamp']}\n"
            all_history_text += f"Input: {h['text']}\n"
            all_history_text += f"Top Emotion: {list(h['top_emotions'].keys())[0]} ({list(h['top_emotions'].values())[0]:.1%})\n\n"
        
        b64_history = base64.b64encode(all_history_text.encode('utf-8')).decode()
        st.markdown(
            f'<a href="data:file/txt;base64,{b64_history}" download="full_session_history.txt" class="glow-button" style="text-align:center; margin-top:10px;">📁 Export Full History</a>', 
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption("Developed for Mental Health Emotion Detection System. Not a substitute for professional medical advice.")

# --- Main UI ---
st.markdown("<div class='hero-section'>", unsafe_allow_html=True)
st.markdown("<h1 class='gradient-text'>Mental Health Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown("<div><span class='typing-container'>AI-Powered Emotional Intelligence System</span></div>", unsafe_allow_html=True)

st.markdown("""
<div class='badge-container'>
    <div class='hero-badge'><div class='live-indicator'></div> Live AI</div>
    <div class='hero-badge'>🧠 RoBERTa Transformers</div>
    <div class='hero-badge'>📊 GoEmotions Dataset</div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

predictor = load_predictor()

if predictor is None:
    st.error("⚠️ Trained model not found! Please run `python main.py --mode train` or `python main.py --mode download` first.")
    st.stop()

# --- Emotion Timeline Graph ---
if len(st.session_state.history) > 1:
    with st.expander("📈 View Emotional Timeline Analytics", expanded=False):
        timeline_data = []
        for h in st.session_state.history:
            for emo, score in h['top_emotions'].items():
                timeline_data.append({
                    "Time": h['timestamp'],
                    "Emotion": emo.capitalize(),
                    "Confidence": score
                })
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            fig_line = px.line(df_timeline, x="Time", y="Confidence", color="Emotion", markers=True, 
                               color_discrete_map={k.capitalize(): v for k, v in EMOTION_COLORS.items()})
            fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9'), height=400)
            st.plotly_chart(fig_line, use_container_width=True)

# --- History Rendering (ChatGPT style) ---
for idx, entry in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(entry["text"])
        
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(f"**AI Analysis:** {entry['insight']}")
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
            
            well_idx, stress_lvl = calculate_wellness_metrics(entry["top_emotions"])
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown(f"<div class='metric-box'><div class='metric-value'>{well_idx}%</div><div class='metric-label'>Wellness Score</div></div>", unsafe_allow_html=True)
            with mcol2:
                st.markdown(f"<div class='metric-box'><div class='metric-value'>{stress_lvl}</div><div class='metric-label'>Stress Level</div></div>", unsafe_allow_html=True)
                
            st.markdown("#### 📊 Top Detected Emotions")
            top_emotions = entry["top_emotions"]
            
            for emo, score in top_emotions.items():
                emoji = EMOJI_MAP.get(emo.lower(), "✨")
                color = EMOTION_COLORS.get(emo.lower(), "#00C9FF")
                st.markdown(f"**{emoji} {emo.capitalize()} ({score:.1%})**")
                st.markdown(f"""
                <div style="width: 100%; background-color: rgba(255,255,255,0.1); border-radius: 5px; margin-bottom: 15px;">
                  <div style="width: {score*100}%; height: 8px; background-color: {color}; border-radius: 5px; transition: width 1s ease-in-out;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("#### 🌿 Wellness Recommendations")
            for tip in entry["tips"]:
                st.write(tip)
                
            # Generate Exportable Report Data
            report_text = f"EMOTIONAL ANALYSIS REPORT\nDate: {entry['timestamp']}\n\n-- Input Text --\n{entry['text']}\n\n-- Top Emotions --\n"
            for emo, score in top_emotions.items(): report_text += f"{emo.capitalize()}: {score:.1%}\n"
            report_text += f"\n-- AI Insight --\n{entry['insight']}\n\n-- Wellness Tips --\n"
            for tip in entry["tips"]: report_text += f"{tip}\n"
            report_text += "\nGenerated by Mental Health Emotion Detection AI"
            
            st.markdown(create_download_link(report_text, f"wellness_report_{idx+1}.txt"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🎯 Radar Chart", "🥧 Pie Chart"])
            
            df = pd.DataFrame({
                "Emotion": [e.capitalize() for e in top_emotions.keys()],
                "Confidence": list(top_emotions.values()),
                "Color": [EMOTION_COLORS.get(e.lower(), "#00C9FF") for e in top_emotions.keys()]
            })
            
            # Plotly Theme Setup
            layout_args = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9'), height=350, margin=dict(t=20, b=20, l=10, r=10))
            
            with tab1:
                fig_bar = px.bar(df, x="Confidence", y="Emotion", orientation='h', color="Emotion", color_discrete_map={k.capitalize(): v for k, v in EMOTION_COLORS.items()}, range_x=[0, 1])
                fig_bar.update_layout(**layout_args, yaxis={'categoryorder':'total ascending'}, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{idx}")
                
            with tab2:
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=df["Confidence"], theta=df["Emotion"], fill='toself', line_color='#00C9FF', name='Confidence'))
                fig_radar.update_layout(**layout_args, polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.2)')), showlegend=False)
                st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{idx}")
                
            with tab3:
                fig_pie = px.pie(df, names="Emotion", values="Confidence", hole=0.4, color="Emotion", color_discrete_map={k.capitalize(): v for k, v in EMOTION_COLORS.items()})
                fig_pie.update_layout(**layout_args, showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{idx}")
                
            st.markdown("</div>", unsafe_allow_html=True)

# --- Interactive Chat Input ---
user_text = st.chat_input("Express your feelings... e.g., 'I feel overwhelmed and anxious about the future.'")

if user_text:
    with st.spinner("🧠 Analyzing emotional patterns using RoBERTa..."):
        time.sleep(0.6) # Add small delay for realistic AI processing feel
        all_emotions, active_emotions = predictor.predict_emotion(user_text)
        
        top_5 = dict(list(all_emotions.items())[:5])
        insight = predictor.generate_mental_health_insight(active_emotions)
        tips = generate_wellness_tips(active_emotions)
        
        # Save to history
        st.session_state.history.append({
            "text": user_text,
            "top_emotions": top_5,
            "insight": insight,
            "tips": tips,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    # Reload page to render new response
    st.rerun()
