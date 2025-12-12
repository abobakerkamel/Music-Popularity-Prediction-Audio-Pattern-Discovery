"""
Music Intelligence Platform - Streamlit Version
Professional ML Application with Beautiful UI
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG (Must be first Streamlit command)
# ============================================================================
st.set_page_config(
    page_title="Music Intelligence Platform",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM KERAS LAYER DEFINITION
# ============================================================================
try:
    from tensorflow import keras
    import tensorflow as tf
    
    if hasattr(keras, 'saving'):
        register_serializable = keras.saving.register_keras_serializable
    else:
        register_serializable = keras.utils.register_keras_serializable
    
    @register_serializable(package="Custom", name="SimpleAttention")
    class SimpleAttention(keras.layers.Layer):
        """Custom Attention Layer for Deep Music Model"""
        def __init__(self, units=32, **kwargs):
            super(SimpleAttention, self).__init__(**kwargs)
            self.units = units
        
        def build(self, input_shape):
            self.W = self.add_weight(
                name='attention_weight',
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True
            )
            self.b = self.add_weight(
                name='attention_bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
            self.u = self.add_weight(
                name='attention_context',
                shape=(self.units,),
                initializer='glorot_uniform',
                trainable=True
            )
            super(SimpleAttention, self).build(input_shape)
        
        def call(self, x):
            uit = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
            ait = tf.matmul(uit, tf.expand_dims(self.u, axis=-1))
            ait = tf.squeeze(ait, axis=-1)
            ait = tf.nn.softmax(ait, axis=-1)
            weighted_input = x * tf.expand_dims(ait, axis=-1)
            return weighted_input
        
        def get_config(self):
            config = super(SimpleAttention, self).get_config()
            config.update({'units': self.units})
            return config
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)

except (ImportError, AttributeError, Exception) as e:
    print(f"⚠ Warning: Could not load Keras components: {e}")
    keras = None
    SimpleAttention = None

# ============================================================================
# ULTRA PREMIUM CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* استيراد خطوط جوجل */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* تطبيق الخط على كل العناصر */
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* خلفية متدرجة متحركة */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* تصميم الـ Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,12,41,0.95) 0%, rgba(36,36,62,0.95) 100%);
        border-right: 2px solid rgba(138, 43, 226, 0.3);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] h1 {
        color: #fff;
        text-align: center;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(138, 43, 226, 0.8);
        padding: 20px 0;
    }
    
    /* بطاقات التوقع */
    .prediction-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin: 30px 0;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5);
        animation: float 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .hit-potential { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: 3px solid rgba(240, 147, 251, 0.5);
    }
    
    .good-potential { 
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: 3px solid rgba(79, 172, 254, 0.5);
    }
    
    .moderate-potential { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 3px solid rgba(102, 126, 234, 0.5);
    }
    
    /* عناوين مع تأثير Neon */
    h1, h2, h3 {
        color: #fff !important;
        text-shadow: 0 0 10px rgba(138, 43, 226, 0.8), 0 0 20px rgba(138, 43, 226, 0.5);
        font-weight: 700 !important;
    }
    
   /* تحسين المدخلات */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1a1a2e !important;
        border: 2px solid rgba(138, 43, 226, 0.5) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(26, 26, 46, 0.4) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #8a2be2 !important;
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.6) !important;
        background: rgba(255, 255, 255, 1) !important;
    }
    
    /* تحسين Labels */
    .stTextInput label,
    .stSelectbox label,
    .stSlider label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }
    /* السلايدرات */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #8a2be2, #da70d6) !important;
    }
    
    /* الأزرار */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 40px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* بطاقات البيانات */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
        backdrop-filter: blur(10px);
    }
    
    /* المقاييس */
    [data-testid="stMetricValue"] {
        color: #8a2be2 !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(138, 43, 226, 0.5);
    }
    
    /* الـ Expander */
    .streamlit-expanderHeader {
        background: rgba(138, 43, 226, 0.1) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* الـ Radio Buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(138, 43, 226, 0.3);
    }
    
    /* النصوص */
    p, label, span {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* تحسين Plotly Charts */
    .js-plotly-plot {
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(138, 43, 226, 0.3);
        padding: 10px;
    }
    
    /* شاشة التحميل */
    .stSpinner > div {
        border-color: #8a2be2 transparent transparent transparent !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(138, 43, 226, 0.3) !important;
        margin: 30px 0;
    }
    
    /* Warning/Info boxes */
    .stAlert {
        background: rgba(138, 43, 226, 0.1) !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS (Cached)
# ============================================================================
@st.cache_resource
def load_models():
    """Load all models once and cache them."""
    try:
        ensemble_bundle = joblib.load('model_artifacts/ensemble_bundle.joblib')
        
        xgb_model = ensemble_bundle['xgboost_model']
        rf_model = ensemble_bundle['rf_model']
        scaler = ensemble_bundle['scaler']
        training_cols = ensemble_bundle['training_columns']
        selected_feats = ensemble_bundle['selected_features']
        
        raw_artist_map = ensemble_bundle['artist_map']
        raw_genre_map = ensemble_bundle['genre_map']
        artist_map = raw_artist_map.to_dict() if hasattr(raw_artist_map, 'to_dict') else raw_artist_map
        genre_map = raw_genre_map.to_dict() if hasattr(raw_genre_map, 'to_dict') else raw_genre_map
        
        nn_model = None
        try:
            if keras and os.path.exists('model_artifacts/neural_net.keras'):
                nn_model = keras.models.load_model('model_artifacts/neural_net.keras')
        except Exception as e:
            print(f"⚠ Neural Net loading skipped: {e}")
        
        deep_model = None
        try:
            if keras and SimpleAttention and os.path.exists('advanced_artifacts/deep_music_model.keras'):
                deep_model = keras.models.load_model(
                    'advanced_artifacts/deep_music_model.keras',
                    custom_objects={'SimpleAttention': SimpleAttention}
                )
        except Exception as e:
            print(f"⚠ Deep Model loading skipped: {e}")
        
        kmeans, pca, cluster_scaler, feature_cols = None, None, None, None
        if os.path.exists('advanced_artifacts/unsupervised_bundle.joblib'):
            unsupervised_bundle = joblib.load('advanced_artifacts/unsupervised_bundle.joblib')
            kmeans = unsupervised_bundle['kmeans_model']
            pca = unsupervised_bundle['pca_model']
            cluster_scaler = unsupervised_bundle['cluster_scaler']
            feature_cols = unsupervised_bundle['feature_cols']
        
        print("✅ All essential models loaded!")
        
        return {
            'xgb': xgb_model,
            'rf': rf_model,
            'nn': nn_model,
            'scaler': scaler,
            'artist_map': artist_map,
            'genre_map': genre_map,
            'training_cols': training_cols,
            'selected_feats': selected_feats,
            'deep': deep_model,
            'kmeans': kmeans,
            'pca': pca,
            'cluster_scaler': cluster_scaler,
            'feature_cols': feature_cols
        }
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_dataset():
    if os.path.exists('advanced_artifacts/final_music_data.csv'):
        return pd.read_csv('advanced_artifacts/final_music_data.csv')
    return pd.DataFrame()

# Load everything
with st.spinner('🚀 Loading Intelligence Systems...'):
    models = load_models()
    df = load_dataset()

if models is None:
    st.stop()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def make_prediction(artist_name, genre, tempo, energy, danceability, loudness, 
                   speechiness, acousticness, instrumentalness, liveness, 
                   valence, duration_ms, explicit, mode, key, time_signature):
    try:
        duration_min = duration_ms / 60000.0
        energy_dance_ratio = energy / (danceability + 0.01)
        electronic_score = (energy * 0.4) + (danceability * 0.3) + ((1 - acousticness) * 0.3)
        
        if pd.isna(artist_name) or str(artist_name).strip() == "":
            artist_encoded = 45.0
        else:
            artists = [a.strip().lower() for a in str(artist_name).split(';')]
            encoded_vals = [models['artist_map'].get(a, 45.0) for a in artists]
            artist_encoded = np.max(encoded_vals) if encoded_vals else 45.0
        
        genre_clean = str(genre).strip().lower()
        genre_encoded = models['genre_map'].get(genre_clean, 50.0)
        
        raw_features = {
            'tempo': tempo, 'energy': energy, 'danceability': danceability,
            'loudness': loudness, 'speechiness': speechiness,
            'acousticness': acousticness, 'instrumentalness': instrumentalness,
            'liveness': liveness, 'valence': valence, 'duration_ms': duration_ms,
            'explicit': explicit, 'mode': mode, 'key': key,
            'time_signature': time_signature, 'duration_min': duration_min,
            'energy_dance_ratio': energy_dance_ratio,
            'electronic_score': electronic_score,
            'artists_encoded': artist_encoded,
            'track_genre_encoded': genre_encoded
        }
        
        X_input = pd.DataFrame([raw_features])
        X_aligned = pd.DataFrame(0, index=[0], columns=models['training_cols'])
        
        for col in X_input.columns:
            if col in X_aligned.columns:
                X_aligned[col] = X_input[col].values
        
        X_scaled = models['scaler'].transform(X_aligned)
        
        if models['selected_feats'] is not None:
            if len(models['selected_feats']) > 0:
                first_elem = models['selected_feats'][0]
                
                if isinstance(first_elem, (str, np.str_)):
                    selected_col_indices = [models['training_cols'].index(feat) 
                                          for feat in models['selected_feats'] 
                                          if feat in models['training_cols']]
                    X_selected = X_scaled[:, selected_col_indices]
                else:
                    X_selected = X_scaled[:, models['selected_feats']]
            else:
                X_selected = X_scaled
        else:
            X_selected = X_scaled
        
        pred_xgb = models['xgb'].predict(X_selected)[0]
        pred_rf = models['rf'].predict(X_selected)[0]
        
        if models['nn'] is not None:
            try:
                pred_nn = models['nn'].predict(X_scaled, verbose=0)[0][0]
            except:
                pred_nn = (pred_xgb + pred_rf) / 2
        else:
            pred_nn = (pred_xgb + pred_rf) / 2
        
        final_pred = (pred_xgb * 0.7) + (pred_rf * 0.2) + (pred_nn * 0.1)
        return np.clip(final_pred, 0, 100), raw_features
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return 50.0, {}

# ============================================================================
# UI COMPONENTS - ENHANCED
# ============================================================================
def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Popularity Score", 'font': {'size': 28, 'color': 'white', 'family': 'Poppins'}},
        delta={'reference': 50, 'increasing': {'color': "#00ff88"}},
        number={'font': {'size': 60, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 3, 'tickcolor': "#8a2be2"},
            'bar': {'color': "#8a2be2", 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 3,
            'bordercolor': "rgba(138, 43, 226, 0.5)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(244, 67, 54, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(76, 175, 80, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ff0066", 'width': 6},
                'thickness': 0.85,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=400, 
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Poppins'}
    )
    return fig

def create_radar_chart(features):
    categories = ['Energy', 'Danceability', 'Valence', 'Acousticness', 
                 'Instrumentalness', 'Liveness', 'Speechiness']
    values = [
        features.get('energy', 0)*100,
        features.get('danceability', 0)*100,
        features.get('valence', 0)*100,
        features.get('acousticness', 0)*100,
        features.get('instrumentalness', 0)*100,
        features.get('liveness', 0)*100,
        features.get('speechiness', 0)*100
    ]
    
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(138, 43, 226, 0.4)',
        line=dict(color='#da70d6', width=3)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 100],
                gridcolor='rgba(138, 43, 226, 0.3)',
                tickfont={'color': 'white'}
            ),
            angularaxis=dict(
                gridcolor='rgba(138, 43, 226, 0.3)',
                tickfont={'color': 'white', 'size': 12}
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        title={'text': "Vibe Analysis", 'font': {'size': 24, 'color': 'white'}},
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Poppins'}
    )
    return fig

def create_cluster_scatter(cluster_id):
    if df.empty or 'cluster_kmeans' not in df.columns:
        return go.Figure()
    
    df_plot = df.copy()
    df_plot['Selected'] = (df_plot['cluster_kmeans'] == cluster_id).astype(int)
    df_plot = df_plot.sort_values('Selected')
    
    fig = px.scatter(
        df_plot,
        x='pca_1',
        y='pca_2',
        color='cluster_kmeans',
        hover_data=['track_name', 'artists', 'popularity'],
        title=f"Cluster Landscape (Cluster {cluster_id})",
        color_continuous_scale='twilight',
        size='Selected',
        size_max=12
    )
    
    fig.update_layout(
        height=500, 
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)',
        font={'color': 'white', 'family': 'Poppins'},
        title={'font': {'size': 22}}
    )
    return fig

def create_feature_importance():
    if models['xgb'] and hasattr(models['xgb'], 'feature_importances_'):
        try:
            importances = models['xgb'].feature_importances_
            selected = models['selected_feats']
            
            if len(selected) > 0 and isinstance(selected[0], (str, np.str_)):
                feature_names = [f for f in selected if f in models['training_cols']]
                feature_indices = [models['training_cols'].index(f) for f in feature_names]
                valid_indices = [i for i in feature_indices if i < len(importances)]
                feature_importances = importances[valid_indices]
                feature_names = [feature_names[i] for i, idx in enumerate(feature_indices) if idx < len(importances)]
            else:
                n_feats = min(len(selected), len(importances))
                feature_indices = selected[:n_feats]
                valid_indices = [i for i in feature_indices if i < len(importances)]
                feature_names = [models['training_cols'][i] for i in valid_indices if i < len(models['training_cols'])]
                feature_importances = importances[valid_indices]
            
            if len(feature_importances) == 0:
                feature_names = models['training_cols'][:len(importances)]
                feature_importances = importances
            
            top_n = min(15, len(feature_importances))
            sorted_indices = np.argsort(feature_importances)[-top_n:]
            
            plot_names = [feature_names[i] for i in sorted_indices]
            plot_values = [feature_importances[i] for i in sorted_indices]
            
            fig = go.Figure(go.Bar(
                x=plot_values,
                y=plot_names,
                orientation='h',
                marker=dict(
                    color=plot_values, 
                    colorscale='Purples',
                    line=dict(color='rgba(138, 43, 226, 0.8)', width=2)
                )
            ))
            
            fig.update_layout(
                title={'text': "Top Feature Importances", 'font': {'size': 22, 'color': 'white'}},
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=500,
                margin=dict(l=150, r=40, t=80, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.2)',
                font={'color': 'white', 'family': 'Poppins'},
                xaxis={'gridcolor': 'rgba(138, 43, 226, 0.2)'},
                yaxis={'gridcolor': 'rgba(138, 43, 226, 0.2)'}
            )
            return fig
        
        except Exception as e:
            st.warning(f"⚠ Could not generate feature importance chart: {e}")
            return go.Figure()
    
    return go.Figure()

def create_genre_distribution():
    if not df.empty and 'track_genre' in df.columns:
        genre_counts = df['track_genre'].value_counts().head(15)
        fig = go.Figure(go.Bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation='h',
            marker=dict(
                color=genre_counts.values, 
                colorscale='Teal',
                line=dict(color='rgba(0, 128, 128, 0.8)', width=2)
            )
        ))
        fig.update_layout(
            title={'text': "Top 15 Genre Distribution", 'font': {'size': 22, 'color': 'white'}},
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font={'color': 'white', 'family': 'Poppins'},
            xaxis={'gridcolor': 'rgba(0, 128, 128, 0.2)'},
            yaxis={'gridcolor': 'rgba(0, 128, 128, 0.2)'}
        )
        return fig
    return go.Figure()

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================
st.markdown("<h1 style='text-align: center; font-size: 48px; margin-bottom: 10px;'>🎵 Music Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px; color: rgba(255,255,255,0.7); margin-bottom: 40px;'>AI-Powered Music Analysis & Prediction System</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>⚡ Navigation</h1>", unsafe_allow_html=True)
    tab_selection = st.radio("Select Module:", 
                            ["🔮 Hit Predictor", "🌌 Cluster Explorer", "📈 Model Insights"],
                            label_visibility="collapsed")
    st.markdown("---")
    
    if not df.empty:
        st.metric("🎼 Total Tracks", f"{len(df):,}")
        st.metric("🌟 Clusters", len(df['cluster_kmeans'].unique()) if 'cluster_kmeans' in df.columns else 0)
        st.markdown("---")
        st.markdown("<p style='text-align: center; font-size: 12px; opacity: 0.6;'>Powered by Advanced ML</p>", unsafe_allow_html=True)

if tab_selection == "🔮 Hit Predictor":
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📋 Track Metadata")
        artist_name = st.text_input("🎤 Artist Name", "The Weeknd")
        
        unique_genres = sorted(list(models['genre_map'].keys())) if models else ['pop']
        genre = st.selectbox("🎸 Genre", unique_genres[:100] if len(unique_genres) > 100 else unique_genres)
        
        st.markdown("### 🎹 Audio Features")
        c1, c2 = st.columns(2)
        tempo = c1.slider("🥁 Tempo (BPM)", 0, 250, 120)
        energy = c1.slider("⚡ Energy", 0.0, 1.0, 0.7, 0.01)
        danceability = c1.slider("💃 Danceability", 0.0, 1.0, 0.65, 0.01)
        duration_ms = c2.slider("⏱️ Duration (ms)", 0, 600000, 210000, 1000)
        valence = c2.slider("😊 Valence (Mood)", 0.0, 1.0, 0.5, 0.01)
        loudness = c2.slider("🔊 Loudness (dB)", -60.0, 0.0, -5.0, 0.5)
        
        with st.expander("⚙️ Advanced Audio Features"):
            ac1, ac2 = st.columns(2)
            acousticness = ac1.slider("🎻 Acousticness", 0.0, 1.0, 0.2, 0.01)
            speechiness = ac1.slider("🗣️ Speechiness", 0.0, 1.0, 0.05, 0.01)
            instrumentalness = ac2.slider("🎼 Instrumentalness", 0.0, 1.0, 0.0, 0.01)
            liveness = ac2.slider("🎭 Liveness", 0.0, 1.0, 0.1, 0.01)
            
            st.markdown("#### 🎛️ Technical Parameters")
            tc1, tc2, tc3 = st.columns(3)
            key = tc1.slider("🎹 Key", 0, 11, 5)
            time_sig = tc2.slider("⏲️ Time Signature", 3, 7, 4)
            mode = tc3.slider("🎵 Mode", 0, 1, 1)
            explicit = st.slider("🔞 Explicit Content", 0, 1, 0)
        
        st.markdown("---")
        predict_btn = st.button("🎯 PREDICT POPULARITY", type="primary", use_container_width=True)
    
    with col2:
        if predict_btn:
            with st.spinner('🔮 Analyzing musical DNA...'):
                score, features = make_prediction(
                    artist_name, genre, tempo, energy, danceability, loudness,
                    speechiness, acousticness, instrumentalness, liveness,
                    valence, duration_ms, explicit, mode, key, time_sig
                )
                
                # عرض النتيجة بتأثير درامي
                if score >= 70:
                    emoji = "🔥"
                    label = "MEGA HIT POTENTIAL!"
                    css_class = "hit-potential"
                elif score >= 50:
                    emoji = "✨"
                    label = "GOOD VIBES"
                    css_class = "good-potential"
                else:
                    emoji = "📊"
                    label = "MODERATE POTENTIAL"
                    css_class = "moderate-potential"
                
                st.markdown(f'''
                <div class="prediction-box {css_class}">
                    <div style="font-size: 48px; margin-bottom: 10px;">{emoji}</div>
                    <div style="font-size: 56px; font-weight: 800;">{score:.1f}</div>
                    <div style="font-size: 18px; margin-top: 10px; opacity: 0.9;">{label}</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # الرسوم البيانية
                st.plotly_chart(create_gauge_chart(score), use_container_width=True)
                st.plotly_chart(create_radar_chart(features), use_container_width=True)
                
                # معلومات إضافية
                with st.expander("📊 Detailed Feature Analysis"):
                    feat_df = pd.DataFrame({
                        'Feature': ['Tempo', 'Energy', 'Danceability', 'Valence', 
                                   'Acousticness', 'Loudness'],
                        'Value': [tempo, f"{energy:.2f}", f"{danceability:.2f}", 
                                 f"{valence:.2f}", f"{acousticness:.2f}", f"{loudness:.1f}"]
                    })
                    st.dataframe(feat_df, use_container_width=True, hide_index=True)

elif tab_selection == "🌌 Cluster Explorer":
    st.markdown("<h2 style='text-align: center;'>🌌 Cluster Explorer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.7;'>Explore musical universes and discover similar tracks</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not df.empty and models['kmeans']:
        unique_clusters = sorted(df['cluster_kmeans'].unique())
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            sel_cluster = st.selectbox("🎯 Select Cluster to Explore", unique_clusters)
        
        st.markdown("---")
        
        # إحصائيات الكلاستر
        cluster_data = df[df['cluster_kmeans'] == sel_cluster]
        
        st.markdown("### 📊 Cluster Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("🎵 Tracks", f"{len(cluster_data):,}")
        with stat_col2:
            avg_pop = cluster_data['popularity'].mean()
            st.metric("⭐ Avg Popularity", f"{avg_pop:.1f}")
        with stat_col3:
            top_genre = cluster_data['track_genre'].mode()[0] if 'track_genre' in cluster_data.columns else "N/A"
            st.metric("🎸 Top Genre", top_genre)
        with stat_col4:
            avg_energy = cluster_data['energy'].mean() if 'energy' in cluster_data.columns else 0
            st.metric("⚡ Avg Energy", f"{avg_energy:.2f}")
        
        st.markdown("---")
        
        # الرسم البياني
        st.plotly_chart(create_cluster_scatter(sel_cluster), use_container_width=True)
        
        st.markdown("---")
        
        # أفضل الأغاني في الكلاستر
        st.markdown(f"### 🏆 Top 10 Tracks in Cluster {sel_cluster}")
        cluster_songs = df[df['cluster_kmeans'] == sel_cluster].sort_values('popularity', ascending=False).head(10)
        
        # تنسيق أفضل للجدول
        display_df = cluster_songs[['track_name', 'artists', 'popularity', 'track_genre']].copy()
        display_df.columns = ['🎵 Track', '🎤 Artist', '⭐ Popularity', '🎸 Genre']
        display_df['⭐ Popularity'] = display_df['⭐ Popularity'].apply(lambda x: f"{x:.0f}")
        
        st.dataframe(
            display_df, 
            use_container_width=True, 
            hide_index=True,
            height=400
        )
    else:
        st.warning("⚠️ Cluster data not available. Please ensure models are loaded correctly.")

else:  # Model Insights
    st.markdown("<h2 style='text-align: center;'>📈 Model Intelligence Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.7;'>Deep dive into model behavior and data patterns</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Model Performance Metrics
    st.markdown("### 🎯 Model Performance")
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("🤖 XGBoost", "Active", delta="70% Weight")
    with perf_col2:
        st.metric("🌲 Random Forest", "Active", delta="20% Weight")
    with perf_col3:
        st.metric("🧠 Neural Net", "Active" if models['nn'] else "N/A", delta="10% Weight")
    with perf_col4:
        st.metric("🎯 Deep Model", "Active" if models['deep'] else "N/A")
    
    st.markdown("---")
    
    # Charts
    chart_col1, chart_col2 = st.columns(2, gap="large")
    
    with chart_col1:
        st.plotly_chart(create_feature_importance(), use_container_width=True)
    
    with chart_col2:
        st.plotly_chart(create_genre_distribution(), use_container_width=True)
    
    st.markdown("---")
    
    # Dataset Overview
    if not df.empty:
        st.markdown("### 📊 Dataset Overview")
        
        overview_col1, overview_col2 = st.columns(2)
        
        with overview_col1:
            st.markdown("#### 🎵 Audio Features Distribution")
            if 'energy' in df.columns:
                feature_stats = df[['energy', 'danceability', 'valence', 'acousticness']].describe()
                st.dataframe(feature_stats, use_container_width=True)
        
        with overview_col2:
            st.markdown("#### ⭐ Popularity Statistics")
            if 'popularity' in df.columns:
                pop_stats = df['popularity'].describe()
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{pop_stats['mean']:.2f}",
                        f"{df['popularity'].median():.2f}",
                        f"{pop_stats['std']:.2f}",
                        f"{pop_stats['min']:.0f}",
                        f"{pop_stats['max']:.0f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Correlation Heatmap
        st.markdown("### 🔥 Feature Correlation Matrix")
        if all(col in df.columns for col in ['energy', 'danceability', 'valence', 'acousticness', 'loudness', 'popularity']):
            corr_features = ['energy', 'danceability', 'valence', 'acousticness', 'loudness', 'popularity']
            corr_matrix = df[corr_features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_features,
                y=corr_features,
                colorscale='Purples',
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 12},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title={'text': "Feature Correlation Heatmap", 'font': {'size': 22, 'color': 'white'}},
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.2)',
                font={'color': 'white', 'family': 'Poppins'},
                xaxis={'side': 'bottom'},
                yaxis={'side': 'left'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; opacity: 0.6;'>
    <p style='font-size: 14px;'>🎵 Music Intelligence Platform v2.0</p>
    <p style='font-size: 12px;'>Powered by Advanced Machine Learning | Built with ❤️</p>
</div>
""", unsafe_allow_html=True)