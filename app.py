import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models

# Streamlit Page Config
st.set_page_config(
    page_title="MRI Brain Tumor Classifier: CNN vs Transfer Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 5px;
    }
    .header-box {
        background: linear-gradient(90deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 25px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .class-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .glioma { background-color: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid #ef4444; }
    .meningioma { background-color: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid #f59e0b; }
    .notumor { background-color: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid #10b981; }
    .pituitary { background-color: rgba(14, 165, 233, 0.2); color: #7dd3fc; border: 1px solid #0ea5e9; }
</style>
""", unsafe_html=True)

# Constants & Paths
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMAGE_SIZE = (224, 224)
ARTIFACTS_DIR = r"c:\Users\SAIF-PC\Downloads\archive\artifacts_output"
TRAIN_DIR = r"c:\Users\SAIF-PC\Downloads\archive\Training"
TEST_DIR = r"c:\Users\SAIF-PC\Downloads\archive\Testing"

# Device setup
device = torch.device("cpu")

# Model Definitions
class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_transfer_model(model_name, num_classes=4):
    if model_name == "EfficientNetB0":
        backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return backbone
    elif model_name == "ResNet50":
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return backbone
    elif model_name == "MobileNetV2":
        backbone = models.mobilenet_v2(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return backbone
    elif model_name == "DenseNet121":
        backbone = models.densenet121(weights=None)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return backbone
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

@st.cache_resource
def load_trained_model(model_name):
    weights_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_best.pt")
    if not os.path.exists(weights_path):
        return None
    
    if model_name == "CNN":
        model = CustomCNN(len(CLASSES))
    else:
        model = build_transfer_model(model_name, len(CLASSES))
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

@st.cache_data
def load_results():
    results_path = os.path.join(ARTIFACTS_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return None

# Image Transform for Inference
eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, img):
    img_rgb = img.convert("RGB")
    tensor = eval_transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
    return probs

# MAIN APP LAYOUT
st.markdown("""
<div class="header-box">
    <h1 style="color: #ffffff; font-weight: 800; margin-bottom: 8px;">🧠 Brain Tumor MRI Classifier</h1>
    <h3 style="color: #a5b4fc; font-weight: 500; margin-top: 0;">Custom CNN Baseline vs. Transfer Learning Architectures</h3>
    <p style="color: #cbd5e1; margin-bottom: 0;">
        Comparison of <b>Custom CNN</b> against 4 state-of-the-art ImageNet pre-trained backbones: 
        <b>EfficientNetB0</b>, <b>ResNet50</b>, <b>MobileNetV2</b>, and <b>DenseNet121</b> on 7,200 Brain MRI scans across 4 classes.
    </p>
    <div style="margin-top: 15px;">
        <span class="class-badge glioma">Glioma</span>
        <span class="class-badge meningioma">Meningioma</span>
        <span class="class-badge notumor">No Tumor</span>
        <span class="class-badge pituitary">Pituitary</span>
    </div>
</div>
""", unsafe_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation 🧭")
nav_selection = st.sidebar.radio(
    "Choose View",
    ["📊 Model Comparison & Leaderboard", "🔬 Live MRI Classifier", "📉 Training Curves & Confusion Matrices", "📁 Dataset Explorer"]
)

results_data = load_results()

if nav_selection == "📊 Model Comparison & Leaderboard":
    st.header("📊 Model Comparison & Leaderboard")
    
    if results_data is None:
        st.warning("⏳ Model training is currently in progress or results.json is generating... Refreshing in a moment!")
        st.stop()
        
    summary_rows = []
    for name, r in results_data.items():
        summary_rows.append({
            "Model": r["name"],
            "Test Accuracy": r["test_accuracy"],
            "Best Val Accuracy": r["best_val_accuracy"],
            "Macro Precision": r["macro_precision"],
            "Macro Recall": r["macro_recall"],
            "Macro F1": r["macro_f1"],
            "Parameters (M)": r["parameters"] / 1e6,
            "Training Time (min)": r["train_time_min"]
        })
    df_summary = pd.DataFrame(summary_rows).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    
    best_model = df_summary.iloc[0]
    
    # Key Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_model['Model']}</div>
            <div class="metric-label">🏆 Top Performing Model</div>
        </div>
        """, unsafe_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_model['Test Accuracy']*100:.2f}%</div>
            <div class="metric-label">🎯 Highest Test Accuracy</div>
        </div>
        """, unsafe_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_model['Macro F1']:.4f}</div>
            <div class="metric-label">⚡ Best Macro F1-Score</div>
        </div>
        """, unsafe_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df_summary['Parameters (M)'].min():.2f}M</div>
            <div class="metric-label">📦 Smallest Parameter Count</div>
        </div>
        """, unsafe_html=True)
        
    st.markdown("<br>", unsafe_html=True)
    
    # Leaderboard Table
    st.subheader("📋 Comprehensive Model Performance Leaderboard")
    st.dataframe(
        df_summary.style.format({
            "Test Accuracy": "{:.4f}",
            "Best Val Accuracy": "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Parameters (M)": "{:.2f}",
            "Training Time (min)": "{:.1f}"
        }).background_gradient(subset=["Test Accuracy", "Macro F1"], cmap="Greens"),
        use_container_width=True
    )
    
    # Bar Chart Comparisons
    st.subheader("📈 Test Accuracy & F1-Score Comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=df_summary, x="Model", y="Test Accuracy", palette="viridis", ax=ax)
        ax.set_ylim(max(0, df_summary["Test Accuracy"].min() - 0.05), 1.0)
        ax.set_title("Test Accuracy by Model", fontsize=14, fontweight="bold")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", padding=3)
        plt.tight_layout()
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=df_summary, x="Model", y="Macro F1", palette="magma", ax=ax)
        ax.set_ylim(max(0, df_summary["Macro F1"].min() - 0.05), 1.0)
        ax.set_title("Macro F1-Score by Model", fontsize=14, fontweight="bold")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", padding=3)
        plt.tight_layout()
        st.pyplot(fig)

    # Per-class F1 Heatmap
    st.subheader("🎯 Per-Class F1 Score Breakdown")
    f1_dict = {name: r["per_class_f1"] for name, r in results_data.items()}
    df_f1 = pd.DataFrame(f1_dict).T
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(df_f1, annot=True, fmt=".4f", cmap="Blues", ax=ax, vmin=df_f1.values.min() - 0.02, vmax=1.0)
    ax.set_title("Per-Class F1 Score Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

elif nav_selection == "🔬 Live MRI Classifier":
    st.header("🔬 Live MRI Image Classifier & Multi-Model Inference")
    st.markdown("Upload a Brain MRI image scan or select a sample from the test set to predict tumor presence across all trained models!")
    
    input_source = st.radio("Choose Input Method", ["Upload Image", "Select Sample from Test Dataset"], horizontal=True)
    
    selected_img = None
    true_label = None
    
    if input_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload Brain MRI Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            selected_img = Image.open(uploaded_file)
    else:
        sample_class = st.selectbox("Select Tumor Category", CLASSES)
        sample_dir = os.path.join(TEST_DIR, sample_class)
        sample_files = os.listdir(sample_dir)
        selected_file = st.selectbox("Select Image File", sample_files[:30])
        if selected_file:
            selected_img_path = os.path.join(sample_dir, selected_file)
            selected_img = Image.open(selected_img_path)
            true_label = sample_class

    if selected_img is not None:
        col_img, col_pred = st.columns([1, 2])
        with col_img:
            st.image(selected_img, caption=f"Selected MRI Scan {f'(True: {true_label})' if true_label else ''}", use_column_width=True)
            
        with col_pred:
            st.subheader("⚡ Prediction Comparison Across All Models")
            
            model_names = ["CNN", "EfficientNetB0", "ResNet50", "MobileNetV2", "DenseNet121"]
            predictions = {}
            
            for m_name in model_names:
                model = load_trained_model(m_name)
                if model is not None:
                    probs = predict_image(model, selected_img)
                    pred_idx = np.argmax(probs)
                    predictions[m_name] = {
                        "label": CLASSES[pred_idx],
                        "confidence": probs[pred_idx],
                        "probs": probs
                    }
            
            if not predictions:
                st.error("No trained models found yet. Please wait for model training to complete.")
            else:
                pred_df_rows = []
                for m_name, res in predictions.items():
                    pred_df_rows.append({
                        "Model": m_name,
                        "Predicted Class": res["label"].upper(),
                        "Confidence": f"{res['confidence']*100:.2f}%",
                        "Glioma": f"{res['probs'][0]*100:.1f}%",
                        "Meningioma": f"{res['probs'][1]*100:.1f}%",
                        "No Tumor": f"{res['probs'][2]*100:.1f}%",
                        "Pituitary": f"{res['probs'][3]*100:.1f}%"
                    })
                st.dataframe(pd.DataFrame(pred_df_rows), use_container_width=True)
                
                # Detailed view for best model
                top_m = "EfficientNetB0" if "EfficientNetB0" in predictions else list(predictions.keys())[0]
                top_res = predictions[top_m]
                
                st.markdown(f"#### 🏆 Model Choice ({top_m}) Detailed Probabilities")
                for i, cls in enumerate(CLASSES):
                    prob_val = float(top_res["probs"][i])
                    st.write(f"**{cls.capitalize()}**: {prob_val*100:.2f}%")
                    st.progress(prob_val)

elif nav_selection == "📉 Training Curves & Confusion Matrices":
    st.header("📉 Training Curves & Confusion Matrices")
    
    if results_data is None:
        st.warning("⏳ Results JSON is loading...")
        st.stop()
        
    model_choice = st.selectbox("Select Model to Inspect", list(results_data.keys()))
    res = results_data[model_choice]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"📈 {model_choice} Training vs Validation Accuracy")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        epochs = range(1, len(res["history"]["accuracy"]) + 1)
        ax.plot(epochs, res["history"]["accuracy"], label="Train Accuracy", marker="o")
        ax.plot(epochs, res["history"]["val_accuracy"], label="Val Accuracy", marker="s")
        if res.get("fine_tune_start") is not None:
            ax.axvline(res["fine_tune_start"] + 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
    with col2:
        st.subheader(f"📉 {model_choice} Training vs Validation Loss")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        epochs = range(1, len(res["history"]["loss"]) + 1)
        ax.plot(epochs, res["history"]["loss"], label="Train Loss", marker="o", color="crimson")
        ax.plot(epochs, res["history"]["val_loss"], label="Val Loss", marker="s", color="orange")
        if res.get("fine_tune_start") is not None:
            ax.axvline(res["fine_tune_start"] + 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
    st.subheader(f"🧩 {model_choice} Confusion Matrix")
    cm = np.array(res["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"{model_choice} Confusion Matrix (Test Acc: {res['test_accuracy']*100:.2f}%)")
    plt.tight_layout()
    st.pyplot(fig)

elif nav_selection == "📁 Dataset Explorer":
    st.header("📁 Brain Tumor MRI Dataset Explorer")
    st.markdown("Explore the class distribution and sample MRI scans from the dataset.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Training Dataset Distribution (5,600 images)")
        train_counts = {c: len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in CLASSES}
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()), palette="Blues_d", ax=ax)
        ax.set_ylabel("Count")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)
        
    with c2:
        st.markdown("### Testing Dataset Distribution (1,600 images)")
        test_counts = {c: len(os.listdir(os.path.join(TEST_DIR, c))) for c in CLASSES}
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=list(test_counts.keys()), y=list(test_counts.values()), palette="Oranges_d", ax=ax)
        ax.set_ylabel("Count")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    st.subheader("🖼️ Random Sample Scans Across Categories")
    cols = st.columns(4)
    for i, c in enumerate(CLASSES):
        cls_dir = os.path.join(TRAIN_DIR, c)
        sample_img_name = os.listdir(cls_dir)[0]
        img = Image.open(os.path.join(cls_dir, sample_img_name))
        with cols[i]:
            st.image(img, caption=f"Class: {c.capitalize()}", use_column_width=True)
