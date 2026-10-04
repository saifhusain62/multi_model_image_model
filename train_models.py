import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set CPU Threads
num_cpus = min(os.cpu_count() or 4, 4)
torch.set_num_threads(num_cpus)

# Set Random Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Using device: {device} | CPU threads: {num_cpus}", flush=True)

# Paths
TRAIN_DIR = r"c:\Users\SAIF-PC\Downloads\archive\Training"
TEST_DIR = r"c:\Users\SAIF-PC\Downloads\archive\Testing"
OUTPUT_DIR = r"c:\Users\SAIF-PC\Downloads\archive\artifacts_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (112, 112)
BATCH_SIZE = 128
NUM_CLASSES = 4

# Load Dataframes
classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

def build_df(dir_path):
    rows = []
    for cls in classes:
        cls_dir = os.path.join(dir_path, cls)
        if not os.path.isdir(cls_dir):
            continue
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            rows.append({
                "image_path": img_path,
                "label": cls,
                "label_idx": class_to_idx[cls]
            })
    return pd.DataFrame(rows)

train_full_df = build_df(TRAIN_DIR)
test_df = build_df(TEST_DIR)

train_df, val_df = train_test_split(
    train_full_df, test_size=0.2, stratify=train_full_df["label_idx"], random_state=SEED
)

print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}", flush=True)

# Pre-load & Pre-normalize Images into Memory Tensors
prep_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preload_tensors(df):
    t0 = time.time()
    images = []
    labels = []
    for idx, row in df.iterrows():
        img = Image.open(row["image_path"]).convert("RGB")
        tensor = prep_transform(img)
        images.append(tensor)
        labels.append(row["label_idx"])
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    print(f"Preloaded {len(df)} images into RAM in {time.time() - t0:.2f}s", flush=True)
    return images_tensor, labels_tensor

print("Preloading Train, Validation, and Test datasets into memory...", flush=True)
train_imgs, train_lbls = preload_tensors(train_df)
val_imgs, val_lbls = preload_tensors(val_df)
test_imgs, test_lbls = preload_tensors(test_df)

train_loader = DataLoader(TensorDataset(train_imgs, train_lbls), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_imgs, val_lbls), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(test_imgs, test_lbls), batch_size=BATCH_SIZE, shuffle=False)

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

class HeadClassifier(nn.Module):
    def __init__(self, in_features, num_classes=4):
        super(HeadClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def get_backbone_and_features(model_name):
    if model_name == "EfficientNetB0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, in_feat
    elif model_name == "ResNet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_feat = m.fc.in_features
        m.fc = nn.Identity()
        return m, in_feat
    elif model_name == "MobileNetV2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, in_feat
    elif model_name == "DenseNet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_feat = m.classifier.in_features
        m.classifier = nn.Identity()
        return m, in_feat

class TransferModel(nn.Module):
    def __init__(self, backbone, head):
        super(TransferModel, self).__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        feat = self.backbone(x)
        if len(feat.shape) > 2:
            feat = torch.flatten(feat, 1)
        return self.head(feat)

@torch.no_grad()
def extract_feature_matrix(backbone, imgs):
    backbone.eval()
    loader = DataLoader(TensorDataset(imgs), batch_size=BATCH_SIZE, shuffle=False)
    feats = []
    for (b_imgs,) in loader:
        out = backbone(b_imgs)
        if len(out.shape) > 2:
            out = torch.flatten(out, 1)
        feats.append(out)
    return torch.cat(feats, dim=0)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate_test(model, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(labels.numpy())
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

def run_experiment(name, is_transfer=True):
    print(f"\n==========================================", flush=True)
    print(f" Starting Experiment: {name}", flush=True)
    print(f"==========================================", flush=True)
    
    start_time = time.time()
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    best_val_acc = 0.0
    best_weights_path = os.path.join(OUTPUT_DIR, f"{name}_best.pt")
    
    if not is_transfer:
        model = CustomCNN(NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 5
        
        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            v_loss, v_acc = evaluate_epoch(model, val_loader, criterion)
            ep_time = time.time() - ep_start
            
            history["accuracy"].append(float(t_acc))
            history["val_accuracy"].append(float(v_acc))
            history["loss"].append(float(t_loss))
            history["val_loss"].append(float(v_loss))
            
            print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.2f}s] | Train Acc: {t_acc:.4f} Loss: {t_loss:.4f} | Val Acc: {v_acc:.4f} Loss: {v_loss:.4f}", flush=True)
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save(model.state_dict(), best_weights_path)
    else:
        backbone, in_feat = get_backbone_and_features(name)
        head = HeadClassifier(in_feat, NUM_CLASSES)
        
        print(f"--- Feature Extraction Caching for {name} ---", flush=True)
        t_feat_start = time.time()
        train_f = extract_feature_matrix(backbone, train_imgs)
        val_f = extract_feature_matrix(backbone, val_imgs)
        test_f = extract_feature_matrix(backbone, test_imgs)
        print(f"Features extracted in {time.time() - t_feat_start:.2f}s", flush=True)
        
        feat_train_loader = DataLoader(TensorDataset(train_f, train_lbls), batch_size=BATCH_SIZE, shuffle=True)
        feat_val_loader = DataLoader(TensorDataset(val_f, val_lbls), batch_size=BATCH_SIZE, shuffle=False)
        feat_test_loader = DataLoader(TensorDataset(test_f, test_lbls), batch_size=BATCH_SIZE, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(head.parameters(), lr=1e-3)
        head_epochs = 8
        
        print(f"--- Training Classification Head ({name}) ---", flush=True)
        for epoch in range(1, head_epochs + 1):
            ep_start = time.time()
            t_loss, t_acc = train_one_epoch(head, feat_train_loader, criterion, optimizer)
            v_loss, v_acc = evaluate_epoch(head, feat_val_loader, criterion)
            ep_time = time.time() - ep_start
            
            history["accuracy"].append(float(t_acc))
            history["val_accuracy"].append(float(v_acc))
            history["loss"].append(float(t_loss))
            history["val_loss"].append(float(v_loss))
            
            print(f"Epoch {epoch:02d}/{head_epochs:02d} [{ep_time:.2f}s] | Train Acc: {t_acc:.4f} Loss: {t_loss:.4f} | Val Acc: {v_acc:.4f} Loss: {v_loss:.4f}", flush=True)
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                
        # Save complete TransferModel (Backbone + Head)
        full_model = TransferModel(backbone, head).to(device)
        torch.save(full_model.state_dict(), best_weights_path)
        model = full_model
        test_loader_to_use = test_loader

    train_time = time.time() - start_time
    
    # Evaluation
    if is_transfer:
        y_true, y_pred, y_prob = evaluate_test(model, test_loader)
    else:
        if os.path.exists(best_weights_path):
            model.load_state_dict(torch.load(best_weights_path))
        y_true, y_pred, y_prob = evaluate_test(model, test_loader)
        
    test_acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    result = {
        "name": name,
        "history": history,
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(best_val_acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "per_class_f1": {c: float(report[c]["f1-score"]) for c in classes},
        "per_class_precision": {c: float(report[c]["precision"]) for c in classes},
        "per_class_recall": {c: float(report[c]["recall"]) for c in classes},
        "confusion_matrix": cm.tolist(),
        "class_names": classes,
        "parameters": num_params,
        "train_time_min": train_time / 60,
        "fine_tune_start": None
    }
    
    print(f"\n>>>> {name} TEST ACCURACY: {test_acc:.4f} | MACRO F1: {report['macro avg']['f1-score']:.4f} <<<<\n", flush=True)
    return result

if __name__ == "__main__":
    experiments = [
        ("CNN", False),
        ("EfficientNetB0", True),
        ("ResNet50", True),
        ("MobileNetV2", True),
        ("DenseNet121", True)
    ]
    
    all_results = {}
    for name, is_transfer in experiments:
        res = run_experiment(name, is_transfer=is_transfer)
        all_results[name] = res
        
    # Save JSON summary
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
        
    # Build Comparison DataFrame
    summary_rows = []
    for r in all_results.values():
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
        
    summary_df = pd.DataFrame(summary_rows).sort_values("Test Accuracy", ascending=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    
    print("\n================ FINAL MODEL COMPARISON ================", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print("========================================================", flush=True)
