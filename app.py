import torch
import cv2
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# ===== إنشاء الـ Flask app وتفعيل CORS =====
app = Flask(__name__)
CORS(app)  # السماح للـ frontend يرسل طلبات
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # حد أقصى للفيديو 50 ميجا

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_COUNT = 16
IMG_SIZE = 128

# ===== إعادة بناء الموديل =====
class SignLanguageModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4))
        )
        self.lstm = torch.nn.LSTM(64 * 4 * 4, 256, batch_first=True)
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.reshape(B, T, -1)
        _, (h, _) = self.lstm(feats)
        out = self.fc(h[-1])
        return out

# ===== تحميل الـ checkpoint =====
checkpoint = torch.load("sign_model.pth", map_location=DEVICE)
model = SignLanguageModel(num_classes=len(checkpoint["label2idx"])).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
idx2label = {v: k for k, v in checkpoint["label2idx"].items()}

# ===== دالة لتحويل فيديو لفريمات =====
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    # لو عدد الفريمات أقل من المطلوب، نكمل بصفر
    while len(frames) < FRAME_COUNT:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    frames = np.stack(frames, axis=0).transpose(0, 3, 1, 2) / 255.0
    return frames

# ===== دالة التنبؤ =====
def predict(video_path):
    frames = load_video_frames(video_path)
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(frames)
        pred_idx = output.argmax(dim=1).item()
    return idx2label[pred_idx]

# ===== API endpoint =====
@app.route("/predict", methods=["POST"])
def predict_api():
    if "video" not in request.files:
        return jsonify({"error": "اختر فيديو أولاً"}), 400
    video_file = request.files["video"]
    video_path = f"temp_{video_file.filename}"
    video_file.save(video_path)
    try:
        result = predict(video_path)
    finally:
        os.remove(video_path)  # إزالة الفيديو المؤقت بعد التنبؤ
    return jsonify({"translation": result})

# مفيش app.run() هنا — Hugging Face هيشغل السيرفر بنفسه
