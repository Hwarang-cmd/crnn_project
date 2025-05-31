# app.py
import os
import io
import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
from utils.crnn import CRNN, decode
from utils.prepare import preprocess_digit_regions
from utils.inference import detect_sys_dia_pulse, detect_digits

app = Flask(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === โหลด YOLO ทั้ง 5 ตัว ===
yolo_models = {
    "pos1": YOLO("yolo_model/pos1.pt"),  # SYS/DIA/PULSE
    "pos2": YOLO("yolo_model/pos2.pt"),
    "digit1": YOLO("yolo_model/digit1.pt"),
    "digit2": YOLO("yolo_model/digit2.pt"),
    "digit3": YOLO("yolo_model/digit3.pt"),
}

# === โหลด CRNN ===
CHARACTERS = '0123456789'
NUM_CLASSES = len(CHARACTERS) + 1
crnn_model = CRNN(NUM_CLASSES).to(DEVICE)
crnn_model.load_state_dict(torch.load("crnn_model.pth", map_location=DEVICE))
crnn_model.eval()

@app.route('/')
def index():
    return "🩺 Blood Pressure Reader API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. Detect SYS/DIA/PULSE ตำแหน่ง
    positions = detect_sys_dia_pulse(img_bgr, [yolo_models["pos1"], yolo_models["pos2"]])

    # 2. Detect Digits ทั้งภาพ
    digit_results = detect_digits(img_bgr, [yolo_models["digit1"], yolo_models["digit2"], yolo_models["digit3"]])

    # 3. จับคู่ SYS/DIA/PULSE กับ digits และตัดรูป
    crops = preprocess_digit_regions(img_bgr, positions, digit_results)

    # 4. ส่งเข้า CRNN
    results = {}
    with torch.no_grad():
        for label, crop in crops.items():
            img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (128, 32))
            img_tensor = torch.tensor(img_resized / 255.0).unsqueeze(0).unsqueeze(0).float()
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.to(DEVICE)

            output = crnn_model(img_tensor)
            pred = decode(output)[0]
            results[label] = pred

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
