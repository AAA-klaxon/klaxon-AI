import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from flask import Flask, request, jsonify, render_template
import base64

# Flask 앱 생성
app = Flask(__name__)

# SignClassifier 클래스 정의
class SignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignClassifier, self).__init__()
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x

# 모델 경로 설정
model_path = 'mobileNet.pth'

# 모델 로드 함수
def load_model(model_path):
    try:
        model = SignClassifier(num_classes=4)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# 모델 로드
model = load_model(model_path)

# 클래스 정의
classes = ['notEnter', 'notLeft', 'right', 'slow']

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# PGD 공격 함수 정의
def pgd_attack(model, images, labels, eps, alpha, iters):
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss = nn.CrossEntropyLoss()

    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

# PGD 공격 매개변수 설정
eps = 1.8
alpha = 3/255
iters = 13

@app.route('/attack', methods=['POST'])
def attack():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    # 사용자로부터 이미지 파일 가져오기
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    processed_image = transform(img).unsqueeze(0)

    # 원본 이미지 Base64 인코딩
    buffered_original = io.BytesIO()
    img.save(buffered_original, format="PNG")
    original_image_base64 = base64.b64encode(buffered_original.getvalue()).decode()

    # 원본 이미지에 대해 예측 수행
    with torch.no_grad():
        outputs_original = model(processed_image)
        _, predicted_original = torch.max(outputs_original, 1)
        labels = predicted_original  # 모델이 예측한 레이블 사용

    # 공격 실행
    adv_images = pgd_attack(model, processed_image, labels, eps, alpha, iters)

    # 공격된 이미지를 NumPy 배열로 변환 후 Base64 인코딩
    adv_images_np = adv_images.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_images_pil = Image.fromarray((adv_images_np * 255).astype('uint8'))
    buffered_adv = io.BytesIO()
    adv_images_pil.save(buffered_adv, format="PNG")
    adv_images_base64 = base64.b64encode(buffered_adv.getvalue()).decode()

    # 적대적 이미지에 대한 예측 수행
    with torch.no_grad():
        outputs_adversarial = model(adv_images)
        _, predicted_adversarial = torch.max(outputs_adversarial, 1)

    # 결과 반환
    return jsonify({
        'adversarial_image': adv_images_base64,
        'adversarial_label': classes[predicted_adversarial.item()],
        'original_label': classes[predicted_original.item()],
        'original_image': original_image_base64
    })

@app.route('/attackraspi', methods=['POST'])
def attack_raspi():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    # 사용자로부터 이미지 파일 가져오기
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    processed_image = transform(img).unsqueeze(0)

    # 원본 이미지에 대해 예측 수행
    with torch.no_grad():
        outputs_original = model(processed_image)
        _, predicted_original = torch.max(outputs_original, 1)

    # 공격 실행
    adv_images = pgd_attack(model, processed_image, predicted_original, eps, alpha, iters)

    # 적대적 이미지에 대한 예측 수행
    with torch.no_grad():
        outputs_adversarial = model(adv_images)
        _, predicted_adversarial = torch.max(outputs_adversarial, 1)

    # 결과 반환
    return jsonify({
        'adversarial_label': classes[predicted_adversarial.item()],
        'original_label': classes[predicted_original.item()]
    })


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)