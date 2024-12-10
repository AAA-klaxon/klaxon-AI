import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models 

# SignClassifier 클래스 정의
class SignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignClassifier, self).__init__()
        # MobileNetV3 모델을 가져와 수정
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Adaptive Average Pooling 추가하여 고정된 크기의 출력 얻기
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 기존의 classifier를 제거하고 새로운 다층 구조 추가
        self.model.classifier = nn.Sequential(
            nn.Linear(576, 128),  # Adaptive Pooling 후 크기 576을 사용
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout 추가
            nn.Linear(128, 64),  # 두 번째 레이어
            nn.ReLU(),
            nn.Linear(64, num_classes)  # 마지막 레이어
        )

    def forward(self, x):
        x = self.model.features(x)  # 특징 추출 부분
        x = self.model.avgpool(x)   # Adaptive Pooling으로 크기 고정
        x = torch.flatten(x, 1)  # (batch_size, 576)으로 평탄화
        x = self.model.classifier(x)  # 새로운 classifier 통과
        return x

# 모델 경로 설정 (Google Drive)
model_path = 'mobileNet.pth'

# 모델 로드 함수
def load_model(model_path):
    model = SignClassifier(num_classes=4)  # 클래스 수에 맞게 모델 정의
    model.load_state_dict(torch.load(model_path))  # 학습된 가중치 로드
    model.eval()
    return model

# 모델 로드
model = load_model(model_path)

# 클래스 정의 (숫자에 해당하는 클래스 이름)
classes = ['notEnter', 'notLeft', 'right', 'slow']

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력 크기에 맞춤
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 사용자로부터 단일 이미지 입력 받기
image_file = 'aug_398.jpg'  # 사용할 이미지 파일의 경로
img = Image.open(image_file).convert('RGB')  # 이미지 파일 열기 및 RGB 변환
processed_image = transform(img).unsqueeze(0)  # 전처리 및 배치 차원 추가

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
eps = 1.0
alpha = 10/255
iters = 30

# 단일 이미지에 대한 레이블 설정 (여기서는 0으로 설정, 필요에 따라 수정)
labels = torch.tensor([0])  # 예: 'notEnter'에 대한 레이블

# 공격 실행
adv_images = pgd_attack(model, processed_image, labels, eps, alpha, iters)

# 적대적 이미지 예측 함수
def predict(model, images):
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 예측 클래스 인덱스
    return predicted

# 공격된 이미지에 대한 라벨 예측
predicted_label = predict(model, adv_images)

# 결과 출력
def show_images(images, title="", predictions=None):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        if predictions is not None:
            plt.title(f"Predicted: {classes[predictions[i].item()]}")
    plt.suptitle(title)
    plt.show()

# 원본 이미지와 공격된 이미지 출력
show_images(adv_images, title="Adversarial Image", predictions=predicted_label)
