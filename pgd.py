import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from PIL import Image
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MobileNetV2 모델 정의 및 수정
class SignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignClassifier, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)  # MobileNetV2로 변경
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

# 데이터셋 경로 설정
train_data_dir = 'C:/Users/phss1/Downloads/trainCNN_Nomal'
valid_data_dir = 'C:/Users/phss1/Downloads/validCNN_Nomal'
test_data_dir = 'C:/Users/phss1/Downloads/testCNN_Nomal'

# 데이터 전처리 및 로더 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 및 학습 설정
num_classes = 4  # 우회전, 천천히, 진입금지, 좌회전금지
model = SignClassifier(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# 학습된 모델 저장
model_path = 'C:/Users/phss1/Downloads/model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

# 나중에 모델을 로드하여 사용
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# PGD 공격 함수 정의
def pgd_attack(model, images, labels, eps, alpha, iters):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    loss = nn.CrossEntropyLoss()
    
    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

# PGD 공격 매개변수 설정
eps = 0.1
alpha = 1/255
iters = 10

# 클래스 라벨 정의
classes = ['notEnter', 'notLeft', 'right', 'slow']

# 데이터셋에 대해 PGD 공격 적용 및 평가
def evaluate_with_pgd(loader, loader_name):
    model.eval()
    total = 0
    correct = 0
    results = {cls: {'total': 0, 'correct': 0} for cls in classes}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, eps, alpha, iters)
        
        # 적대적 예제로 모델 평가
        outputs = model(adv_images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 각 클래스별로 정확도 기록
        for i in range(len(images)):
            true_class = classes[labels[i].item()]
            predicted_class = classes[predicted[i].item()]
            results[true_class]['total'] += 1
            if predicted_class == true_class:
                results[true_class]['correct'] += 1

    accuracy = 100 * correct / total
    print(f'{loader_name} - 전체 정확도: {accuracy:.2f}%')

    for cls, stats in results.items():
        class_accuracy = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f'\n실제 표지판: {cls} - 정확도: {class_accuracy:.2f}%')

print("Train set에서 PGD 공격 결과:")
evaluate_with_pgd(train_loader, "Train set")

print("Test set에서 PGD 공격 결과:")
evaluate_with_pgd(test_loader, "Test set")

print("Validation set에서 PGD 공격 결과:")
evaluate_with_pgd(valid_loader, "Validation set")
