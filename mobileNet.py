import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from PIL import Image
import os

# MobileNetV3 모델 정의
class SignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignClassifier, self).__init__()
        # pretrained 대신 weights 사용
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)  # 마지막 레이어 수정

    def forward(self, x):
        return self.model(x)

num_classes = 4  # 우회전, 천천히, 진입금지, 좌회전금지
model = SignClassifier(num_classes)

# 데이터셋 경로 설정
train_data_dir = 'C:/dataCNN_real/trainCNN_Nomal'
valid_data_dir = 'C:/dataCNN_real/validCNN'
test_data_dir = 'C:/dataCNN_real/testCNN'

# 데이터 전처리 및 로더 설정
# Train 데이터에 대해 정규화만 적용
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Valid와 Test 데이터에 대해 리사이즈와 정규화 적용
valid_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 리사이즈
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_data_dir, transform=valid_test_transform)
test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
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
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# 학습된 모델 저장
model_path = 'C:/ImageNet/model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

# 나중에 모델을 로드하여 사용
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 이미지 예측 함수 정의
def predict_image(image_path, model, transform, classes):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()

# 클래스 라벨 정의
classes = ['notEnter', 'notLeft', 'right', 'slow']

# 이미지 경로 설정 (다운로드 폴더 내의 이미지 경로)
image_folder = 'C:/dataCNN_real/testCNN'  # 실제 이미지 폴더 경로
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg'))]

# 각 이미지에 대해 예측 수행 및 출력
for image_path in image_paths:
    predicted_class = predict_image(image_path, model, valid_test_transform, classes)
    print(f'Image: {os.path.basename(image_path)} -> Predicted Class: {predicted_class}')

# 예측된 클래스와 실제 클래스를 비교하여 출력
predicted_classes = []
true_classes = []
for inputs, labels in test_loader:
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted_classes.extend(predicted.tolist())  # 예측된 클래스를 리스트에 추가
    true_classes.extend(labels.tolist())          # 실제 클래스를 리스트에 추가

for i, (predicted_class, true_class) in enumerate(zip(predicted_classes, true_classes)):
    predicted_sign = classes[predicted_class]
    true_sign = classes[true_class]
    print(f"테스트 이미지 {i+1}: 예측된 표지판 - {predicted_sign}, 실제 표지판 - {true_sign}")