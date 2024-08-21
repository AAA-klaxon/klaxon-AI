import cv2
import os
import numpy as np
import random

def resize_and_normalize(image, target_size=(224, 224)):
    # 이미지 리사이즈
    resized_image = cv2.resize(image, target_size)
    # Min-Max Scaling을 통해 이미지 정규화
    normalized_image = resized_image / 255.0
    return normalized_image

def augment_image(image):
    # 데이터 증강: 회전, 반전, 밝기 조정 등을 적용
    # 회전
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    augmented_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 수평 반전
    if random.random() > 0.5:
        augmented_image = cv2.flip(augmented_image, 1)

    return augmented_image

def process_images(input_dir, output_dir, target_count=None):
    # 입력 디렉토리에서 이미지 파일 리스트 가져오기
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    # 이미지 처리 및 저장
    processed_images = []
    
    # 1. 원본 이미지 리사이즈 및 정규화 후 저장
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        
        # 이미지 리사이즈 및 정규화
        normalized_image = resize_and_normalize(image)

        # 정규화된 이미지 저장
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, (normalized_image * 255).astype(np.uint8))
        processed_images.append(normalized_image)
        print(f"Saved resized and normalized image to {output_path}")

    if target_count is not None:
        # 2. 증강을 통해 이미지 수를 target_count로 맞춤
        current_count = len(processed_images)
        while current_count < target_count:
            for image in processed_images:
                if current_count >= target_count:
                    break
                augmented_image = augment_image(image)
                output_path = os.path.join(output_dir, f"aug_{current_count}.jpg")
                cv2.imwrite(output_path, (augmented_image * 255).astype(np.uint8))
                current_count += 1
                print(f"Saved augmented image to {output_path}")

# 입력 및 출력 디렉토리 구조
input_directories = {
    "train": {
        "notEnter": r"C:\dataCNN_real\trainCNN\notEnter",
        "notLeft": r"C:\dataCNN_real\trainCNN\notLeft",
        "right": r"C:\dataCNN_real\trainCNN\right",
        "slow": r"C:\dataCNN_real\trainCNN\slow"
    },
    "test": {
        "notEnter": r"C:\dataCNN_real\testCNN\notEnter",
        "notLeft": r"C:\dataCNN_real\testCNN\notLeft",
        "right": r"C:\dataCNN_real\testCNN\right",
        "slow": r"C:\dataCNN_real\testCNN\slow"
    },
    "valid": {
        "notEnter": r"C:\dataCNN_real\validCNN\notEnter",
        "notLeft": r"C:\dataCNN_real\validCNN\notLeft",
        "right": r"C:\dataCNN_real\validCNN\right",
        "slow": r"C:\dataCNN_real\validCNN\slow"
    }
}

output_directories = {
    "train": {
        "notEnter": r"C:\dataCNN_real\trainCNN_Nomal\notEnter",
        "notLeft": r"C:\dataCNN_real\trainCNN_Nomal\notLeft",
        "right": r"C:\dataCNN_real\trainCNN_Nomal\right",
        "slow": r"C:\dataCNN_real\trainCNN_Nomal\slow"
    },
    "test": {
        "notEnter": r"C:\dataCNN_real\testCNN_Nomal\notEnter",
        "notLeft": r"C:\dataCNN_real\testCNN_Nomal\notLeft",
        "right": r"C:\dataCNN_real\testCNN_Nomal\right",
        "slow": r"C:\dataCNN_real\testCNN_Nomal\slow"
    },
    "valid": {
        "notEnter": r"C:\dataCNN_real\validCNN_Nomal\notEnter",
        "notLeft": r"C:\dataCNN_real\validCNN_Nomal\notLeft",
        "right": r"C:\dataCNN_real\validCNN_Nomal\right",
        "slow": r"C:\dataCNN_real\validCNN_Nomal\slow"
    }
}

# 각 데이터셋(train, test, valid) 및 클래스(notEnter, notLeft, right, slow)별로 처리 수행
for dataset, classes in input_directories.items():
    for sign_type, input_directory in classes.items():
        output_directory = output_directories[dataset][sign_type]
        print(f"Processing {dataset} dataset for {sign_type} signs...")
        if dataset == "train":  # Train 데이터셋에만 증강 적용
            process_images(input_directory, output_directory, target_count=200)
        else:
            process_images(input_directory, output_directory)