import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn import metrics
import pandas as pd

from ViT_train import *

# 모델 경로 및 설정
model_path = 'vit/model_saved/train/vit/'
weight_file = 'model-066-0.874390-0.860614.pth'  # PyTorch 모델 가중치 파일
test_path = test_loader  # 테스트 이미지 폴더

# 테스트 데이터 변환
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 및 데이터로더
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(os.path.join(model_path, weight_file), map_location=device)
model.eval()

# 예측 수행
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# 결과 평가 및 저장
report = metrics.classification_report(y_true=true_labels, y_pred=pred_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
output_path = f'./output/report_test_{weight_file[:-4]}.csv'
report_df.to_csv(output_path, index=True, encoding='cp949')

print(report_df)