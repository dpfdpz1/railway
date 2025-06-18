import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from flask import Flask, request, jsonify
import pathlib
from PIL import Image

current_dir = pathlib.Path(__file__).parent.resolve()


class RailDataset(Dataset):
    def __init__(self, data_dir='data/train', transform=None, classes=None):
        self.data_dir = os.path.join(current_dir, data_dir)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录 '{self.data_dir}' 不存在。请确认路径是否正确。")

        self.classes = [d.name for d in pathlib.Path(self.data_dir).glob('*') if d.is_dir()]
        self.transform = transform
        self.all_paths = []  # 存储所有图像路径

        # 收集所有图像路径
        for category in self.classes:
            cat_dir = os.path.join(self.data_dir, category)
            for file in os.listdir(cat_dir):
                if file.lower().split('.')[-1] in ['png', 'jpg', 'jpeg']:
                    self.all_paths.append((os.path.join(cat_dir, file), category))

        if not self.all_paths:
            raise ValueError(f"未找到任何图像文件，路径: {self.data_dir}")

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        img_path, category = self.all_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法加载图像文件: {img_path}")

        # 将OpenCV图像从BGR转换为RGB，并转换为PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 应用预处理变换
        if self.transform:
            image_transformed = self.transform(pil_image)

        label = torch.tensor(self.classes.index(category), dtype=torch.long)
        return image_transformed, label


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    train_dataset = RailDataset(data_dir='data/train', transform=transform)
    test_dataset = RailDataset(data_dir='data/test', transform=transform)
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
except Exception as e:
    print(f"加载数据失败：{str(e)}")
    exit(1)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet50(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader)}')


def test_model(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += images.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {correct / total}')


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'result': 'No image uploaded'}), 400
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'result': 'No image selected'}), 400
        if image_file and image_file.filename.lower().split('.')[-1] in ['png', 'jpg', 'jpeg']:
            try:
                image_array = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)

                transform_test = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                image_transformed = transform_test(pil_image)
                image_transformed = image_transformed.unsqueeze(0)
                image_transformed = image_transformed.to(device)

                model.eval()
                with torch.no_grad():
                    output = model(image_transformed)
                    _, predicted = torch.max(output.data, 1)
                    result = train_dataset.classes[predicted.item()]
                    return jsonify({'disease': result})
            except Exception as e:
                return jsonify({'result': f'Error processing image: {str(e)}'}), 500
        else:
            return jsonify({'result': 'Invalid image file type'}), 400
    return jsonify({'result': 'Invalid request'}), 400


if __name__ == '__main__':
    data_root = os.path.join(current_dir, 'data')
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"数据目录已创建：{data_root}")
    print(f"请将训练集图像放置在：{train_dir}")
    print(f"请将测试集图像放置在：{test_dir}")

    train_model(model, train_loader, optimizer, criterion, epochs=10)

    app.run(debug=True)