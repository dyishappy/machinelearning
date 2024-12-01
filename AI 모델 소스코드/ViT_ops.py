# -*- coding:utf-8 -*-
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import pandas as pd
import matplotlib.pyplot as plt
from shutil import copyfile


# 시드 고정
seed = 2
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed)

class ImportData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        # Load the entire dataset with ImageFolder
        full_dataset = datasets.ImageFolder(root=self.data_path, transform=self.train_transform)
        
        # Calculate sizes for train and test splits
        total_size = len(full_dataset)
        train_size = int(total_size * 0.8)  # 80% for training
        test_size = total_size - train_size  # 20% for testing

        # Split the dataset
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        # Apply test transform separately
        test_dataset.dataset.transform = self.test_transform

        return train_dataset, test_dataset

class LoadModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def vit(self):
        """Load Vision Transformer (ViT) model"""
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=self.num_classes
        )
        return model

class FineTuning:
    def __init__(self, source_path, processed_path, model_name, epochs, device='cuda'):
        self.source_path = source_path
        self.processed_path = processed_path
        self.model_name = model_name
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load and preprocess data
        data = ImportData(source_path, processed_path)
        self.train_loader, self.val_loader, self.num_classes = data.get_data_loaders()

        # Load model
        model_loader = LoadModel(self.num_classes)
        self.model = model_loader.vit()
        self.model.to(self.device)

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        save_folder = f'./model_saved/{self.model_name}/'
        os.makedirs(save_folder, exist_ok=True)

        train_acc_history, val_acc_history = [], []
        train_loss_history, val_loss_history = [], []

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss, correct, total = 0.0, 0, 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            train_loss /= total
            train_acc = correct / total
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)

            # Validation phase
            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images).logits
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)

            val_loss /= total
            val_acc = correct / total
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            print(f"Epoch {epoch+1}/{self.epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save model checkpoint
            torch.save(self.model.state_dict(), save_folder + f'model_epoch_{epoch+1}.pth')

        # Save training history
        self.save_accuracy(train_acc_history, val_acc_history, train_loss_history, val_loss_history, save_folder)

    def save_accuracy(self, train_acc, val_acc, train_loss, val_loss, save_folder):
        epochs = list(range(1, len(train_acc) + 1))
        df = pd.DataFrame({
            'epoch': epochs,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        df.to_csv(save_folder + 'accuracy.csv', index=False)

        # Plot and save graphs
        plt.plot(epochs, train_acc, label='Train Accuracy')
        plt.plot(epochs, val_acc, label='Val Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(save_folder + 'accuracy.png')
        plt.cla()

        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Val Loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig(save_folder + 'loss.png')
        plt.cla()

# Usage
if __name__ == "__main__":
    source_path = '/data'  # 원본 데이터 경로
    processed_path = './processed_data'  # 전처리된 데이터 경로
    model_name = 'vit'
    epochs = 10

    fine_tune = FineTuning(source_path, processed_path, model_name, epochs)
    fine_tune.train()
