import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import timm
import seaborn as sns
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import time
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

class Config:
    DATA_ROOT = "fish_dataset"  
    IMG_SIZE = 384  
    BATCH_SIZE = 16  
    NUM_WORKERS = 4  
    
    MODEL_NAME = "tf_efficientnetv2_m"
    PRETRAINED = False
    
    EPOCHS = 50
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 2
    
    MIXED_PRECISION = True
    
    VALID_SIZE = 0.2
    
    AUGMENTATION_LEVEL = "strong"
    
    CHECKPOINT_DIR = "checkpoints"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

def get_augmentations(aug_level="medium"):
    if aug_level == "light":
        train_transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    elif aug_level == "medium":
        train_transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    elif aug_level == "strong":
        train_transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.7, brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(p=0.7, hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
            A.ShiftScaleRotate(p=0.7, shift_limit=0.1, scale_limit=0.2, rotate_limit=30),
            A.OneOf([
                A.MotionBlur(p=0.7),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.5),
                A.GridDistortion(p=0.5),
                A.ElasticTransform(p=0.5),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=8, min_width=8, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"Unsupported augmentation level: {aug_level}")
    
    valid_transform = A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
    
    return train_transform, valid_transform

class FishDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

def prepare_data():
    print("正在准备数据...")
    
    class_folders = sorted([d for d in os.listdir(Config.DATA_ROOT) if os.path.isdir(os.path.join(Config.DATA_ROOT, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_folders)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(class_folders)}
    
    image_paths = []
    labels = []
    
    for class_name in class_folders:
        class_dir = os.path.join(Config.DATA_ROOT, class_name)
        class_idx = class_to_idx[class_name]
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_idx)
    
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        image_paths, labels, test_size=Config.VALID_SIZE, stratify=labels, random_state=42
    )
    
    train_transform, valid_transform = get_augmentations(Config.AUGMENTATION_LEVEL)
    
    train_dataset = FishDataset(train_paths, train_labels, train_transform)
    valid_dataset = FishDataset(valid_paths, valid_labels, valid_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"找到 {len(class_folders)} 种鱼类")
    print(f"训练样本数: {len(train_paths)}")
    print(f"验证样本数: {len(valid_paths)}")
    print(f"鱼类种类: {class_folders}")
    
    dataset_info = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_folders)
    }
    
    return train_loader, valid_loader, dataset_info

class FishClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(FishClassifier, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.model(x)

def train_epoch(model, loader, optimizer, criterion, scheduler, device, scaler=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="训练中")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if Config.MIXED_PRECISION and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': train_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })
    
    return train_loss/len(loader), 100.*correct/total

def valid_epoch(model, loader, criterion, device):
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="验证中")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            pbar.set_postfix({
                'loss': valid_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
    
    return valid_loss/len(loader), 100.*correct/total, all_targets, all_predictions

def get_scheduler(optimizer):
    if Config.SCHEDULER == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=Config.EPOCHS,
            eta_min=Config.MIN_LR
        )
    elif Config.SCHEDULER == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
    elif Config.SCHEDULER == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.LEARNING_RATE,
            epochs=Config.EPOCHS,
            steps_per_epoch=len(train_loader)
        )
    else:
        return None

def train_model(model, train_loader, valid_loader, dataset_info):
    print("开始训练模型...")
    
    device = Config.DEVICE
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = get_scheduler(optimizer)
    
    scaler = torch.cuda.amp.GradScaler() if Config.MIXED_PRECISION and torch.cuda.is_available() else None
    
    best_acc = 0
    best_epoch = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }
    
    for epoch in range(Config.EPOCHS):
        print(f"\n周期 {epoch+1}/{Config.EPOCHS}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, 
            scheduler if Config.SCHEDULER == "OneCycleLR" else None,
            device, scaler
        )
        
        valid_loss, valid_acc, targets, predictions = valid_epoch(
            model, valid_loader, criterion, device
        )
        
        if scheduler is not None and Config.SCHEDULER == "ReduceLROnPlateau":
            scheduler.step(valid_loss)
        elif scheduler is not None and Config.SCHEDULER != "OneCycleLR":
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        
        print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {valid_loss:.4f} | 验证准确率: {valid_acc:.2f}%")
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_acc': best_acc,
                'dataset_info': dataset_info
            }, os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'))
            
            print(f"最佳模型已保存! (准确率: {best_acc:.2f}%)")
            
            idx_to_class = dataset_info['idx_to_class']
            class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
            
            print("\n分类报告:")
            print(classification_report(targets, predictions, target_names=class_names))
    
    torch.save({
        'epoch': Config.EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'last_acc': valid_acc,
        'dataset_info': dataset_info
    }, os.path.join(Config.CHECKPOINT_DIR, 'final_model.pth'))
    
    print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}% (周期 {best_epoch+1})")
    
    return model, history, best_epoch

def evaluate_model(model, valid_loader, dataset_info):
    print("评估模型...")
    device = Config.DEVICE
    criterion = nn.CrossEntropyLoss()
    
    valid_loss, valid_acc, all_targets, all_predictions = valid_epoch(
        model, valid_loader, criterion, device
    )
    
    idx_to_class = dataset_info['idx_to_class']
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    print("\n分类报告:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_targets, all_predictions)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('归一化混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存为 'confusion_matrix.png'")
    
    return valid_acc

def plot_training_history(history):
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='训练准确率', marker='o')
    plt.plot(history['valid_acc'], label='验证准确率', marker='o')
    plt.title('模型准确率')
    plt.ylabel('准确率 (%)')
    plt.xlabel('周期')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='训练损失', marker='o')
    plt.plot(history['valid_loss'], label='验证损失', marker='o')
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('周期')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("训练历史图已保存为 'training_history.png'")

def predict_image(image_path, model=None, dataset_info=None):
    if model is None or dataset_info is None:
        if not os.path.exists(os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')):
            print("错误: 未找到模型文件，请先训练模型")
            return
        
        checkpoint = torch.load(
            os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'),
            map_location=Config.DEVICE
        )
        
        dataset_info = checkpoint['dataset_info']
        num_classes = dataset_info['num_classes']
        
        model = FishClassifier(Config.MODEL_NAME, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(Config.DEVICE)
    
    model.eval()
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    _, valid_transform = get_augmentations()
    transformed = valid_transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(Config.DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
    idx_to_class = dataset_info['idx_to_class']
    predicted_idx = output.argmax(1).item()
    predicted_class = idx_to_class[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f'预测: {predicted_class} ({confidence:.2f}%)')
    plt.axis('off')
    plt.savefig('prediction_result.png')
    plt.show()
    
    print("\n各鱼类预测概率:")
    probs_dict = {}
    for idx, prob in enumerate(probabilities):
        class_name = idx_to_class[idx]
        prob_percent = prob.item() * 100
        probs_dict[class_name] = prob_percent
        print(f"{class_name}: {prob_percent:.2f}%")
    
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    print("\n前3个预测:")
    for i, (class_name, prob) in enumerate(sorted_probs[:3]):
        print(f"{i+1}. {class_name}: {prob:.2f}%")
    
    return predicted_class, confidence

def create_ensemble_model(num_classes, models_list=None):
    if models_list is None:
        models_list = [
            "tf_efficientnetv2_s",
            "tf_efficientnetv2_m",
            "tf_efficientnetv2_l",
            "resnet50",
            "vit_base_patch16_384"
        ]
    
    models = []
    for model_name in models_list:
        model = FishClassifier(model_name, num_classes)
        models.append(model)
    
    return models

def ensemble_predict(image_path, models, dataset_info):
    device = Config.DEVICE
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    _, valid_transform = get_augmentations()
    transformed = valid_transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    all_probs = []
    
    for model in models:
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            all_probs.append(probs.cpu().numpy())
    
    avg_probs = np.mean(all_probs, axis=0)
    
    idx_to_class = dataset_info['idx_to_class']
    predicted_idx = np.argmax(avg_probs)
    predicted_class = idx_to_class[predicted_idx]
    confidence = avg_probs[predicted_idx] * 100
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f'集成预测: {predicted_class} ({confidence:.2f}%)')
    plt.axis('off')
    plt.savefig('ensemble_prediction.png')
    plt.show()
    
    print("\n各鱼类预测概率 (集成):")
    probs_dict = {}
    for idx, prob in enumerate(avg_probs):
        class_name = idx_to_class[idx]
        prob_percent = prob * 100
        probs_dict[class_name] = prob_percent
        print(f"{class_name}: {prob_percent:.2f}%")
    
    return predicted_class, confidence

def main():
    train_loader, valid_loader, dataset_info = prepare_data()
    
    num_classes = dataset_info['num_classes']
    model = FishClassifier(Config.MODEL_NAME, num_classes, pretrained=Config.PRETRAINED)
    
    model, history, best_epoch = train_model(model, train_loader, valid_loader, dataset_info)
    
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    best_acc = evaluate_model(model, valid_loader, dataset_info)
    
    plot_training_history(history)
    
    print("\n训练和评估完成！")
    
    print("\n示例预测:")
    image_batch, _ = next(iter(valid_loader))
    sample_img = TF.to_pil_image(image_batch[0])
    
    sample_path = 'sample_image.jpg'
    sample_img.save(sample_path)
    
    predict_image(sample_path, model, dataset_info)

def load_and_predict(image_path):
    if not os.path.exists(os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')):
        print("错误: 未找到模型文件，请先训练模型")
        return
    
    checkpoint = torch.load(os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'), map_location=Config.DEVICE)
    dataset_info = checkpoint['dataset_info']
    num_classes = dataset_info['num_classes']
    
    model = FishClassifier(Config.MODEL_NAME, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    
    predicted_class, confidence = predict_image(image_path, model, dataset_info)
    print(f"\n预测结果: {predicted_class}，置信度: {confidence:.2f}%")

if __name__ == "__main__":
    main()

# R8前置：分支2修改
