import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score
from prettytable import PrettyTable
import numpy as np
import os

# Eğitim ve doğrulama için sınıf
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device,configuration):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.configuration = configuration

        self.train_metrics_df = pd.DataFrame()
        self.val_metrics_df = pd.DataFrame()
        self.best_val_loss_ = float('inf')  # En düşük doğrulama hatası için başlangıç değeri
        folder_path="weights"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")
        
    def train(self, num_epochs):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    pbar.update(1)
            train_loss = running_loss / len(self.train_loader)
            train_metrics = self.compute_metrics(all_labels, all_preds, 'train')
            self.log_metrics(epoch + 1, train_loss, train_metrics, 'Train')
            val_loss=self.validate(epoch + 1, num_epochs)
            if val_loss < self.best_val_loss_:
                self.best_val_loss_ = val_loss
                self.save_model_weights(epoch)  # En düşük doğrulama hatasına sahip olan ağırlıkları kaydet
        print("Best Validation Loss : ",self.best_val_loss_)
        
        # Save metrics to Excel
        with pd.ExcelWriter(f'{self.configuration}_metrics.xlsx') as writer:
            self.train_metrics_df.to_excel(writer, sheet_name='Train Metrics')
            self.val_metrics_df.to_excel(writer, sheet_name='Validation Metrics')

    def validate(self, epoch, num_epochs):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with tqdm(total=len(self.val_loader), desc=f"Validation {epoch}/{num_epochs}", unit="batch") as pbar:
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    pbar.update(1)
        val_loss = running_loss / len(self.val_loader)
        val_metrics = self.compute_metrics(all_labels, all_preds, 'val')
        self.log_metrics(epoch, val_loss, val_metrics, 'Validation')
        return val_loss
    def save_model_weights(self, epoch):
        torch.save(self.model.state_dict(), f'weights/best_model_weights_epoch_{self.configuration}_{epoch}_{self.best_val_loss_}.pt')
    def load_best_model_weights(self):
        self.model.load_state_dict(torch.load(f'weights/best_model_weights_epoch_{self.configuration}_{epoch}_{self.best_val_loss_}.pt'))
    def compute_metrics(self, labels, preds, stage):
        accuracy = np.mean(np.array(labels) == np.array(preds))
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc, 'kappa': kappa}

    def log_metrics(self, epoch, loss, metrics, stage):
        metrics_data = {
            'Epoch': epoch,
            'Loss': loss,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'MCC': metrics['mcc'],
            'Cohen\'s Kappa': metrics['kappa']
        }
        if stage == 'Train':
            self.train_metrics_df = pd.concat([self.train_metrics_df, pd.DataFrame([metrics_data])], ignore_index=True)
        else:
            self.val_metrics_df = pd.concat([self.val_metrics_df, pd.DataFrame([metrics_data])], ignore_index=True)
        
        table = PrettyTable()
        table.field_names = ["Epoch", "Stage", "Loss", "Accuracy", "Precision", "Recall", "F1-Score", "MCC", "Cohen's Kappa"]
        table.add_row([
            epoch, stage, f"{loss:.4f}", 
            f"{metrics['accuracy']:.4f}", f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", 
            f"{metrics['f1']:.4f}", f"{metrics['mcc']:.4f}", f"{metrics['kappa']:.4f}"
        ])
        print(table)
    @property
    def best_val_loss(self):
        return self.best_val_loss_