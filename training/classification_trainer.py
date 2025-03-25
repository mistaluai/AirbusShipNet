import torch
import torch.nn as nn
from jinja2 import optimizer
from tqdm import tqdm
from copy import deepcopy

class ClassificationTrainer:
    def __init__(self, model, optimizer, loss_fn, dataloaders, device='cuda', model_path='./'):
        self.model = model
        self.criterion = loss_fn
        self.dataloader = dataloaders
        self.device = device
        self.scaler = torch.amp.GradScaler(device=device)
        self.model_path = model_path

        self.optimizer = optimizer

    def save_model(self, model, verbose):
        torch.save(model.state_dict(), self.model_path + "best_model.pth")
        if verbose:
            print(f"Saved model to {self.model_path}best_model.pth")

    def train(self, epochs, verbose=False):
        best_val_accuracy = 0
        best_model = None
        history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training step
            train_loss, train_accuracy = self.one_train_epoch(verbose=verbose)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)

            # Validation step
            val_loss, val_accuracy = self.one_eval_epoch(verbose=verbose)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            if verbose:
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            else:
                print(f"epoch {epoch + 1}/{epochs}\ntrain_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.2f}%\nval_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = deepcopy(self.model)

        if verbose:
            print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")

        self.save_model(model=best_model, verbose=verbose)

        return best_model, history

    def one_train_epoch(self, verbose):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        if verbose:
            dataloader = tqdm(self.dataloader['train'], desc='Training')
        else:
            dataloader = self.dataloader['train']

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device):
                outputs = self.model(inputs)
                outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            if verbose:
                accuracy = total_correct / total_samples * 100
                avg_loss = total_loss / total_samples
                dataloader.set_postfix(loss=avg_loss, accuracy=f"{accuracy:.2f}%")

        avg_loss = total_loss / len(dataloader.dataset)
        avg_accuracy = total_correct / total_samples * 100

        return avg_loss, avg_accuracy

    def one_eval_epoch(self, verbose):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        if verbose:
            dataloader = tqdm(self.dataloader['val'], desc='Evaluating')
        else:
            dataloader = self.dataloader['val']

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                with torch.amp.autocast(device_type=self.device):
                    outputs = self.model(inputs)
                    outputs = outputs.squeeze(1)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

                if verbose:
                    accuracy = total_correct / total_samples * 100
                    avg_loss = total_loss / total_samples
                    dataloader.set_postfix(loss=avg_loss, accuracy=f"{accuracy:.2f}%")

        avg_loss = total_loss / len(dataloader.dataset)
        avg_accuracy = total_correct / total_samples * 100

        return avg_loss, avg_accuracy

