import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

class ClassificationEvaluator:
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def evaluate(self, verbose=False):
        self.model.eval()
        val_loss = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []

        dataloader = tqdm(self.dataloader, desc="Testing") if verbose else self.dataloader

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device):
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)
                    val_loss += loss.item()

                    # Predictions
                    probabilities = torch.sigmoid(logits)  # Binary classification probability
                    predictions = (probabilities >= 0.5).float()

                    # Store labels, predictions, and probabilities
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                    if verbose:
                        dataloader.set_postfix(loss=val_loss)

        # Calculate average loss
        avg_loss = val_loss / len(self.dataloader)


        # Convert lists to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        metrics_dict = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='binary', zero_division=0),
            'recall': recall_score(all_labels, all_predictions, average='binary', zero_division=0),
            'f1': f1_score(all_labels, all_predictions, average='binary', zero_division=0),
            'per_class': {
                        'precision': precision_score(all_labels, all_predictions, average=None, zero_division=0),
                        'recall': recall_score(all_labels, all_predictions, average=None, zero_division=0),
                        'f1': f1_score(all_labels, all_predictions, average=None, zero_division=0)
                    }
                }

        # Confusion matrix data
        confusion_data = {
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'true_labels': all_labels,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }

        if verbose:
            print(f"\nAverage Loss: {avg_loss:.4f}")
            print(f"Metrics: {metrics_dict}")
            print(f"Confusion Matrix:\n{confusion_data['confusion_matrix']}")

        return avg_loss, metrics_dict, confusion_data