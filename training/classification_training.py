from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn as nn
from data.classification_stage.data_processor import DataProcessor
from data.classification_stage.dataset import ClassificationDataset
from evaluation.classification_evaluator import ClassificationEvaluator
from loss_fn.BCEWithLogits_mixup import BCEWithLogitsMixup
from models.classification_stage.resnet34 import ResNet34
from training.classification_trainer import ClassificationTrainer
from utils.mixup_dataloader_initializer import MixupDataLoader
from utils.plotter import plot_training_val, plot_test
from utils.random_seed import set_seed
import torch
import torch.optim as optim

##Seed
seed = 2005
set_seed(seed=seed)

## Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Data processor
dataset_root = '/kaggle/input/airbus-ship-detection'
csv_file = '/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv'
dp = DataProcessor(csv_file=csv_file, dataset_root=dataset_root, seed=seed)
train_data, val_data, test_data = dp.get_data()

## Datasets
train_aug = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),  # Horizontal flipping with 50% probability
        v2.RandomRotation(degrees=15),  # Random rotation within a range of ±15 degrees
        v2.RandomResizedCrop(size=(224), scale=(0.8, 1.0)),  # Random zoom by cropping
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, etc.
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

train_dataset = ClassificationDataset(train_data, transform=train_aug)
val_dataset = ClassificationDataset(val_data)
test_dataset = ClassificationDataset(test_data)

## Dataloaders
batch_size = 450
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

## Criterion
pos_weight = train_dataset.get_pos_weight()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

## Model
model = ResNet34().to(device)

## Optimizer
early_cnn_lr = 3e-4
later_cnn_lr = 1.5e-3
fc_layer_lr = 3e-3
weight_decay = 0

early_cnn_params = list(model.get_layer('conv1').parameters()) + \
                   list(model.get_layer('layer1').parameters())
later_cnn_params = list(model.get_layer('layer2').parameters()) + \
                   list(model.get_layer('layer3').parameters()) + \
                   list(model.get_layer('layer4').parameters())
fc_layer_params = list(model.get_layer('fc').parameters())

# Initialize the Adam optimizer with different learning rates
optimizer = optim.Adam(
    params=[
    {'params': early_cnn_params, 'lr': early_cnn_lr},
    {'params': later_cnn_params, 'lr': later_cnn_lr},
    {'params': fc_layer_params, 'lr': fc_layer_lr},
],
    lr=early_cnn_lr,
    weight_decay=weight_decay)

## Trainer
trainer = ClassificationTrainer(model=model, optimizer=optimizer, loss_fn=criterion, dataloaders=dataloaders, device=device)

best_model, history = trainer.train(epochs=10)

## plotting
train_loss, val_loss, train_acc, val_acc = history['train_loss'], history['val_loss'], history['train_accuracy'], history['val_accuracy']
plot_training_val(train_loss, val_loss, train_acc, val_acc)

## Evaluation
evaluator = ClassificationEvaluator(best_model, test_loader, criterion, device)
avg_loss, metrics_dict, confusion_data = evaluator.evaluate()

## plotting
plot_test(avg_loss, metrics_dict, confusion_data)