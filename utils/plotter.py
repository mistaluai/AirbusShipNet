import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_training_val(training_loss, val_loss, training_accuracy, val_accuracy):
    epochs = range(1, len(training_loss) + 1)

    fig, ax1 = plt.subplots()
    ax1.plot(epochs, training_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')

    plt.figure()
    ax1.plot(epochs, training_accuracy, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'g-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.show()

def plot_test(loss, metrics, confusion_data):
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    cm = confusion_matrix(confusion_data['true_labels'], confusion_data['predictions'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def display_image(image_batch: torch.Tensor, mask_batch: torch.Tensor,
                idx: int=2, alpha: float=0.4, cmap: str='gray'):
    """
    # Example usage:
    image, mask = next(iter(dataloader))
    display_image(image, mask)
    """
    # Select the sample image and its corresponding mask
    image = image_batch[idx].permute(1, 2, 0).cpu().numpy()
    mask = mask_batch[idx].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    
    # Display mask 
    axes[1].imshow(mask, cmap=cmap)
    axes[1].set_title("Mask")
    
    # Display overlay of mask on image
    axes[2].imshow(image)
    axes[2].imshow(mask, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay")
    
    for ax in axes:
        ax.axis('off')
    
    # Set a suptitle for the entire figure
    fig.suptitle(f"Sample {idx} with Mask Overlay", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()