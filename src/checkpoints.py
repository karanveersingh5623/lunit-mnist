import os
import torch
import mlflow

def save_checkpoint(model, optimizer, epoch, path="model_checkpoint.pth", mlflow_run_id=None):
    """Save the model and optimizer state, and optionally register the model in MLflow registry."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    checkpoint_dir = os.path.dirname(path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint_epoch{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at epoch {epoch}: {checkpoint_path}")

    if mlflow_run_id:
        mlflow.pytorch.log_model(model, "checkpoints/epoch_{}".format(epoch))

def load_checkpoint(model, optimizer, path="model_checkpoint.pth"):
    """Load a model checkpoint and return epoch, model, and optimizer states."""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model checkpoint loaded from {path} (Epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    else:
        print(f"No checkpoint found at {path}, starting fresh.")
        return 0  # Starting from epoch 0 if no checkpoint exists
