from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def load_mnist_data(batch_size):
    """Load the MNIST dataset."""
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader
