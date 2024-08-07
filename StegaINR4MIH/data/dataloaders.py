from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os

def mnist(path_to_data, batch_size=16, size=28, train=True, download=False):
    """MNIST dataloader.

    Args:
        path_to_data (string): Path to MNIST data files.
        batch_size (int):
        size (int): Size (height and width) of each image. Default is 28 for no resizing. 
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(path_to_data, train=train, download=download,
                             transform=all_transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def celebahq(path_to_data, batch_size=16, size=256, secret_size=None):
    if secret_size is None:
        secret_size = size

    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(path_to_data, transform=all_transforms)
    dataset.samples = [i for i in dataset.samples if i[1] == 0]

    if secret_size is not None:
        transform = transforms.Compose([
            transforms.Resize(secret_size),
            transforms.ToTensor()
        ])
        secret_dataset = datasets.ImageFolder(path_to_data, transform=transform)
        secret_dataset.samples = [i for i in secret_dataset.samples if i[1] == 1]

        # sort by index
        secret_dataset.samples = sorted(secret_dataset.samples, 
                                        key=lambda x: int(os.path.basename(x[0]).split('.jpg')[0].split('secret')[-1]) 
                                        if os.path.basename(x[0]).split('.jpg')[0] != 'cover' else -1)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    secret_dataloader = DataLoader(secret_dataset, batch_size=batch_size, shuffle=False)

    return dataloader, secret_dataloader
