import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def create_dataloader(root, transforms, batch_size, num_workers=os.cpu_count(), shuffle=True, pin_memory=True):

    dataset = ImageFolder(
        root=root,
        transform=transforms
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory
    )