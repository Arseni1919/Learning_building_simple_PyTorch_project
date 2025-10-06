import matplotlib.pyplot as plt
from pprint import pprint

import torch

from globals import *


class PlayingCardDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


if __name__ == '__main__':
    # implementing transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = PlayingCardDataset(
        data_dir=DATA_DIR,
        transform=transform
    )

    # initial info
    print(len(dataset))
    image, label = dataset[6000]
    print(f'{label=}')

    if isinstance(image, torch.Tensor):
        print(image.shape)
    else:
        # to show the card - use without transforms
        plt.imshow(image)
        plt.show()


    # looking at the labels
    target_to_class = {v: k for k, v in ImageFolder(DATA_DIR).class_to_idx.items()}
    pprint(target_to_class)

    # to iterate over a dataset
    for image, label in dataset:
        break

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # to iterate over this dataloader
    for images, labels in dataloader:
        print(f'{images.shape=}')
        print(f'{labels.shape=}')
        print(f'{labels=}')
        break





