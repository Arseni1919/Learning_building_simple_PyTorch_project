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
    dataset = PlayingCardDataset(
        data_dir=DATA_DIR
    )

