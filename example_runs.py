from p1_data_source import *
from p2_model import *


def example_run_1():
    # example
    # implementing transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = PlayingCardDataset(
        data_dir=DATA_DIR,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    images, labels = next(iter(dataloader))

    model = SimpleCardClassifier(num_classes=53)
    example_out = model(images)
    print(f"{example_out.shape=}")


if __name__ == '__main__':
    example_run_1()