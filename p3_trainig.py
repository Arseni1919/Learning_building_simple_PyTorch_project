import matplotlib.pyplot as plt
import torch

from globals import *
from p1_data_source import *
from p2_model import *


def train():

    # --- #
    # implementing transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = PlayingCardDataset(TRAIN_DATA_DIR, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = PlayingCardDataset(VAL_DATA_DIR, transform)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataset = PlayingCardDataset(TEST_DATA_DIR, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- #

    model = SimpleCardClassifier(num_classes=53)

    # --- #

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # --- #
    images, labels = next(iter(train_dataloader))
    example_out = model(images)
    criterion(example_out, labels)
    print(f"{example_out.shape=}")
    print(f"{labels.shape=}")

    # --- #

    # Simple training loop

    num_epochs = 5
    train_losses, val_losses = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        # training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc="Training loop"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # validation phase
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc="Validation loop"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_dataloader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    # --- #

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.show()

    # --- #
    # --- #
    # --- #


if __name__ == '__main__':
    train()