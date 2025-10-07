import matplotlib.pyplot as plt
from torchvision.transforms.v2 import Compose

from globals import *
from PIL import Image

from p1_data_source import PlayingCardDataset
from p2_model import SimpleCardClassifier


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_ylabel("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # preps
    transform = transforms, Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data
    dataset = PlayingCardDataset(
        data_dir=DATA_DIR,
        transform=transform
    )

    # model
    model = SimpleCardClassifier(num_classes=53)
    model.load_state_dict(torch.load("model_weights.pth"))

    test_image = "data/test/five of diamonds/2.jpg"

    original_image, image_tensor = preprocess_image(test_image, transform)
    probabilities = predict(model, image_tensor, device)

    class_names = dataset.classes
    visualize_predictions(original_image, probabilities, class_names)