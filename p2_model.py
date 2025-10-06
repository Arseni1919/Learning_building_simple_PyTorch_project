import timm

from globals import *



class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()

        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features =nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


if __name__ == '__main__':
    model = SimpleCardClassifier(num_classes=53)
    print(str(model))



