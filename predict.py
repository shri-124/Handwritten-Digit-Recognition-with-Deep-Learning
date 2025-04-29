import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Remove this line:
# from model import CNNModel

# Paste this instead:
import torch
from torch import nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Load model
inference_model = CNNModel()
inference_model.load_state_dict(torch.load('cnn_mnist_model.pth'))
inference_model.eval()

# Inference helper
def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Example usage
if __name__ == "__main__":
    # Load your data here OR import test_images_tensor from a shared module
    import numpy as np
    import gzip
    import struct
    import os

    # Reuse your dataset loading logic
    def load_test_images():
        MNIST_TEST_IMS_GZ = os.path.join("dataset", "t10k-images-idx3-ubyte.gz")
        with gzip.open(MNIST_TEST_IMS_GZ, mode='rb') as f:
            _, test_sz, nrows, ncols = struct.unpack('>llll', f.read(16))
            data_bn = f.read()
            data = struct.unpack('<' + 'B' * test_sz * nrows * ncols, data_bn)
            test_ims = np.asarray(data).reshape(test_sz, 1, 28, 28)
        return torch.tensor(test_ims).float()

    test_images_tensor = load_test_images()

    sample_img = test_images_tensor[0]
    prediction = predict_image(inference_model, sample_img)
    print("Predicted digit:", prediction)

    # Optional: show the image
    plt.imshow(sample_img.squeeze(), cmap='gray')
    plt.title(f'Predicted: {prediction}')
    plt.show()
