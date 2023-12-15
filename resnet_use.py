import torch
import torchvision
from torchvision import transforms
from torch import nn
from PIL import Image
import os
import numpy as np

# Function to define the custom resnet model and load the pretrained weights


def load_model(path):
    model = torchvision.models.resnet18(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, 13)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return model


# Function to predict the class of the image

classes = os.listdir("dataset/Pure Naruto Hand Sign Data/train")

transform_dataset = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def predict_image(image, model):
    # convert image to tensor when image is an ndarray
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    else:
        image = Image.open(image)  # image is a path to an image

    image_tensor = transform_dataset(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    model.eval()
    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()

    # get the confidence score of the prediction
    score = torch.nn.functional.softmax(output.data, dim=1)
    score = score.data.cpu().numpy().max()

    return index, score
