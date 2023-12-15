import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import numpy as np

from sklearn.model_selection import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, InputLayer
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import cv2
import os
import glob
import gc

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image


def lire_images(img_dir, xdim, ydim, nmax=5000):
    """
    Lit les images dans les sous répertoires de img_dir
    nmax images lues dans chaque répertoire au maximum
    Renvoie :
    X : liste des images lues, matrices xdim*ydim
    y : liste des labels numériques
    label : nombre de labels
    label_names : liste des noms des répertoires lus
    """
    label = 0
    label_names = []
    X = []
    y = []
    for dirname in os.listdir(img_dir):
        print(dirname)
        label_names.append(dirname)
        data_path = os.path.join(img_dir + "/" + dirname, "*g")
        files = glob.glob(data_path)
        n = 0
        for f1 in files:
            if n > nmax:
                break
            img = cv2.imread(f1)  # Lecture de l'image dans le repertoire
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Conversion couleur RGB
            img = cv2.resize(img, (xdim, ydim))  # Redimensionnement de l'image
            X.append(
                np.array(img)
            )  # Conversion en tableau et ajout a la liste des images
            y.append(label)  # Ajout de l'etiquette de l'image a la liste des etiquettes
            n = n + 1
        print(n, " images lues")
        label = label + 1
    X = np.array(X)
    y = np.array(y)
    gc.collect()  # Récupération de mémoire
    return X, y, label, label_names


X, y, num_classes, labels = lire_images(
    "dataset/Pure Naruto Hand Sign Data/train", 224, 224, 1000
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train1 = X_train.reshape(len(X_train), -1) / 255
X_test1 = X_test.reshape(len(X_test), -1) / 255

X = X / 255

model = torchvision.models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 13)


def train(model, train_loader, val_loader, loss_fn, optimizer, epochs=20, device="cpu"):
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        running_loss = 0.0
        training_acc = []
        model.train()
        model.to(device)
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            training_acc = accuracy_score(labels.cpu(), preds.cpu())
        training_acc = np.mean(training_acc)
        train_acc.append(training_acc)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        val_loss = 0.0
        validation_acc = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                validation_acc = accuracy_score(labels.cpu(), preds.cpu())
            validation_acc = np.mean(validation_acc)
            val_acc.append(validation_acc)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
        print(
            f"[{epoch+1}] Training loss: {epoch_loss:.4f}\t Validation loss: {val_loss:.4f}"
        )
    return train_losses, val_losses, train_acc, val_acc


transform_dataset = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


train_dataset = torchvision.datasets.ImageFolder(
    root="dataset/Pure Naruto Hand Sign Data/train", transform=transform_dataset
)
val_dataset = torchvision.datasets.ImageFolder(
    root="dataset/Pure Naruto Hand Sign Data/test", transform=transform_dataset
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses, train_acc, val_acc = train(
    model, train_loader, val_loader, loss, optimizer, epochs=20, device="cpu"
)

torch.save(model.state_dict(), "resnet18_naruto_local.pth")
