import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# data paths
# dataset_folder = './data/selected_wavfiles_label/BSC5_Spectrograms(Amplitude)'
# dataset_folder = './data/selected_wavfiles_label/BSC5_Spectrograms(Power)'
# dataset_folder = './data/selected_wavfiles_label/BSC5_Spectrograms(DB)'
#--------------------------------------------------------------------------------------------------------------
# dataset_folder = './data/selected_wavfiles_label/BSC5_Adaf_Spectrograms_128(Amplitude)'
# dataset_folder = './data/selected_wavfiles_label/BSC5_Adaf_Spectrograms_128(Power)'
dataset_folder = './data/selected_wavfiles_label/BSC5_Adaf_Spectrograms_128(DB)'
#--------------------------------------------------------------------------------------------------------------
# dataset_folder = './data/selected_wavfiles_label/BSC5_Mel_Spectrograms_128'


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.net = torchvision.models.efficientnet_b0(weights="DEFAULT")

        # average the pretrained first layer weights, adapt to single channel
        original_weights = self.net.features[0][0].weight.detach().clone()
        new_weights = original_weights.mean(dim=1, keepdim=True)

        # modify the first layer of EfficientNet, change the input channel from 3 to 1
        self.net.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.net.features[0][0].weight = nn.Parameter(new_weights)

        in_features = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)


class SpectrogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# scan all image paths and labels
all_images = []
all_labels = []
for file_name in os.listdir(dataset_folder):
    if file_name.endswith('.png'):
        img_path = os.path.join(dataset_folder, file_name)
        label = int(file_name.split('_')[0])
        all_images.append(img_path)
        all_labels.append(label)

all_images = np.array(all_images)
all_labels = np.array(all_labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics for each fold
fold_train_losses = []
fold_val_losses = []
fold_train_accuracies = []
fold_val_accuracies = []
fold_auc_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
    print(f"\n=== Fold {fold_idx + 1}/5 ===")

    train_images, val_images = all_images[train_idx], all_images[val_idx]
    train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

    train_dataset = SpectrogramDataset(train_images, train_labels, transform=transform)
    val_dataset = SpectrogramDataset(val_images, val_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    num_epochs = 10
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    auc_scores = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            labels = labels.long().view(-1).to(device)
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_true = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.long().view(-1).to(device)
                images = images.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_true.extend(labels.cpu().numpy())
                all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

        val_accuracy = correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        auc = roc_auc_score(all_true, all_probs, multi_class='ovo', average='macro')
        auc_scores.append(auc)
        scheduler.step(val_losses[-1])

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}, AUC: {auc:.4f}, '
              f'Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')

    # save the model after the current fold is trained
    model_path = f'BSC-5_Adaf_Spectrograms_128(DB)_5fold_{fold_idx + 1}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"--- Model for Fold {fold_idx + 1} saved to {model_path} ---")

    # Store final epoch metrics
    fold_train_losses.append(train_losses[-1])
    fold_val_losses.append(val_losses[-1])
    fold_train_accuracies.append(train_accuracies[-1])
    fold_val_accuracies.append(val_accuracies[-1])
    fold_auc_scores.append(auc_scores[-1])

# Print final cross-validation results
mean_train_loss = np.mean(fold_train_losses)
std_train_loss = np.std(fold_train_losses)
mean_val_loss = np.mean(fold_val_losses)
std_val_loss = np.std(fold_val_losses)

mean_train_acc = np.mean(fold_train_accuracies)
std_train_acc = np.std(fold_train_accuracies)
mean_val_acc = np.mean(fold_val_accuracies)
std_val_acc = np.std(fold_val_accuracies)

mean_auc = np.mean(fold_auc_scores)
std_auc = np.std(fold_auc_scores)

print("\nCross-Validation Results:")
print(f"Average Train Loss:       {mean_train_loss:.4f} ± {std_train_loss:.4f}")
print(f"Average Validation Loss:  {mean_val_loss:.4f} ± {std_val_loss:.4f}")
print(f"Average Train Accuracy:   {mean_train_acc*100:.2f} ± {std_train_acc*100:.2f} %")
print(f"Average Validation Acc:   {mean_val_acc*100:.2f} ± {std_val_acc*100:.2f} %")
print(f"Average Validation AUC:   {mean_auc:.4f} ± {std_auc:.4f}")

# calculate and print class distribution (from last fold)
print("\nnumber of images for each class in the training set:")
train_class_counts = {}
for _, label in train_dataset:
    train_class_counts[label] = train_class_counts.get(label, 0) + 1
for label, count in sorted(train_class_counts.items()):
    print(f"  class {label}: {count} images")

print("number of images for each class in the test set:")
test_class_counts = {}
for _, label in val_dataset:
    test_class_counts[label] = test_class_counts.get(label, 0) + 1
for label, count in sorted(test_class_counts.items()):
    print(f"  class {label}: {count} images")

# save sample images from the last fold's training set
for i in range(5):
    image, _ = train_dataset[i]
    np_image = image.permute(1, 2, 0).numpy()
    plt.imshow(np_image)
    plt.title(f"Image {i+1}")
    plt.savefig(f'sample_{i+1}.png')
    plt.close()
