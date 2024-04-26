import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ViT_L_16_Weights, vit_l_16
from torchvision.transforms import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from Transformers import ViT


def train_model(model, train_loader, validation_loader, device, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    scheduler = OneCycleLR(optimizer, max_lr=0.003, total_steps=num_epochs * len(train_loader))
    scaler = GradScaler()
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_correct = 0
        epoch_samples = 0

        all_targets = []
        all_predictions = []

        progress_bar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', position=0, leave=True)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            batch_loss = loss.item() * inputs.size(0)
            epoch_loss += batch_loss
            epoch_samples += targets.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            progress_bar.set_postfix(loss=(epoch_loss / epoch_samples), accuracy=(total_correct / epoch_samples))
            progress_bar.update(1)

        progress_bar.close()
        training_loss = epoch_loss / epoch_samples
        training_accuracy = total_correct / epoch_samples
        validation_loss, validation_accuracy = validation(model, validation_loader, device, epoch, num_epochs)

        history['train_loss'].append(training_loss)
        history['train_accuracy'].append(training_accuracy)
        history['val_loss'].append(validation_loss)
        history['val_accuracy'].append(validation_accuracy)

        if epoch == num_epochs-1:
            plot_confusion_matrix(all_targets, all_predictions, 'Reds', 'train')

        scheduler.step()

    return model, history


def validation(model, loader, device, epoch, num_epochs):
    model.eval()
    validation_loss = 0
    correct_predictions = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_validation_loss = validation_loss / total_samples
    validation_accuracy = correct_predictions / total_samples

    if epoch == num_epochs-1:
        plot_confusion_matrix(all_targets, predictions, 'Blues', 'validation')

    return avg_validation_loss, validation_accuracy


def test_model(model, test_loader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct_predictions += preds.eq(targets).sum().item()
            total_samples += inputs.size(0)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    plot_confusion_matrix(all_targets, all_predictions, 'Greens', 'test')
    accuracy = correct_predictions / total_samples * 100

    print(f'Test set: Accuracy: {correct_predictions}/{total_samples} ({accuracy:.2f}%)')


def plot_confusion_matrix(all_targets, all_predictions, color, value):
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    matrix_size = conf_matrix.shape[0]
    figsize = (max(8, matrix_size), max(6, matrix_size * 0.75))

    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=color, cbar=False)
    plt.title('Confusion Matrix Test')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'matrice_confusion_{value}.png')
    plt.close()


def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def main():
    use_gpu = True
    if not torch.cuda.is_available():
        print('Not connected to a GPU')
    else:
        print('Connected to a GPU')

    torch.cuda.empty_cache()
    device = torch.device("cuda" if use_gpu else "cpu")

    data_dir = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = "test"
    saving_path = './plant-disease-model2.pth'
    diseases = os.listdir(train_dir)

    print(diseases)

    batch_size = 50

    transforms_images = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train = ImageFolder(train_dir, transform=transforms_images)
    valid = ImageFolder(valid_dir, transform=transforms_images)
    test = ImageFolder(test_dir, transform=transforms_images)

    print(len(train.classes))

    data_loader_train = DataLoader(train, batch_size, shuffle=True, num_workers=4)
    data_loader_validation = DataLoader(valid, batch_size, shuffle=True, num_workers=4)
    data_loader_test = DataLoader(test, batch_size, shuffle=True, num_workers=4)

    #model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    image, label = train[0]
    my_model = ViT(img_size=image.shape[1], num_classes=len(train.classes), device=device)
    #num_features = model.heads.head.in_features
    #for param in my_model.parameters():
    #    param.requires_grad = True

    #model.heads.head = nn.Linear(num_features, len(train.classes))
    if use_gpu:
        my_model.to(device)

    model, history = train_model(my_model, data_loader_train, data_loader_validation, device, num_epochs=1)

    plot_training_history(history)
    test_model(model, data_loader_test, device)
    torch.save(model.state_dict(), saving_path)


if __name__ == "__main__":
    main()
