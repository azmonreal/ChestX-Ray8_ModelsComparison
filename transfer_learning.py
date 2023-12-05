import os
import time
import torch

from torchvision import transforms, models
from PIL import Image
from tempfile import TemporaryDirectory

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    dataset_sizes = {phase: len(model.dataset)
                     for (phase, model) in dataloaders.items()}

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        # Handle GoogLeNet specific output
                        if isinstance(outputs, tuple):
                            # Use just the final output logits for loss and predictions
                            logits = outputs.logits
                        else:
                            logits = outputs

                        _, preds = torch.max(logits, 1)
                        loss = criterion(logits, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(
            f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def save_model(model, destination, name):
    torch.save(model.state_dict(), f'{destination}/{name}.pth')


def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img.to(device)


def predict_image(image_path, model):
    img = process_image(image_path)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return predicted


def test_model(model, dataloader):
    # Initialize counters for correct predictions and total predictions for each class
    class_correct = list(0. for i in range(len(dataloader.dataset.classes)))
    class_total = list(0. for i in range(len(dataloader.dataset.classes)))

    # Move model to the computing device and set it to evaluation mode
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Loop over all batches in the dataloader
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Handle GoogLeNet specific output
            if isinstance(outputs, tuple):
                logits = outputs.logits
            else:
                logits = outputs

            _, predicted = torch.max(logits, 1)
            c = (predicted == labels).squeeze()

            # Update class-specific accuracy counts
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Print accuracy for each class after processing all images
    for i in range(len(dataloader.dataset.classes)):
        if class_total[i] > 0:
            print(
            f'Accuracy of {dataloader.dataset.classes[i]}: {100 * class_correct[i] / class_total[i]}%')
