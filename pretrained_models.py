import torch

from torch import nn
from torchvision import models


def download_pretrained_models(model_names, destination):
    for name in model_names:
        try:
            # dowload model
            model = getattr(models, name)(weights='IMAGENET1K_V1')

            # save model
            torch.save(model.state_dict(), f'{destination}/{name}.pth')

            print(f"{name} successfully downloaded and saved.")
        except Exception as e:
            print(f"Error downloading and saving {name}: {e}")


def load_models(models_dict, models_path):
    loaded_models = {}
    for (name, model) in models_dict.items():
        try:
            # Create an instance of the model without pretrained weights
            model = getattr(models, name)(weights=None)

            # Deactivate auxiliary layers in the GoogLeNet model if necessary.
            if name == 'googlenet':
                model.aux_logits = False
                model.aux1.fc1 = nn.Identity()
                model.aux1.fc2 = nn.Identity()
                model.aux2.fc1 = nn.Identity()
                model.aux2.fc2 = nn.Identity()
                model.aux1.conv = nn.Identity()
                model.aux2.conv = nn.Identity()

            # Load weights from file
            model.load_state_dict(torch.load(f'{models_path}/{name}.pth'),
                                  strict=False)

            loaded_models[name] = model

            print(f"{name} successfully loaded.")
        except Exception as e:
            print(f"Error loading {name}: {e}")

    return loaded_models
