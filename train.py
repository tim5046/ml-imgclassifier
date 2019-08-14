import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import time

from utils import IOUtils

def parseInput():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train_data_directory', action='store',
        help="Training data directory")
    parser.add_argument('-test', '--test_data_directory', action='store',
        help="Test data directory")
    parser.add_argument('-s', '--save_directory', action='store',
        help="Directory to save model")
    parser.add_argument('-a', '--arch', action='store',
        help="Pretrained model architecture")

    args = parser.parse_args()
    required_arguments = ['train_data_directory', 'test_data_directory', 'save_directory']

    for arg in required_arguments:
        if not vars(args)[arg]:
            print(f"You are missing an argument: {arg}")
            break

    valid_architectures = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    if args.arch:
        if not args.arch in valid_architectures:
            print(f"Invalid architecture. Must be one of {valid_architectures}")

    return {
        'architecture': args.arch,
        'train_dir': args.train_data_directory,
        'test_dir': args.test_data_directory,
        'save_dir': args.save_directory,
    }

class Trainer:
    def __init__(self, *args, **kwargs):
        print("KWARGS", kwargs)
        self.architecture = kwargs['architecture']
        self.epochs = kwargs['epochs']
        self.hidden_layers = kwargs['hidden_layers']
        self.learning_rate = kwargs['learning_rate']
        self.save_dir = kwargs['save_dir']
        self.shouldTryGPU = kwargs['shouldTryGPU']
        self.test_dir = kwargs['test_dir']
        self.train_dir = kwargs['train_dir']

    def buildModel(self, *args, **kwargs):
        arch = self.architecture
        # A dictionary would be more elegant here, but if you put the models as values in a dict, python will go download them all as well (which we dont want unless we're going to use it)
        if arch == 'vgg11':
            self.model = models.vgg11(pretrained=True)
        elif arch == 'vgg13':
            self.model = models.vgg13(pretrained=True)
        elif arch == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        elif arch == 'vgg19':
            self.model = models.vgg16(pretrained=True)
        else:
            # No architecture provided. Let them define one now, or use a default
            shouldUseDefault = IOUtils.yesOrNo("No architecture was provided. Press (y) to use the default [vgg19], or (n) to define your own architecture.")
            if shouldUseDefault:
                self.model = models.vgg19(pretrained=True)
            else:
                supportedModels = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
                chosenArchitecture = IOUtils.getResponse(f"Choose model architecture. Options are {supportedModels}:",
                    supportedModels)
                print("YOU CHOSE ", chosenArchitecture)
                self.architecture = chosenArchitecture
                self.buildModel()

    def printModel(self):
        print("MODEL CHOSEN:", self.model)

    def train(self, *args, **kwargs):
        self.buildModel()
        self.loadImages()
        self.instantiateModel()
        self.train()
        self.promptSave()

    def loadImages(self, *args, **kwargs):
        data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]),
            'test': transforms.Compose([transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
        }

        image_datasets = {
            'train': datasets.ImageFolder(self.train_dir, transform=data_transforms['train']),
            'test': datasets.ImageFolder(self.test_dir, transform=data_transforms['test']),
        }

        self.dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=128),
            'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=128),
        }


    def instantiateModel(self, *args, **kwargs):
        if (self.architecture):
            self.model = modelFromArch(self.architecture)
        else: #Default to vgg 19
            self.model = models.vgg19(pretrained=True)
        print("MODEL", self.model)
        classifier = nn.Sequential(
            nn.Linear(25088, 1646),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1646, 1584),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1584, 102),
            nn.LogSoftmax(dim=1)
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = classifier
        self.model.criterion = nn.NLLLoss()
        self.model.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.00001)


    def train(self, *args, **kwargs):
        print("##########\nTraining started")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.epochs = 1
        running_loss = 0
        for epoch in (range(self.model.epochs)):
            start_time = time.time()
            for inputs, labels in self.dataloaders['train']:
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                self.model.optimizer.zero_grad()

                logps = self.model.forward(inputs)
                loss = self.model.criterion(logps, labels)
                loss.backward()
                self.model.optimizer.step()

                running_loss += loss.item()
            self.validateModel(epoch, running_loss, start_time)
            running_loss = 0

    def promptSave(self):
        print("Training complete. Would you like to save your trained model?")
        shouldSave = IOUtils.yesOrNo("Press 'y' to save or 'n' to end without saving.")
        if shouldSave:
            savePath = IOUtils.getResponse("Enter filename (it should end in .pth)")
            # TODO: Save the checkpoint
            checkpoint = {
                          'classifier': self.model.classifier,
                          'input_size': 25088,
                          'output_size': 102,
                          'hidden_layers': [1646, 1584],
                          'state_dict': self.model.state_dict(),
                         }
            os.makedirs(f"./{self.save_dir}", exist_ok=True)  # Make the save_dir (OK if it already exists)
            torch.save(checkpoint, f"{self.save_dir}/{savePath}")
        else:
            print("Not saving model. Program terminated.")



    def validateModel(self, epoch, running_loss, start_time):
        print("Validating model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## Output test results
        test_loss = 0
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = self.model.forward(inputs)
                batch_loss = self.model.criterion(logps, labels)
                test_loss += batch_loss.item()

                # Accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{self.model.epochs}.. "
              f"Time: {time.time() - start_time:.3f} "
              f"Train loss: {running_loss/len(self.dataloaders['train']):.3f}.. "
              f"Test loss: {test_loss/len(self.dataloaders['test']):.3f}.. "
              f"Test accuracy: {accuracy/len(self.dataloaders['test']):.3f}")
        self.model.train()

modelInputs = parseInput()
trainer = Trainer(**modelInputs)

trainer.buildModel()
trainer.printModel()
# print("VGG 19", models.vgg19(pretrained=True))

#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace)
#     (2): Dropout(p=0.5)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace)
#     (5): Dropout(p=0.5)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )

# print("\n\nVGG 16", models.vgg16(pretrained=True))

  # (classifier): Sequential(
  #   (0): Linear(in_features=25088, out_features=4096, bias=True)
  #   (1): ReLU(inplace)
  #   (2): Dropout(p=0.5)
  #   (3): Linear(in_features=4096, out_features=4096, bias=True)
  #   (4): ReLU(inplace)
  #   (5): Dropout(p=0.5)
  #   (6): Linear(in_features=4096, out_features=1000, bias=True)
  # )
# print("\n\nVGG 13", models.vgg13(pretrained=True))

  # (classifier): Sequential(
  #   (0): Linear(in_features=25088, out_features=4096, bias=True)
  #   (1): ReLU(inplace)
  #   (2): Dropout(p=0.5)
  #   (3): Linear(in_features=4096, out_features=4096, bias=True)
  #   (4): ReLU(inplace)
  #   (5): Dropout(p=0.5)
  #   (6): Linear(in_features=4096, out_features=1000, bias=True)
  # )
# print("\n\nVGG 11", models.vgg11(pretrained=True))
  # (classifier): Sequential(
  #   (0): Linear(in_features=25088, out_features=4096, bias=True)
  #   (1): ReLU(inplace)
  #   (2): Dropout(p=0.5)
  #   (3): Linear(in_features=4096, out_features=4096, bias=True)
  #   (4): ReLU(inplace)
  #   (5): Dropout(p=0.5)
  #   (6): Linear(in_features=4096, out_features=1000, bias=True)
  # )

