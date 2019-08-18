import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import time

from utils import IOUtils, TypeUtils

def parseInput():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-train', '--train_data_directory', action='store',
        help="Training data directory")
    parser.add_argument('-test', '--test_data_directory', action='store',
        help="Test data directory")
    parser.add_argument('-s', '--save_directory', action='store',
        help="Directory to save model")
    parser.add_argument('-a', '--arch', action='store',
        help="Pretrained model architecture")
    parser.add_argument('-o', '--num_outputs', action='store',
        help="Number of model outputs")
    parser.add_argument('-hidden', '--hidden_layers', action='store',
        help="Number of model outputs")
    parser.add_argument('-e', '--epochs', action='store',
        help="Number of epochs")
    parser.add_argument('-gpu', '--gpu', action='store_true',
        help="Use GPU if it's available")
    parser.add_argument('-alpha', '--learning_rate', action='store',
        help="Learning rate")

    args = parser.parse_args()
    required_arguments = ['train_data_directory', 'test_data_directory', 'num_outputs']

    for arg in required_arguments:
        if not vars(args)[arg]:
            IOUtils.notify(f"You are missing a required argument: {arg}")
            exit()

    valid_architectures = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    if args.arch:
        if not args.arch in valid_architectures:
            IOUtils.notify(f"Invalid architecture. Must be one of {valid_architectures}")

    return {
        'architecture': args.arch,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'train_dir': args.train_data_directory,
        'test_dir': args.test_data_directory,
        'save_dir': args.save_directory,
        'num_outputs': args.num_outputs,
        'hidden_layers': args.hidden_layers,
        'shouldTryGPU': args.gpu
    }

class Trainer:
    def __init__(self, *args, **kwargs):
        self.architecture = kwargs.get('architecture')
        self.dropout_rate = TypeUtils.tryCastToFloat(kwargs.get('dropout'))
        self.epochs = TypeUtils.tryCastToInt(kwargs.get('epochs'))
        self.hidden_layers = kwargs.get('hidden_layers')
        self.learning_rate = TypeUtils.tryCastToFloat(kwargs.get('learning_rate'))
        self.save_dir = kwargs.get('save_dir')

        self.test_dir = kwargs.get('test_dir')
        self.train_dir = kwargs.get('train_dir')
        self.num_outputs = kwargs.get('num_outputs')

        if kwargs.get('shouldTryGPU'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            IOUtils.notify("ALERT: You are not training using GPU. Training may take a long time. To use GPU, call the function with the -gpu flag.")
            self.device = torch.device("cpu")


    def buildModel(self, *args, **kwargs):
        self.defineModel()
        self.defineArchitecture()

    def defineModel(self, *args, **kwargs):
        arch = self.architecture
        # A dictionary would be more elegant here, but if you put the models as values in a dict, python will go download them all as well (which we dont want unless we're going to use it)
        if arch == 'vgg11':
            self.model = models.vgg11(pretrained=True)
        elif arch == 'vgg13':
            self.model = models.vgg13(pretrained=True)
        elif arch == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        elif arch == 'vgg19':
            self.model = models.vgg19(pretrained=True)
        else:
            # No architecture provided. Let them define one now, or use a default
            shouldUseDefault = IOUtils.yesOrNo("No architecture was provided. Press (y) to use the default [vgg19], or (n) to define your own architecture.")
            if shouldUseDefault:
                self.architecture = 'vgg19'
                self.model = models.vgg19(pretrained=True)
            else:
                supportedModels = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
                chosenArchitecture = IOUtils.getResponse(f"Choose model architecture. Options are {supportedModels}:",
                    supportedModels)
                self.architecture = chosenArchitecture
                self.defineModel()

    def defineArchitecture(self, *args, **kwargs):
        num_vgg_inputs = 25088 # True for any vggX model

        if not self.learning_rate:
            IOUtils.notify("Learning rate not specified. Using default 0.001 learning rate.")
            self.learning_rate = 0.001
        if not self.epochs:
            IOUtils.notify("Number of training epochs not defined. Using default 10 epochs.")
            self.epochs = 10
        if not self.hidden_layers or not isinstance(self.hidden_layers, list):
            IOUtils.notify("Hidden layers not defined as array. Using default 2 hidden layers.")
        if not self.dropout_rate:
            IOUtils.notify("Dropout rate not defined. Using default 0.5 dropout rate for all hidden layers.")
            self.dropout_rate = 0.5
        if not self.epochs:
            IOUtils.notify("Num epochs not provided. Using default 1 epoch.")
            self.epochs = 1

        ## Build the hidden layers architecture
        hidden_layers = []
        self.num_outputs = int(self.num_outputs)
        if self.hidden_layers:
            # Hidden layers comes in as a string. First need to convert it to a list of integers
            self.hidden_layers = list(map(int, (self.hidden_layers.split(','))))
            for i in range(len(self.hidden_layers)):
                # Hidden layer should look like nn.Linear(in_count, out_count)
                # unless it's the last hidden layer,
                # which shoul look like nn.Linear(in_count, self.num_outputs)
                if i == len(self.hidden_layers) - 1:
                    # This is the last hidden layer
                    hidden_layers.append((self.hidden_layers[i], self.num_outputs))
                else:
                    hidden_layers.append((self.hidden_layers[i], self.hidden_layers[i+1]))
        else:
            # Hidden layers not provided. We'll build them ourselves using the
            # algorithm from http://dstath.users.uth.gr/papers/IJRS2009_Stathakis.pdf:
            #
            # 1st hidden layer = sqrt(m+2)*N + 2*sqrt(N/(m+2))
            # 2nd hidden layer = m*sqrt(N/(m+2))
            # where m = number of model outputs and N = number of model inputs

            first_layer_inputs = int(((self.num_outputs + 2)*num_vgg_inputs)**0.5 + 2*((num_vgg_inputs / (self.num_outputs + 2))**0.5))
            second_layer_inputs = int(self.num_outputs * ((num_vgg_inputs / (self.num_outputs + 2))**0.5))
            hidden_layers.append((first_layer_inputs, second_layer_inputs))
            hidden_layers.append((second_layer_inputs, self.num_outputs))

        ## Build the classifier
        classifier_layers = [
            nn.Linear(num_vgg_inputs, hidden_layers[0][0]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)]

        for layer in hidden_layers:
            classifier_layers.append(nn.Linear(layer[0], layer[1]))
            if not layer == hidden_layers[-1]:
                #This is not the last layer
                classifier_layers.append(nn.ReLU())
                classifier_layers.append(nn.Dropout(self.dropout_rate))
        classifier_layers.append(nn.LogSoftmax(dim=1))

        classifier = nn.Sequential(*classifier_layers)

        for param in self.model.parameters():
            param.requires_grad = False # Freeze parameters so we don't backprop through them

        self.model.classifier = classifier
        self.model.criterion = nn.NLLLoss()
        self.model.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)

        IOUtils.notify(f"Model definition complete."
            f"\n\t* Model architecture: {self.architecture}"
            f"\n\t* Learning rate: {self.learning_rate}"
            f"\n\t* Classifier: {self.model.classifier}"
            f"\n\t* Droput rate: {self.dropout_rate}"
            f"\n\t* Number of epochs: {self.epochs}"
            )

    def trainModel(self, *args, **kwargs):
        self.buildModel()
        self.loadImages()
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

    def train(self, *args, **kwargs):
        IOUtils.notify("Training started")
        self.model.to(self.device)
        running_loss = 0
        self.model.train()
        for epoch in (range(self.epochs)):
            start_time = time.time()
            for inputs, labels in self.dataloaders['train']:
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.model.optimizer.zero_grad()

                logps = self.model.forward(inputs)
                loss = self.model.criterion(logps, labels)

                loss.backward()
                self.model.optimizer.step()

                running_loss += loss.item()
            self.validateModel(epoch, running_loss, start_time)
            running_loss = 0

    def promptSave(self):
        IOUtils.notify("Training complete. Save the trained model?")
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
            self.save_dir = self.save_dir if self.save_dir else 'model_checkpoints'
            os.makedirs(f"./{self.save_dir}", exist_ok=True)  # Make the save_dir (OK if it already exists)
            torch.save(checkpoint, f"{self.save_dir}/{savePath}")
        else:
            IOUtils.notify("Not saving model. Program terminated.")

    def validateModel(self, epoch, running_loss, start_time):
        test_loss = 0
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.dataloaders['test']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logps = self.model.forward(inputs)
                batch_loss = self.model.criterion(logps, labels)
                test_loss += batch_loss.item()

                # Accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        IOUtils.notify(f"Epoch {epoch+1}/{self.epochs}.. "
              f"Time: {time.time() - start_time:.3f} "
              f"Train loss: {running_loss/len(self.dataloaders['train']):.3f}.. "
              f"Test loss: {test_loss/len(self.dataloaders['test']):.3f}.. "
              f"Test accuracy: {accuracy/len(self.dataloaders['test']):.3f}")
        self.model.train()

modelInputs = parseInput()
trainer = Trainer(**modelInputs)

trainer.trainModel()