import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np

import json

import argparse

import workspace_utils


def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 

    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: that's a path to a folder
    parser.add_argument('data_directory', type = str,
                        help = 'path to the folder of image data') 
    parser.add_argument('--save_dir', type = str, default = '', 
                        help = 'path to the folder for saving the checkpoint') 
    parser.add_argument('--arch', type = str, default = 'vgg', 
                        help = 'cnn model Architecture to use for the classifier') 
    parser.add_argument('--learning_rate', type = float, default = 0.01, 
                        help = 'learning rate for classifier training')
    parser.add_argument('--hidden_units', type = int, default = 512, 
                        help = 'number of units in hidden layer of classifier') 
    parser.add_argument('--epochs', type = int, default = 20, 
                        help = 'number of epochs for classifier training') 
    parser.add_argument('--gpu', type = int, action = 'store_const', const = 1, default = 0, 
                        help = 'specify if gpu will be used if available in classifier training') 
    
    

    return parser.parse_args()


#
# main
#

args = get_input_args()

# TODO: parse argument instead
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# : Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# : Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

# : Using the image datasets and the trainforms, define the dataloaders
train_batch_size = 64
test_batch_size = 64
trainloader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size)
validationloader= torch.utils.data.DataLoader(validation_data, batch_size=test_batch_size)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: learning rate from parser
# TODO: model architecture from parser
# Build and train your network
model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 1024)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(1024, len(cat_to_name))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
for param in model.classifier.parameters():
    param.requires_grad = True


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

# TODO: gpu usage from parser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device);

# TODO: epochs from parser
epochs = 10
steps = 0
running_loss = 0
print_every = 15
model.train()

for epoch in workspace_utils.keep_awake(range(epochs)):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
                              
            running_loss = 0
            model.train()

# : Do validation on the validation set
accuracy = 0
test_loss = 0
with torch.no_grad():
    for inputs, labels in validationloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
print(f"Validation loss: {test_loss/len(validationloader):.3f}.. "
        f"Validation accuracy: {accuracy/len(validationloader):.3f}")

# : Save the checkpoint 
# TODO: save director from parser
path = 'vgg16_flowers.pth'
torch.save(model.state_dict(), path)
