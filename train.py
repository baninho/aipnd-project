import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np

import json

import argparse
from os import path

import workspace_utils


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', type = str,
                        help = 'path to the folder of image data') 
    parser.add_argument('--save_dir', type = str, default = 'flowers_checkpoint.pth', 
                        help = 'path to the folder for saving the checkpoint') 
    parser.add_argument('--arch', type = str, default = 'vgg', 
                        help = 'cnn model Architecture to use for the feature detector: vgg or alexnet') 
    parser.add_argument('--learning_rate', type = float, default = 0.01, 
                        help = 'learning rate for classifier training')
    parser.add_argument('--hidden_units', type = int, default = 512, 
                        help = 'number of units in hidden layer of classifier') 
    parser.add_argument('--epochs', type = int, default = 20, 
                        help = 'number of epochs for classifier training') 
    parser.add_argument('--gpu', action = 'store_const', const = 1, default = 0, 
                        help = 'specify if gpu will be used if available in classifier training') 
    
    

    return parser.parse_args()



#
# main
#

args = get_input_args()

# : use argument for data_dir
data_dir = args.data_directory
train_dir = path.join(data_dir, 'train')
valid_dir = path.join(data_dir, 'valid')
test_dir = path.join(data_dir, 'test')

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
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=test_batch_size)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# : learning rate from parser
# : model architecture from parser
# Build and train your network
if args.arch == 'vgg':
	model = models.vgg16(pretrained=True)
	feature_units = 25088
elif args.arch == 'alexnet':
	model = models.alexnet(pretrained=True)
	feature_units = 9216
else:
	print('model architecture parameter must be vgg or alexnet')
	exit

learning_rate = args.learning_rate

for param in model.parameters():
    param.requires_grad = False


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(feature_units, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(args.hidden_units, len(cat_to_name))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
for param in model.classifier.parameters():
    param.requires_grad = True


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# : gpu usage from parser
if args.gpu:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU for training')
    else:
        device = torch.device('cpu')
        print('No GPU available, using CPU for training')
else: 
    device = torch.device('cpu')
    print('Using CPU for training')
    
model.to(device);

# : epochs from parser
epochs = args.epochs
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
# : save directory from parser
path = args.save_dir
model.to(torch.device('cpu'))

checkpoint = {
	'state_dict': model.state_dict(),
	'arch': args.arch,
	'hidden_units': args.hidden_units,
}
torch.save(checkpoint, path)
