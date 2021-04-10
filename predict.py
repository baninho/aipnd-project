import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np

import json

import argparse

import workspace_utils


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type = str,
                        help = 'path to the folder of image data') 
    parser.add_argument('checkpoint', type = str, 
                        help = 'path to the checkpoint') 
    parser.add_argument('--top_k', type = int, default = 3, 
                        help = 'number of most likely classes to display') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                        help = 'category name mapping json file') 
    parser.add_argument('--gpu', action = 'store_const', const = 1, default = 0, 
                        help = 'specify if gpu will be used if available in classifier training') 
    
    

    return parser.parse_args()

# : Write a function that loads a checkpoint and rebuilds the model
# TODO: allow different architectures and get user choice from args
def load_vgg16_flowers(path):
    model = models.vgg16()
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 1024)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(1024, len(cat_to_name))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.load_state_dict(torch.load(path))
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # : Process a PIL image for use in a PyTorch model
    return test_transforms(image)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # : Implement the code to predict the class from an image file
    input_tensor = process_image(Image.open(image_path))
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available() and args.gpu:
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        logits = model(input_batch)

    ps = F.softmax(logits[0], dim=0)

    top_p, top_class = ps.topk(topk, dim=0)
    
    return top_p.to('cpu').numpy(), [str(cat+1) for cat in top_class.to('cpu').tolist()]


#
# main
#

args = get_input_args()

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# : model path from args
model = load_vgg16_flowers(args.checkpoint)
model.eval()

# : topk from args
probs, classes = predict(args.image_path, model, args.top_k)

names = [cat_to_name[i] for i in classes]

print(names)
print(probs)