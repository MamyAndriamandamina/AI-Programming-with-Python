import torch
from torch import nn
from collections import OrderedDict
from torchvision import datasets, transforms, models

#selection between vgg13 and vgg16
def selectclassifier(model, hidden_layer):
    
    if (hidden_layer is None):
        hl = 0
    else:
        hl = int(hidden_layer)
        
    if (model == 'vgg13'):
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad=False
        neuralinput = 25088
        neuraloutput = 102
        if (hl <= neuralinput and hl >= neuraloutput):
            classifier = nn.Sequential(OrderedDict([
                ('input_hidden', nn.Linear(neuralinput,hl)),
                ('Activation_ReLU', nn.ReLU()),
                ('Dropout', nn.Dropout(0.2)),
                ('hidden_output', nn.Linear(hl,neuraloutput)),
                ('output', nn.LogSoftmax(dim=1))]))
            model.classifier = classifier
            print(model)
            return model
        else:
            classifier = nn.Sequential(OrderedDict([
                     ('input_hidden', nn.Linear(neuralinput,4096)),
                     ('Activation_ReLU', nn.ReLU()),
                     ('Dropout', nn.Dropout(0.2)),
                     ('hidden_output', nn.Linear(4096,neuraloutput)),
                     ('output', nn.LogSoftmax(dim=1))]))
            model.classifier = classifier
            print(model)
            return model

    if (model == 'vgg16'):
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad=False
        neuralinput = 25088
        neuraloutput = 102
        if (hl <= neuralinput and hl >= neuraloutput):
            classifier = nn.Sequential(OrderedDict([
                     ('input_hidden', nn.Linear(neuralinput,hl)),
                     ('Activation_ReLU', nn.ReLU()),
                     ('Dropout', nn.Dropout(0.2)),
                     ('hidden_output', nn.Linear(hl,neuraloutput)),
                     ('output', nn.LogSoftmax(dim=1))]))
            model.classifier = classifier
            print(model)
            return model
        else:
            classifier = nn.Sequential(OrderedDict([
                ('input_hidden', nn.Linear(neuralinput,4096)),
                ('Activation_ReLU', nn.ReLU()),
                ('Dropout', nn.Dropout(0.2)),
                ('hidden_output', nn.Linear(4096,neuraloutput)),
                ('output', nn.LogSoftmax(dim=1))]))
            model.classifier = classifier
            print(model)
            return model