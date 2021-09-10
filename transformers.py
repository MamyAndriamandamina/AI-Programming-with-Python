import torch
from torchvision import datasets, transforms, models
def load_transformers():
    load_train_transformers()
    load_valid_transformers()
    load_train_datasets()
    print('...transformers loaded')
    
def load_train_transformers():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(224),
                                           transforms.RandomResizedCrop(255),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])
                                         ])
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64,shuffle=True)
    return trainloader

def load_train_datasets():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(224),
                                           transforms.RandomResizedCrop(255),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])
                                         ])
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64,shuffle=True)
    return train_datasets

def load_valid_transformers():
    data_dir = 'flowers'
    valid_dir = data_dir + '/valid'
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.RandomResizedCrop(255),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])
                                         ])
    # Load the datasets with ImageFolder
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    # Using the image datasets and the trainforms, define the dataloaders
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    return validloader
