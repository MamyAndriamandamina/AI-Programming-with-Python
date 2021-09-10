import torch
from torchvision import models
from PIL import Image
import glob, os
import numpy as np
import json

def load_checkpoint(path, checkpoint, topk, catname, processor):
    #loading of deep learning model checkpoint
    #loading of saved file
    checkpoint = torch.load(checkpoint)
    #downloading of pretrained model assuming that your checkpoint has been generated using vgg16
    model = models.vgg16(pretrained=True);
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad=False
    #transfer
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.state_dict = checkpoint['state_dict']
    model.optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    print('...checkpoint loaded')
    predict(path, model, topk, processor, catname)
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. 
    shortest_width = 256
    shortest_height = 256
    new_width = 224
    new_height = 224
    
    size = shortest_width, shortest_height

    for infile in glob.glob(image):
        file, ext = os.path.splitext(infile)
        with Image.open(infile) as im:
            im.thumbnail(size)
            #Then you'll need to crop out the center 224x224 portion of the image.
            #get the size of the resized image.
            width, height = im.size
            #crop attributes
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            #crop out the center of the image
            im = im.crop((left, top, right, bottom)) 
            #convert to np.array with floats [0-1]
            np_image = np.asarray(im)/255
            #normalization
            #means and Standard Deviation
            means = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            #You'll want to subtract the means from each color channel, then divide by the standard deviation.
            image = (np_image-means)/std
            #reorder dimension for PyTorch
            image = image.transpose(2,0,1)
            #return image to PyTorch
        return image
    
def predict(path, model, topk, processor, catname):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    # TODO: Implement the code to predict the class from an image file
    if (processor == 'gpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    else:
        device = torch.device("cpu")
        
    
    model.to(device)
    print('processor: '+ str(device))
    #since we are on predicting and not on training, let's use the method evaluation
    model.eval()
    #our array image
    x = process_image(path)
    #Insert a new axis that will appear at the axis position in the expanded array shape.
    x = x[np.newaxis, :]
    #convert from numpy to PyTorch
    image = torch.from_numpy(x).type(torch.FloatTensor).to("cuda" if torch.cuda.is_available() and processor == "gpu" else "cpu")
    # find the probabilities by feed forwarding through the model
    logps = model.forward(image)
    #let's calculate the probabilities distribution
    prob_dist = torch.exp(logps)
    #let's calculate the top k
    top_prob, top_label = prob_dist.topk(int(topk))
    #converting a torch.tensor to np.ndarray
    #let's remove the computational graph of the tensor using the detach() command.
    top_prob = np.array(top_prob.detach())[0]
    top_label = np.array(top_label.detach())[0]
    #get classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_label = [idx_to_class[label] for label in top_label]
    with open(catname, 'r') as f:
        cat_to_name = json.load(f)
    top_flower = [cat_to_name[label] for label in top_label]
    
    x = np.array(list(zip(top_flower,np.round(top_prob,3)*100)))
    print('Below the list of Flower Name with Class Probabilities:\n' + str(x))
  
    
    
