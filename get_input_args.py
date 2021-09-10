import argparse
from datetime import datetime

def get_input_args():
    dateTimeObj = datetime.now()
    yy=str(dateTimeObj.year)
    mm=str(dateTimeObj.month)
    dd=str(dateTimeObj.day)
    ss=str(dateTimeObj.second)
    mn=str(dateTimeObj.minute)
    hh=str(dateTimeObj.hour)
    checkpoint = 'checkpoint_' + dd + mm + yy + hh + mn + ss +'.pth'
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default = 'saved_models/'+checkpoint+'', help = 'Checkpoint Filename')
    parser.add_argument('--arch', default = 'vgg16', help = 'Pretrained Network [vgg13/vgg16] - default:vgg16')
    parser.add_argument('--learning_rate', default = '0.001', help = 'Learning Rate - default:0.001')
    parser.add_argument('--hidden_units', help = 'Number of Hidden Neural Network - default: based on pretrained network')
    parser.add_argument('--epochs', default = '1', help = 'Number of time an entire dataloader will be scanned - default:1')
    parser.add_argument('--gpu', default = 'gpu', help = 'Processor Type Selection [cpu/gpu] - default: gpu')
    print('...get input completed')
    return parser.parse_args()