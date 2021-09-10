import argparse
from datetime import datetime

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', default = 5, help = 'Return top K most likely classes')
    parser.add_argument('--category_names', default = 'cat_to_name.json', help = 'Use of mapping of categories to real names')
    parser.add_argument('--gpu', default = 'gpu', help = 'Use GPU for inference')
    print('...get input completed')
    return parser.parse_args()