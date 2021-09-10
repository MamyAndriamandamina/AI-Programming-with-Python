#Import
from get_input_predict_args import *
from predicting import *

def main():
    
    imagepath = input("Please provide your flowers folder name like this: <parent/child/flower.jpg> [mandatory]: ") 
    checkpoint = input("Please specify your checkpoint folder name like this: <parent/child/checkpoint.pth> [mandatory]: ")

    in_arg = get_input_args()
    
    load_checkpoint(imagepath, checkpoint, in_arg.top_k, in_arg.category_names, in_arg.gpu)

# Call to main function to run the program
if __name__ == "__main__":
    main()