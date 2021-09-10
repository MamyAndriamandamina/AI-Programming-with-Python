#Import
from get_input_args import *
from transformers import *
from selectclassifier import selectclassifier
from training import training

def main():
    #function that run the training process
    in_arg = get_input_args()
    load_transformers()
    train_loader = load_train_transformers()
    valid_loader = load_valid_transformers()
    train_dataset = load_train_datasets()
    selectclassifier(in_arg.arch, in_arg.hidden_units)
    model = selectclassifier(in_arg.arch, in_arg.hidden_units)
    training(in_arg.epochs, in_arg.gpu, model, train_loader, valid_loader, train_dataset, in_arg.learning_rate, in_arg.save_dir)
# Call to main function to run the program
if __name__ == "__main__":
    main()