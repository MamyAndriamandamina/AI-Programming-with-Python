import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

def training(epoch, processor, model, trainloader, validloader, traindataset, learning_rate, filename):
    #processor parameter
    if (processor == 'gpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print('processor: '+str(device))
    #parameters for model training and testing
    criterion = nn.NLLLoss()
    #optimizer to update weights with gradients
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(learning_rate))
    model.to(device)
    #pass through the datasets
    epochs = int(epoch)
    steps = 0
    print("Start of Training...\n")
    print_every = 5
    for epoch in range(epochs):
        running_loss = 0
        #Train the classifier layers using backpropagation using the pre-trained network to get the features
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            #feed forward pass
            logps = model.forward(inputs)
            #loss calculation
            loss = criterion(logps, labels)
            #reinitialize the gradient to zero
            optimizer.zero_grad()
            #back propagation pass
            loss.backward()
            #update of weights
            optimizer.step()

            running_loss += loss.item()
            #training ends here
            #Track the loss and accuracy on the validation set to determine the best hyperparameters
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train loss: {running_loss/print_every:.3f} | "
                      f"Validation loss: {test_loss/len(validloader):.3f} | "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    print("\n End of Training...")
    savecheckpoint(epochs, optimizer, filename,traindataset,model)

def savecheckpoint(epochs, optimizer, filename, traindataset, model):
    model.class_to_idx = traindataset.class_to_idx
    checkpoint = {'epochs':epochs,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer':optimizer.state_dict()
                 }
    torch.save(checkpoint, filename)
    
    print("\n Checkpoint available in {}".format(filename))