import numpy as np
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score

#Check whether GPU is avaliable, else it will use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

import assignment_ann as a4
original_data = a4.SignalDataset() # Provides back a PyTorch Dataset

# Task C 

# Define a more generalized function for training, to reuse the code
def task_c_train(original_data,**kwargs):

    def evaluate_model_performance(dataset, model):
        # Function that takes in a model and a dataset
        # and outputs an performance estimate of the classification
        # accuracy of the model.
        # Make a Dataloader for the dataset.
        # Note, we are not performing any SGD here, so our batch
        # size is the whole # dataset we want to evaluate model performance on.
        d_loader = DataLoader(dataset = dataset, batch_size=len(dataset))
        cost_function = nn.CrossEntropyLoss() # For classification evaluation
        model.eval()
        # Make predictions for the eval dataset
        with torch.no_grad():
            for X, y in d_loader:
                X = X.to(device)
                y = y.to(device)
                raw_y_preds = model(X).to(device)
            y_class_preds = raw_y_preds.argmax(dim=1)
            eval_cost = cost_function(raw_y_preds, y).item()
        model.train()
        # compare predictions with true labels and compute performance metric
        # performance metric in this example is classification accuracy
        y_class_preds = y_class_preds.cpu().numpy()
        y = y.cpu().numpy()
        eval_acc = accuracy_score(y_pred = y_class_preds, y_true = y)
        return eval_cost, eval_acc  
  
    batch_size = kwargs['batch_size']
    size_of_original_data = len(original_data)

    # Specify split fractions
    train_fraction = kwargs['train_fraction']
    val_fraction = kwargs['val_fraction']

    # Determine size of each set
    train_dataset_size = int(train_fraction * size_of_original_data)
    val_dataset_size = int(val_fraction * size_of_original_data)
    test_dataset_size = int(size_of_original_data - train_dataset_size - val_dataset_size)

    # Split whole original data into train, val and test datsets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(original_data,
    [train_dataset_size,
    val_dataset_size,
    test_dataset_size])

    # It will use the external test dataset if it has been given by user,
    # no matter what the test fraction is
    try:
        test_dataset = kwargs['external_test_data']
    except KeyError:
        pass  # do nothing if external_test_data is not given

    cost_function = nn.CrossEntropyLoss()

    model = kwargs['model'].to(device)

    optim = torch.optim.SGD(model.parameters(), lr=kwargs['lr'] )

    training_minibatch_Js =[]
    history_cross_entropy_train,history_cross_entropy_valid = [],[]
    history_accuracy_train,history_accuracy_valid = [],[]
    history_epoch_i = [] #records the index of epoch that evaluated 
    
    nr_epochs = kwargs['nr_epochs']
    
    eval_every_kth = kwargs['eval_every_kth']
    
    for epoch_i in range(nr_epochs):
        
        if epoch_i%eval_every_kth ==0:
            model.eval()
            train_cost, train_acc = evaluate_model_performance(model=model, dataset = train_dataset)
            val_cost, val_acc = evaluate_model_performance(model=model, dataset = val_dataset)
            history_cross_entropy_train.append(train_cost)
            history_cross_entropy_valid.append(val_cost)            
            history_accuracy_train.append(train_acc)
            history_accuracy_valid.append(val_acc)
            history_epoch_i.append(epoch_i)
            print(f'Epoch:{epoch_i} - Train cost:{train_cost},- Train acc:{train_acc}')
            print(f'Epoch:{epoch_i} - Val cost:{val_cost},- Val acc:{val_acc}') 
            print('..............')
            model.train()
            
        for x_batch, y_batch in DataLoader(train_dataset,batch_size=batch_size, shuffle=True):
#            with torch.no_grad():
            # Move everything to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            raw_y_preds = model(x_batch).to(device) # predict model output for X batch


            y_class_preds = raw_y_preds.argmax(dim=1) #compute class labels for each x_n
            cost = cost_function(raw_y_preds, y_batch).to(device)# Compute cost
            optim.zero_grad() # zero the grads of all model params
            cost.backward() # compute J gradient of all model params
            training_minibatch_Js.append(cost)
            optim.step() # take one update step for all model params

    #Final evaluation on test dataset
    model.eval()
    test_cost, test_acc = evaluate_model_performance(model=model, dataset = test_dataset)
    print(f'Final Epoch:{epoch_i} - Test cost:{test_cost},- Test acc:{test_acc}')
           
    #Plotting cost  
    plt.figure(figsize=[10,5])
    plt.plot(training_minibatch_Js)
    plt.xlabel('update step i on mini-batch')
    plt.ylabel('Cost')
    plt.title('Cost during training on train-set (per mini-batch)')
    plt.grid()

    plt.figure(figsize=[10,5])
    plt.plot(history_epoch_i,history_cross_entropy_train,label='train set')
    plt.plot(history_epoch_i,history_cross_entropy_valid,label='validation set')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Cross-entropy cost')
    plt.title('Cross-entropy cost throughtout training')
    plt.grid()
   
    plt.figure(figsize=[10,5])
    plt.plot(history_epoch_i,history_accuracy_train,label='train set')
    plt.plot(history_epoch_i,history_accuracy_valid,label='validation set')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy performance throughout training')
    plt.grid()

    plt.show()    
 
    print(f'cost of last step:{training_minibatch_Js[-1]}')

if __name__ == '__main__': #a magic method in Python, this section will not run when you import the .py file
    #Example for Task D

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
    )

    task_c_train(training_data,
    external_test_data = test_data,
    train_fraction =0.8,
    val_fraction = 0.2,
    eval_every_kth = 10,
    batch_size = 256,
    nr_epochs = 20,
    lr = 0.1,
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=300),
        nn.ReLU(),    
        nn.Linear(in_features=300, out_features=100),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=50),
        nn.ReLU(),
        nn.Linear(in_features=50, out_features=10),
    ))

#Example for Task C
'''
    task_c_train(original_data,
    train_fraction =0.7,
    val_fraction = 0.2,
    eval_every_kth = 10,
    batch_size = 200,
    nr_epochs = 20,
    lr = 0.1,
    model = nn.Sequential(
        nn.Linear(in_features=500, out_features=300),
        nn.ReLU(),    
        nn.Linear(in_features=300, out_features=100),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=50),
        nn.ReLU(),
        nn.Linear(in_features=50, out_features=2),
    ))
'''

