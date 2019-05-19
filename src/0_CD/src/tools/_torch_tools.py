import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import csv

# files expects a list of filenames
def load(files, typeF=None, channels_last=False):
    out = []

    for f in files:
        print("Loading",f)
        x = np.load(f)

        if typeF is not None:
            x = x.astype(typeF)
            x /= 255

        if channels_last is True:
            x = x.swapaxes(1,3)
            x = x.swapaxes(1,2)

        out.append(x)

    return out

def loadData(folder, train=False, val=False, test=False, typeF=None, channels_last=False):
    names = []

    if train is True:
        names.append(folder+"X_train.npy")
        names.append(folder+"y_train.npy")

    if val is True:
        names.append(folder+"X_val.npy")
        names.append(folder+"y_val.npy")

    if test is True:        
        names.append(folder+"X_test.npy")
        names.append(folder+"y_test.npy")

    return load(names, typeF=typeF, channels_last=channels_last)

def plotHistory(history, save=None, size=(10,10)) :
    """ Plots a graph of dictionary of lists.
    """
    plt.figure(figsize=size)
    plt.grid(True)

    for key, val in history.items():
        assert type(val) == list
        plt.plot(range(1,len(val)+1),val,label=key)
    plt.legend(loc='upper right')#, prop={'size': 24})

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

def toCSV(file, d):
    """ Awaits dictionary of lists on the input.
        Saves it as a .CSV file.
    """
    with open(file, 'w') as f:
        w = csv.writer(f)
        w.writerow(d.keys())
        w.writerows(zip(*d.values()))

# =============================================================================

class Training():
    def __init__(self, model, device, X, y, X_val=None, y_val=None,
        loss_function=None, optimizer=None):

        # Dataset as numpy array
        self.setTrain(X,y)

        if X_val is not None:
            self.setVal(X_val, y_val)
        else:
            self.X_val = None
            self.y_val = None


        # Model stuff
        self.model = model
        self.device = device

        self.history = {
            'train' : [],
            'val' : []
        }

        self.loss_function = loss_function
        if self.loss_function is None:
            self.loss_function = nn.L1Loss()

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adamax(model.parameters())

        # Load model on GPU if accessible
        self.model.to(self.device)
    # -----------------------------------------------------------------------------
    def getBatch(self, offset, batch_size, val=False):
        """ Datasets are stored as numpy arrays. Method getBatch()
        returns a batch from the right dataset as a pytorch tensor
        on self.device.
        """
        if val is True:
            X = self.X_val
            y = self.y_val
        else:
            X = self.X
            y = self.y

        input = torch.autograd.Variable(
            torch.tensor( X[ offset:offset + batch_size ], dtype=torch.float )
        )
        target = torch.autograd.Variable(
            torch.tensor( y[ offset:offset + batch_size ], dtype=torch.float )
        )
        return input.to(self.device), target.to(self.device)
    # -----------------------------------------------------------------------------
    def fit(self, batch_size, n_epochs, val=False, save=None, save_off=0):
        """ Mehod fit() trains the model on the training part of the dataset for
        n_epochs epochs. If val==True, than the model is validated after each
        epoch. If save is path to a folder, model weights are save in that
        folder after each epoch. Save_off denotes how many epochs were trained
        before this run of fit() for the naming purposes.
        """
        
        print("\n\n====== TRAINING ======")
        
        n_batch = self.X.shape[0] // batch_size
        
        start_T = int(time.time())
        
        for epoch in range(save_off+1,n_epochs+save_off+1):
            print("===> Epoch[{}]".format(epoch), end='', flush=True)
            epoch_T = time.time()
            epoch_loss = 0
            
            for it in range(n_batch):
                input, target = self.getBatch(it*batch_size, batch_size)
                self.optimizer.zero_grad()
                
                output = self.model(input)

                O = torch.cat((output,output,output),1).to(self.device)
                T = torch.cat((target,target,target),1).to(self.device)

                loss = self.loss_function(O, T)
                loss.backward()
                self.optimizer.step()
                
                loss_train = loss.item()
                epoch_loss += loss_train
                
                tick_T = time.time()
                print("\r", end='')
                print("===> Epoch[{}]({}/{}): Loss: {:.4f}\tETA {}\tEpoch Loss: {:.4f}"
                      .format(epoch, it + 1, n_batch, loss_train,
                      self.formatTime((tick_T - epoch_T) / (it + 1) * (n_batch - it + 1)),
                      epoch_loss / (it+1)), end='', flush=True)
                
            epoch_loss /= n_batch
            self.history['train'].append(epoch_loss)
            
            # If save argument is defined, save the weights after each epoch
            if save is not None:
                self.save(save+"weights"+str(epoch).zfill(2)+".pth")
            
            print("\nEpoch[{}] finished in {} with loss {}".format(epoch, self.formatTime(tick_T - epoch_T), epoch_loss))
            
            if val is True:
                self.validate(batch_size)
            
            print("\n----------------------------\n")
        print("Finished training of {} epochs in {}.".format(n_epochs, self.formatTime(int(time.time())-start_T)))
            
        return self.history
    # -----------------------------------------------------------------------------
    def validate(self, batch_size):
        if self.X_val is None:
            print("Cannot validate, no validation dataset given.")
            return None

        loss_val = 0
        n_batch_val = self.X_val.shape[0] // batch_size

        print("Validating on {} samples.".format(n_batch_val * batch_size))

        start_T = int(time.time())
        for it in range(n_batch_val):
            input, target = self.getBatch(it*batch_size, batch_size, val=True)

            output = self.model(input)
            loss = self.loss_function(output, target)
            loss_val += loss.item()

            tick_T = time.time()
            print("\r", end='')
            print("===> Validating ({}/{}):\tETA {}\tValidation Loss: {:.4f}"
                      .format(it + 1, n_batch_val,
                      self.formatTime((tick_T - start_T) / (it + 1) * (n_batch_val - it + 1)),
                      loss_val / (it+1)), end='', flush=True)

        print("\nValidation loss = {:.4f}".format(loss_val / n_batch_val))
        self.history['val'].append( loss_val / n_batch_val )

        return loss_val / n_batch_val
    # -----------------------------------------------------------------------------
    def setTrain(self, X, y):
        assert type(X) == type(y) == np.ndarray
        assert X.shape[0] == y.shape[0]
        assert X.shape[2:4] == y.shape[2:4]

        self.X = X
        self.y = y
    # -----------------------------------------------------------------------------
    def setVal(self, X_val, y_val):
        assert type(X_val) == type(y_val) == np.ndarray
        assert X_val.shape[0] == y_val.shape[0]
        assert X_val.shape[2:4] == y_val.shape[2:4]

        self.X_val = X_val
        self.y_val = y_val
    # -----------------------------------------------------------------------------
    def save(self, name="weights.pth"):
        """Save the weights of the model."""
        torch.save(self.model.state_dict(), name)
    # -----------------------------------------------------------------------------    
    @staticmethod
    def formatTime(t):
        """Takes t as number of seconds, returns formatted string as HH:MM:SS"""
        t = int(t)
        s = t % 60
        m = (t // 60) % 60
        h = t // 3600
        return str(h) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2)