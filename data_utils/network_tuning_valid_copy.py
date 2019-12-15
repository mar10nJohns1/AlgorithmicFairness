# Load functions
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, GRU, Conv2d, Dropout2d, MaxPool2d, BatchNorm2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.utils.data import Dataset, DataLoader

#%matplotlib inline
import argparse
import os
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import math
import pandas as pd

def tune_architecture(layers, activations, IMAGE_SHAPE, conv_out_channels, kernel_size, conv_stride, maxpool, dropout, batchnorm=False):
    
    height, width, channels = IMAGE_SHAPE

    conv_pad    = 0       # <-- Padding

    def conv_dim(dim_size):
        return int(math.ceil(dim_size - kernel_size + 2 * conv_pad / conv_stride + 1))

    conv1_h = conv_dim(height)//maxpool
    conv1_w = conv_dim(width)//maxpool
    # Keep track of features to output layer
    features_cat_size = int(conv_out_channels * conv1_h * conv1_w)

    if layers > 1:
        conv2_h = conv_dim(conv1_h)//maxpool
        conv2_w = conv_dim(conv1_w)//maxpool

        features_cat_size = int(conv_out_channels*2 * conv2_h * conv2_w)

    if layers > 2:
        conv3_h = conv_dim(conv2_h)
        conv3_w = conv_dim(conv2_w)

        features_cat_size = int(conv_out_channels*4 * conv3_h * conv3_w)

    if layers > 3:
        conv4_h = conv_dim(conv3_h)
        conv4_w = conv_dim(conv3_w)

        features_cat_size = int(conv_out_channels*8 * conv4_h * conv4_w)
        print(features_cat_size)
        print(conv3_h)
        print(conv3_w)
        print(conv4_h)
        print(conv4_w)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv_1 = Conv2d(in_channels=channels,
                                 out_channels=conv_out_channels,
                                 kernel_size=kernel_size,
                                 stride=conv_stride,
                                 padding=conv_pad)
            
            self.batch1 = BatchNorm2d(conv_out_channels)
            
            if layers > 1:
                self.conv_2 = Conv2d(in_channels=conv_out_channels,
                                     out_channels=conv_out_channels*2,
                                     kernel_size=kernel_size,
                                     stride=conv_stride,
                                     padding=conv_pad)
                
                self.batch2 = BatchNorm2d(conv_out_channels*2)
           
            if layers > 2: 
                self.conv_3 = Conv2d(in_channels=conv_out_channels*2,
                                     out_channels=conv_out_channels*4,
                                     kernel_size=kernel_size,
                                     stride=conv_stride,
                                     padding=conv_pad)
                
                self.batch3 = BatchNorm2d(conv_out_channels*4)
                
            if layers > 3:
                self.conv_4 = Conv2d(in_channels=conv_out_channels*4,
                                     out_channels=conv_out_channels*8,
                                     kernel_size=kernel_size,
                                     stride=conv_stride,
                                     padding=conv_pad)
                
                self.batch4 = BatchNorm2d(conv_out_channels*8)
            
            self.pool = nn.MaxPool2d(maxpool,maxpool)

            self.dropout = Dropout2d(p=dropout)

            self.l_out = Linear(in_features=features_cat_size,
                                out_features=2,
                                bias=False)

        def forward(self, x_img):
            features = []
            out = {}

            ## Convolutional layer ##
            # - Change dimensions to fit the convolutional layer 
            # - Apply Conv2d
            # - Use an activation function
            # - Change dimensions s.t. the features can be used in the final FFNN output layer
            
            
            features_img = self.pool(activations[0](self.conv_1(x_img)))
            features_img = self.batch1(features_img)
            
            if layers > 1:
                features_img = self.dropout(features_img)
                features_img = self.pool(activations[1](self.conv_2(features_img)))
                if batchnorm:
                    features_img = self.batch2(features_img)
                
            if layers > 2:
                features_img = self.dropout(features_img)
                features_img = activations[2](self.conv_3(features_img))
                if batchnorm:
                    features_img = self.batch3(features_img)
                
            if layers > 3:
                features_img = self.dropout(features_img)
                features_img = activations[3](self.conv_4(features_img))
                if batchnorm:
                    features_img = self.batch4(features_img)
            

            features_img = features_img.view(-1, features_cat_size)

            ## Output layer where all features are in use ##

            out['out'] = self.l_out(features_img)
            return out

    net = Net()
    return net


def accuracy(ys, ts):
    predictions = torch.max(ys, 1)[1]
    correct_prediction = torch.eq(predictions, ts)
    return torch.mean(correct_prediction.float())

    # Function to get label
def get_labels(batch):
    return get_variable(Variable(batch['target']))

# Function to get input
def get_input(batch):
    return {
        'x_img': get_variable(Variable(batch['image']))
    }

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

    
    
    
def tune_train(net, data_train, data_valid, optimizer, learning_rate, weight_decay, batch_size=128,  num_epochs=5, weight_df = None, protected_att = None):
    use_cuda = torch.cuda.is_available()
    print("Running GPU.") if use_cuda else print("No GPU available.")
    
  
    
    criterion = nn.CrossEntropyLoss()

    # weight_decay is equal to L2 regularization
    if optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay, lr=learning_rate)
    else:
        optimizer = optim.RMSprop(net.parameters(), weight_decay=weight_decay, lr=learning_rate)
    

    # Initialize lists for training and validation
    train_iter = []
    train_loss, train_accs = [], []
    valid_iter = []
    valid_loss, valid_accs = [], []

    # Generate batches
    batch_gen_train = DataLoader(data_train, batch_size, shuffle=True, num_workers=6)

    # Train network
    net.train()
    j=0
    for epoch in range(num_epochs):
        print('Epoch: ', epoch)
        for i, batch_train in enumerate(batch_gen_train):
            
            if j%300 == 0:
                l, a = tune_valid(net, data_valid, batch_size)
                valid_loss.append(l)
                valid_accs.append(a)
                net.train()
             # Train network
            output = net(**get_input(batch_train))
            labels_argmax = torch.max(get_labels(batch_train), 1)[1]
            

            if weight_df is not None:
                protected = batch_train['attributes'][:,protected_att]
                l = [str(int(get_numpy(i))) + str(get_numpy(j)) for i, j in zip(protected, labels_argmax)]
                l_df = pd.DataFrame(l, columns=['Group'])
                batch_weights = pd.merge(l_df,weight_df,how='left',on='Group')
                loss = criterion(output['out'], labels_argmax)
                loss_weighted = torch.FloatTensor(batch_weights.iloc[:,protected_att+1].values)*loss
                batch_loss = torch.mean(loss_weighted) # Average across a batch
            else:
                batch_loss = criterion(output['out'], labels_argmax)


            train_iter.append(j)
            train_loss.append(float(get_numpy(batch_loss)))
            train_accs.append(float(get_numpy(accuracy(output['out'], labels_argmax))))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            j+=1
            
    return net, train_loss, train_accs, valid_loss, valid_accs


def tune_valid(net, data_valid, batch_size):
    batch_gen_valid = DataLoader(data_valid, batch_size, shuffle=True, num_workers=6)
    
    criterion = nn.CrossEntropyLoss()
    
    net.eval()
    val_losses, val_accs, val_lengths = 0, 0, 0
    with torch.no_grad(): 
        for batch_valid in batch_gen_valid:
            num = len(batch_valid['target'])
            output = net(**get_input(batch_valid))
            labels_argmax = torch.max(get_labels(batch_valid), 1)[1]
            val_losses += criterion(output['out'], labels_argmax) * num
            val_accs += accuracy(output['out'], labels_argmax) * num
            val_lengths += num

        # Divide by the total accumulated batch sizes
        val_losses /= val_lengths
        val_accs /= val_lengths
        
    return val_losses.item(), val_accs.item()