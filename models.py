## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        #I.uniform_(self.conv1.weight)
        
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout
        self.drop1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        #I.uniform_(self.conv2.weight)
        
        self.drop2 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        #I.uniform_(self.conv3.weight)
        
        self.drop3 = nn.Dropout(p=0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 2)
        #I.uniform_(self.conv4.weight)
        
        self.drop4 = nn.Dropout(p=0.4)
        
        
        self.fc1 = nn.Linear(36864, 1000)
        #I.xavier_uniform_(self.fc1.weight)

        self.drop5 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1000, 1000)
        #I.xavier_uniform_(self.fc2.weight)
        
        self.drop6 = nn.Dropout(p=0.6)

        self.fc3 = nn.Linear(1000, 136)
        #I.xavier_uniform_(self.fc3.weight)
        
        
    def forward(self, x):
        
        x = self.drop1(self.pool(F.relu(self.conv1(x))))

        x = self.drop2(self.pool(F.relu(self.conv2(x))))

        x = self.drop3(self.pool(F.relu(self.conv3(x))))

        x = self.drop4(self.pool(F.relu(self.conv4(x))))

        x = x.view(x.size(0), -1)
        
        x = self.drop5(F.relu(self.fc1(x)))

        x = self.drop6(F.relu(self.fc2(x)))

        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    
    
    
    
    
    
    
    