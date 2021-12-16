import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import random_split
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.models as models
import sys
import matplotlib.pyplot as plt
from PIL import Image

class CNNet(nn.Module):
    
    def __init__(self):
        super(CNNet,self).__init__()
        self.cnn =  nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride =1,padding=1),
                                  nn.BatchNorm2d(8),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.LeakyReLU(),
                                  nn.MaxPool2d(2, 2),
                                 
                                  nn.Conv2d(16, 32, kernel_size=3, stride =1,padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.LeakyReLU(),
                                  nn.MaxPool2d(2, 2),
    
                                  nn.Flatten(),
                                  nn.Linear(16 * 16 * 16, 128),
                                  nn.LeakyReLU(),
                                  nn.Linear(128, 64),
                                  nn.LeakyReLU(),
                                  nn.Linear(64, 4))
        
    def forward(self, x):
        return self.cnn(x)
    

def classifier(filename):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CNNet()
    model.load_state_dict(torch.load('./model_params.pkl', map_location=device))
    model.eval()

    # filename = '../input/african-wildlife/rhino/001.jpg'
    input_image = Image.open(filename)
    plt.imshow(input_image)
    plt.show()
    preprocess = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        model.to(device)
        output = model(input_batch.to(device))

    wildlife = {0: 'buffalo', 1: 'elephant', 2: 'rhino', 3: 'zebra'}
    probabilities = F.softmax(output, dim=1)
    print(wildlife[probabilities.argmax(-1).cpu().numpy()[0]])



filename = ''
classifier(filename)