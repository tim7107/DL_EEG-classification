####################################Import####################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data


#####################################Training settings#####################################
parser = argparse.ArgumentParser(description='BCI dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (default: 164)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                    help='SGD momentum (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#####################################Start of Def#####################################
"""
   date loader
"""
def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    #(1080, 1, 2, 750) (1080,) (1080, 1, 2, 750) (1080,)
    return train_data, train_label, test_data, test_label
    

"""
   EEGNet
"""
class EEGNet(nn.Module):   
    def __init__(self,activation_func):
        super(EEGNet, self).__init__()
        
        ######Layer 1#####
        #nn.Conv2d(input_cannel,output_channel,kernel_size,stride,padding,bias)
        #nn.batchnorm1(num_features,eps,momentum,affine,track_running_stats)
        self.conv1 = nn.Sequential(
                nn.Conv2d(1 , 16 , kernel_size=(1,51) , stride=(1,1) , padding=(0,25) , bias =False),
                nn.BatchNorm2d(16, eps=1e-05 , momentum=0.1 , affine=True, track_running_stats=True)
        )
        
        ######Layer 2#####
        self.conv2 = nn.Sequential(
                nn.Conv2d(16 , 32 ,kernel_size=(2,1) , stride=(1,1) , groups=16 , bias =False),
                nn.BatchNorm2d(32, eps=1e-05 , momentum=0.1 , affine=True, track_running_stats=True),
                activation_func(),
                nn.AvgPool2d(kernel_size=(1,4) , stride=(1,4) , padding =0),
                nn.Dropout( p = 0.25 )
        )
        
        ######Layer 3#####
        self.conv3 =  nn.Sequential(
                nn.Conv2d(32 , 32 , kernel_size=(1,15) , stride=(1,1) , padding=(0,7) , bias =False),
                nn.BatchNorm2d(32, eps=1e-05 , momentum=0.1 , affine=True, track_running_stats=True),
                activation_func(),
                nn.AvgPool2d(kernel_size=(1,8) , stride=(1,8) , padding =0),
                nn.Dropout( p = 0.25 )
        )
    
        self.fc1 = nn.Sequential(
                nn.Linear(in_features=736 , out_features=2, bias=True)
        ) 
    def forward(self, x):
        ######First Conv######
        x = self.conv1(x)
        ######Depthwise Conv######
        x = self.conv2(x)
        ######Separable Conv######
        x = self.conv3(x)     
        ######Classify layer######
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
"""
   learning rate scheduling
"""
def adjust_learning_rate(optimizer, epoch):
    lr=1e-2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
"""
   training function
"""
def train(epoch,net):
    net.train()
    adjust_learning_rate(optimizer, epoch)
    train_loss=0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)   
        target = target.to('cuda').long()
        optimizer.zero_grad()
        output = net(data)
        loss = Loss(output, target)

        train_loss += loss.data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.data[0]))    
    temp=100.*correct.item() / len(train_loader.dataset)
    return temp

"""
   testing function
"""                
def test(epoch,net):
    net.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        target = target.to('cuda').long()
        with torch.no_grad():
        	output = net(data)
        test_loss += Loss(output, target).data[0]
        #the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct.item() / len(test_loader.dataset)))
    temp=100. * correct.item() / len(test_loader.dataset)
    return temp
"""
   Execution of three activation_function(activation_func)
"""
def execute(net,string):
    train_accuracy=[]
    test_accuracy=[]
    for epoch in range(1, args.epochs + 1):
        train_temp=train(epoch,net)
        test_temp=test(epoch,net)
        train_accuracy.append(train_temp)
        test_accuracy.append(test_temp)
        
    final_accuracy=test_accuracy[-1]
    
    plt.title('EEGNet' ,fontsize=14)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(test_accuracy , label= string +'_' +'testingdata')
    plt.legend(loc='best')
    plt.plot(train_accuracy, label = string +'_' + 'trainingdata')
    plt.legend(loc='best') 
    return final_accuracy
#####################################End of Def#####################################

#####################################Start coding#####################################
train_data ,train_label , test_data , test_label = read_bci_data()
Loss = nn.CrossEntropyLoss()


train_data = torch.Tensor(train_data)
train_label = torch.Tensor(train_label)
torch_dataset=Data.TensorDataset(train_data,train_label)        
train_loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
)
test_data = torch.Tensor(test_data)
test_label = torch.Tensor(test_label)
torch_dataset1=Data.TensorDataset(test_data,test_label)        
test_loader=Data.DataLoader(
    dataset=torch_dataset1,
    batch_size=64,
    shuffle=True,
    num_workers=2,
)

#----------------------------------------------------------------
#EEGNET_ReLU
EEGNet_ReLU = EEGNet(activation_func = nn.ReLU)
if args.cuda:
    device = torch.device('cuda')
    EEGNet_ReLU.to(device)
optimizer = optim.SGD(EEGNet_ReLU.parameters(), lr=args.lr)
ReLU_accuracy=execute(net = EEGNet_ReLU ,string = 'ReLU')

#EEGNET_LeakyReLU
EEGNet_LeakyReLU = EEGNet(activation_func = nn.LeakyReLU)
if args.cuda:
    device = torch.device('cuda')
    EEGNet_LeakyReLU.to(device)
optimizer = optim.SGD(EEGNet_LeakyReLU.parameters(), lr=args.lr)
LeakyReLU_accuracy=execute(net = EEGNet_LeakyReLU ,string = 'LeakyRELU')


#EEGNET_ELU
EEGNet_ELU = EEGNet(activation_func = nn.ELU)
if args.cuda:
    device = torch.device('cuda')
    EEGNet_ELU.to(device)
optimizer = optim.SGD(EEGNet_ELU.parameters(), lr=args.lr)
ELU_accuracy=execute(net = EEGNet_ELU ,string = 'ELU')
   
#np.save('relutest',accuracytest)
#np.save('relutrain',accuracytrain)

plt.show()

print("ReLU accuracy= %f"%(ReLU_accuracy) +"%")
print("LeakyReLU accuracy= %f" %(LeakyReLU_accuracy)+"%")
print("ELU accuracy= %f" %(ELU_accuracy)+"%")
