import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import random
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#FILE = "report_model_fold3.pth"
FILE = "this_is_fold1_report.pth"
input_size = 36
hidden_size = 31
num_layers = 2
num_classes = 7
classification_threshold = 0.6
batch_size = 1

class ActionDataset(Dataset):
    
    def __init__(self):
        #data loading
        data_npz = np.load("./KFold_data/KFoldtraining_data.npz")
        training_data = data_npz['training_data']
        training_data = np.float32(training_data)
        self.x = torch.from_numpy(training_data[:, :-1])
        self.y = torch.from_numpy(training_data[:,-1,0])
        self.n_samples = training_data.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.n_samples
    

# create dataset and split into folds
dataset = ActionDataset()
dataset_size = int(round((len(dataset))))
dataset_list = list(range(dataset_size))
test_size = int(dataset_size / 4)
test_list = []
sublists = [dataset_list[i:i+(test_size)] for i in range(0, len(dataset_list), (test_size))]
val_idx = sublists[0]
test_sampler= SubsetRandomSampler(val_idx)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
print(f"  Test:  index={val_idx[0]} to index={val_idx[-1]}")
   
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.05, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
        out, _ = self.gru(x, h0)  
        out = out[:, -1, :]      
        out = self.fc(out)
        return out    
    
loaded_model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()   
classes = ["handgun","laying","running","sitting","standing","walking","waving"]  

TPR = {'TPR0': [1], 'TPR1': [1], 'TPR2': [1], 'TPR3': [1],'TPR4': [1], 'TPR5': [1], 'TPR6': [1], 'TPR7': [1]}
FPR = {'FPR0': [1], 'FPR1': [1], 'FPR2': [1], 'FPR3': [1],'FPR4': [1], 'FPR5': [1], 'FPR6': [1], 'FPR7': [1]}

samples = 20

for count in range(0,samples):
    threshold = np.around(0+((1/samples)*count),3)
    print(f'Threshold Value: {threshold}')
    
    TP = {'TP0': 0, 'TP1': 0, 'TP2': 0, 'TP3': 0,'TP4': 0, 'TP5': 0, 'TP6': 0, 'TP7': 0}
    FP = {'FP0': 0, 'FP1': 0, 'FP2': 0, 'FP3': 0,'FP4': 0, 'FP5': 0, 'FP6': 0, 'FP7': 0}
    TN = {'TN0': 0, 'TN1': 0, 'TN2': 0, 'TN3': 0,'TN4': 0, 'TN5': 0, 'TN6': 0, 'TN7': 0}
    FN = {'FN0': 0, 'FN1': 0, 'FN2': 0, 'FN3': 0,'FN4': 0, 'FN5': 0, 'FN6': 0, 'FN7': 0}
    
    with torch.no_grad():  
       for frames_test, labels_test in test_loader:
           frames_test = frames_test.to(device)
           labels_test = labels_test.to(device)
           output_test = loaded_model(frames_test)
           '''          
           output_test = (torch.max(torch.exp(output_test), 1)[1]).detach().cpu().numpy()
           output_test = [int(output_test) for output_test in output_test]
           labels_test = labels_test.detach().cpu().numpy()
           labels_test = [int(labels_test) for labels_test in labels_test]
           '''

           output_softed = torch.softmax(output_test, dim=1)
           max_value, predicted = torch.max(output_softed.data, 1)
           output_test = np.around(output_softed.cpu().detach().numpy(), 5)
           truth = int(labels_test.cpu().detach().numpy().item())
           predicted_num = int(predicted.cpu().detach().numpy().item())
           #print(f'truth: {truth}')
           #print(f'predicted: {predicted}')
           output_test = output_test[0]
           j = truth
           if output_test[j] > threshold and predicted_num == truth:
                   TP[(f'TP{j}')] += 1
           elif output_test[j] > threshold and predicted_num != truth:
                   FP[(f'FP{j}')] += 1
           elif output_test[j] <= threshold and predicted_num == truth:
                   TN[(f'TN{j}')] += 1
           elif output_test[j] <= threshold and predicted_num != truth:
                   FN[f'FN{j}'] += 1
           else:
               print('lol')
           
    total_FN = sum(FN.values())
    total_FP = sum(FP.values())
    for j in range(len(classes)):
        if not (TP[(f'TP{j}')]+(FN[(f'FN{j}')])) == 0 or (TP[(f'TP{j}')]+(FP[(f'FP{j}')])) == 0:          
            TPR[(f'TPR{j}')].append(TP[(f'TP{j}')]/(TP[(f'TP{j}')]+(FN[(f'FN{j}')])))
            FPR[(f'FPR{j}')].append((FP[(f'FP{j}')])/(TP[(f'TP{j}')]+(FP[(f'FP{j}')])))
        else:
            print('   Threshold skipped due to divde by zero error')
            
        
for j in range(len(classes)):
    TPR[(f'TPR{j}')].append(0)
    FPR[(f'FPR{j}')].append(0)
    print(TPR[(f'TPR{j}')])
    
normx = [0,1]
normy = [0, 1]
fig, ax = plt.subplots(1)
ax.plot(FPR['FPR0'],TPR['TPR0'],'c--x')
ax.plot(FPR['FPR1'],TPR['TPR1'],'r--x')
ax.plot(FPR['FPR2'],TPR['TPR2'],'b--x')
ax.plot(FPR['FPR3'],TPR['TPR3'],'g--x')
ax.plot(FPR['FPR4'],TPR['TPR4'],'y--x')
ax.plot(FPR['FPR5'],TPR['TPR5'],'m--x')
ax.plot(FPR['FPR6'],TPR['TPR6'],'c--x')
ax.plot(normx,normy,'k:')
ax.legend(["Handgun","Lying","Running","Sitting","Standing","Walking","Waving","Random Classifier"])
ax.set_xlim(-0.1,1.1)
ax.set_ylim(-0.10,1.1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Fold 1 ROC Curve')
ax.grid(True)

plt.show
    