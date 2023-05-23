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
from sklearn.metrics import confusion_matrix, classification_report
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_classes = 7
num_epochs = 4
batch_size = 128
learning_rate = 0.005 #0.002

input_size = 36
sequence_length = 240 #number of frames
hidden_size = 31
num_layers = 2
classification_threshold = 0.6
k = 4


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
test_size = int(dataset_size / k)
test_list = []


#random.shuffle(dataset_list)
sublists = [dataset_list[i:i+(test_size)] for i in range(0, len(dataset_list), (test_size))]
# print the sublists
#for i, sublist in enumerate(sublists):
    #print(f"Sublist {i+1}: {sublist}")
    

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



# Loss and optimizer
criterion = nn.CrossEntropyLoss()

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def train_epoch(model,device,train_loader,criterion,optimizer, classification_threshold):
    model.train()
    train_loss, train_correct = 0, 0
    for i, (frames, labels) in enumerate(train_loader):  
        
        frames = frames.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(frames) #calculate output of the model for the given batch
        labels = labels.to(torch.int64)
        loss = criterion(outputs, labels)
        train_loss += loss.item() * frames.size(0)
        _ , predictions = torch.max(outputs.data, 1)
        train_correct += (predictions == labels).sum().item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return(train_loss, train_correct)

def valid_epoch(model, device, test_loader, criterion, classification_threshold):
    model.eval()
    n_dev_correct, n_dev_samples = 0, 0
    with torch.no_grad():
        valid_loss = 0
        for i, (frames_dev, labels_dev) in enumerate(test_loader):
            
            frames_dev = frames_dev.to(device)
            labels_dev = labels_dev.to(device)
            labels_dev = labels_dev.to(torch.int64)
            outputs_dev = model(frames_dev)
            loss_dev = criterion(outputs_dev,labels_dev)
            outputs_dev_soft = torch.softmax(outputs_dev, dim=1)
            max_value, predicted = torch.max(outputs_dev_soft.data, 1)
            valid_loss +=loss_dev.item()*frames_dev.size(0)
            
            n_dev_samples += labels_dev.size(0)
            n_dev_correct += (predicted == labels_dev).sum().item()
        acc = round(100.0 * n_dev_correct / n_dev_samples, 3)
        print(f'Accuracy: {acc}%')
        print(f'Log-Loss: {np.log(loss_dev.item()):.4f}')
    
    return(valid_loss, n_dev_correct)

def findMissingNumbers(n):
    numbers = set(n)
    length = len(n)
    output = []
    for i in range(1, int(n[-1])):
        if i not in numbers:
            output.append(i)
            print(output)
    return output


history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
y_pred = {'y0_pred': [], 'y1_pred': [], 'y2_pred': [], 'y3_pred': []}
y_true = {'y0_true': [], 'y1_true': [], 'y2_true': [], 'y3_true': []}

for fold in range(k): #enumerate(splits.split(np.arange(len(dataset)))):
    

    val_idx = sublists[fold]
    train_idx = [elem for i, elem in enumerate(dataset_list) if i not in val_idx]
              
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler= SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    print(f'Fold: {fold+1}')
    #print(f"  Train: index={train_idx}")
    print(f"  Test:  index={val_idx[0]} to index={val_idx[-1]}")
    print(f'Number of samples: {len(dataset)}')
    print(f'Number of training samples: {len(train_loader.sampler.indices)}')
    print(f'Number of test samples: {len(test_loader.sampler.indices)}')
    
    
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    model.to(device)
    model.apply(reset_weights)
    
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))  
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) 
    
            
    for epoch in range(num_epochs):        
        print (f'\nFold [{fold+1}/{k}]   Epoch [{epoch+1}/{num_epochs}]')
        train_loss, train_correct = train_epoch(model,device,train_loader,criterion,optimizer, classification_threshold)
        test_loss, test_correct = valid_epoch(model, device, test_loader, criterion, classification_threshold)
        scheduler.step()
        
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    
    #create confusion matrix
    with torch.no_grad():  
       for frames_test, labels_test in test_loader:
           frames_test = frames_test.to(device)
           labels_test = labels_test.to(device)
           output_test = model(frames_test)
           output_test = (torch.max(torch.exp(output_test), 1)[1]).detach().cpu().numpy()#.np()
           output_test = [int(output_test) for output_test in output_test]
           y_pred[(f'y{fold}_pred')].extend(output_test)
           
           labels_test = labels_test.detach().cpu().numpy()
           labels_test = [int(labels_test) for labels_test in labels_test]
           y_true[(f'y{fold}_true')].extend(labels_test)
    
           
    FILE = (f'model_fold{fold+1}.pth')
    torch.save(model.state_dict(), FILE)
    
    
    #missing = findMissingNumbers(y_true)
    #if missing:
       # y_true[(f'y{fold}_true')].extend(missing)
        #y_pred[(f'y{fold}_pred')].extend(missing)
        
classes = ("Handgun","Laying","Running","Sitting","Standing","Walking","Waving")
lol = plt.figure(figsize=(10,9))
cf_matrix = confusion_matrix(y_true['y0_true'], y_pred[('y0_pred')])
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
heat0 = plt.subplot(221)
sn.heatmap(df_cm, annot=True, ax=heat0)  
heat0.set_xlabel('Predicted Labels')
heat0.set_ylabel('True Labels')
heat0.set_title('Fold 1')
report0 = classification_report(y_true['y0_true'], y_pred[('y0_pred')],target_names=classes)
print(report0)

cf_matrix = confusion_matrix(y_true[('y1_true')], y_pred[('y1_pred')])
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
heat1 = plt.subplot(222)
sn.heatmap(df_cm, annot=True, ax=heat1)   
heat1.set_xlabel('Predicted Labels')
heat1.set_ylabel('True Labels')
heat1.set_title('Fold 2')
report1 = classification_report(y_true['y1_true'], y_pred[('y1_pred')],target_names=classes)
print(report1)

cf_matrix = confusion_matrix(y_true[('y2_true')], y_pred[('y2_pred')])
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
heat2 = plt.subplot(223)
sn.heatmap(df_cm, annot=True, ax=heat2)   
heat2.set_xlabel('Predicted Labels')
heat2.set_ylabel('True Labels')
heat2.set_title('Fold 3')
report2 = classification_report(y_true['y2_true'], y_pred[('y2_pred')],target_names=classes)
print(report2)

cf_matrix = confusion_matrix(y_true[('y3_true')], y_pred[('y3_pred')])
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
heat3 = plt.subplot(224)
sn.heatmap(df_cm, annot=True, ax=heat3)   
heat3.set_xlabel('Predicted Labels')
heat3.set_ylabel('True Labels')
heat3.set_title('Fold 4')
report3 = classification_report(y_true['y3_true'], y_pred[('y3_pred')],target_names=classes)
print(report3)


lol.tight_layout()
#report = classification_report(y_true,y_pred,target_names=classes)
#print(report) 
    

'''
#plotting for folds
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_xlabel('Fold')
ax1.set_title('Loss at each Fold')
ax1.grid(True)
ax1.plot(history['train_loss'],'kx')
ax1.plot(history['test_loss'], 'rx')
ax1.set_ylabel('Loss')
ax1.legend(['training data','dev. data'])
ax1.set_xlim(0,k)
ax2.set_title('Accuracy at each fold')
ax2.set_xlabel('Fold')
ax2.grid(True)
ax2.plot(history['train_acc'],'kx')
ax2.plot(history['test_acc'],'rx')
ax2.legend(['training data','dev. data'])
ax2.set_ylabel('Accuracy (%)')
ax2.set_xlim(0,k)
ax2.set_ylim(0,100)
fig.show
'''


'''
#plotting for epochs
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_xlabel('Epoch')
ax1.set_title('Loss at each Fold')
ax1.grid(True)
ax1.plot(np.log(history['train_loss']),'kx')
ax1.plot(np.log(history['test_loss']), 'rx')
ax1.set_ylabel('Log Loss')
ax1.legend(['training data','dev. data'])
ax1.set_xlim(0,num_epochs*k)
ax2.set_title('Accuracy at each fold')
ax2.set_xlabel('Epoch')
ax2.grid(True)
ax2.plot(history['train_acc'],'kx')
ax2.plot(history['test_acc'],'rx')
ax2.legend(['training data','dev. data'])
ax2.set_ylabel('Accuracy (%)')
ax2.set_xlim(0,num_epochs*k)
ax2.set_ylim(0,100)
fig.show
'''

# Test the model
'''
y_pred = []
y_true = []
with torch.no_grad():  
    for frames_test, labels_test in test_loader:
        frames_test = frames_test.to(device)
        labels_test = labels_test.to(device)
        output_test = model(frames_test)
        output_test = (torch.max(torch.exp(output_test), 1)[1]).data.cpu()#.np()
        y_pred.extend(output_test)
        
        labels_test = labels_test.data.cpu()
        y_true.extend(labels_test)

classes = ("handgun","laying","running","sitting","standing","walking","waving")
    
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
heat = plt.subplot()
sn.heatmap(df_cm, annot=True, ax=heat)   
heat.set_xlabel('Predicted labels')
heat.set_ylabel('True labels')
heat.set_title('Confusion Matrix')
plt.savefig('output.png') 


report = classification_report(y_true,y_pred,target_names=classes)
print(report)


FILE = "model_31.pth"
torch.save(model.state_dict(), FILE)
'''