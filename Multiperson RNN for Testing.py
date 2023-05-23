import numpy as np
import torch
import torch.nn as nn


def atleast_4d(x):
    if x.ndim < 4:
        y = np.expand_dims(np.atleast_3d(x), axis=3)
    else:
        y = x
    return y

def normalise(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return np.around(v/norm,3)

def augament_data(reconstruction):
    
    video_leng = 240
    
    if reconstruction.shape[1] > video_leng:
        for count in range(reconstruction.shape[1]-video_leng):
            reconstruction = np.delete(reconstruction, -1, axis=1) #delete last elements from array until all equal length, needs replacing with keyframes
        
    #delete neck, shoulders and hips
    shortend_reconstruction = np.array([])
    short_out = np.array([])
    for j in range(reconstruction.shape[0]):
        for i in range(reconstruction.shape[1]):
            recon = reconstruction[j][i]        
            short = np.delete(recon, [1,4,9,11,14], 0)
            short = np.subtract(short,short[5,:]) #translate all data about the spine joint, to achieve [0,0,0] for the spine           
            if i == 0:
                short_out = np.atleast_3d(short)
            else:
                short_out = np.append(short_out, np.atleast_3d(short), axis = 2)
        if j == 0:
            shortend_reconstruction = atleast_4d(short_out)
        else:
            shortend_reconstruction = np.append(shortend_reconstruction, atleast_4d(short_out), axis = 3)
    shortend_reconstruction = shortend_reconstruction.transpose(3, 2, 0, 1)
    
    #print(f'shortened reconstruction: {len(shortend_reconstruction.shape)}')
    
    if len(shortend_reconstruction.shape) == 5:
        shortend_reconstruction = shortend_reconstruction.reshape([shortend_reconstruction.shape[0]*shortend_reconstruction.shape[1],shortend_reconstruction.shape[2],36])
    elif len(shortend_reconstruction.shape) == 4:
        shortend_reconstruction = shortend_reconstruction.reshape([shortend_reconstruction.shape[0]*shortend_reconstruction.shape[1],36])
    
    return shortend_reconstruction

def my_RNN_multiperson(z, num_person):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FILE = "./model/this_is_fold1_report.pth"
    input_size = 36
    hidden_size = 31
    num_layers = 2
    num_classes = 7
    classification_threshold = 0.6
    
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(RNN, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size*2, num_classes)
            
        def forward(self, x):
            # Set initial hidden states (and cell states for LSTM)
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
            # Forward propagate RNN
            out, _ = self.gru(x, h0)  
            out = out[:, -1, :]      
            out = self.fc(out)
            return out
    
    loaded_model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.eval()
    
    action_list = ["handgun","laying","running","sitting","standing","walking","waving"]  
    
    
    #call data
    data = augament_data(z)
    #print(data.shape)
    #print(num_person)
    
    #loop for number of people in video
    for i in range(num_person):
        data1 = np.atleast_3d(data[240*i:240*(i+1)])
        data1 = data1.transpose(2,0,1)
        #print(f'\ndata1 shape: {data1.shape}')
        frames = torch.from_numpy(data1)
        frames = frames.to(device)
        outputs = loaded_model(frames)
        outputs = torch.softmax(outputs, dim=1)
        #print(f'\n{np.around(outputs.cpu().detach().numpy(), 3)}')
        # max returns (value ,index)
        max_value, predicted = torch.max(outputs.data, 1)
        outputs = np.around(outputs.cpu().detach().numpy(), 3)
        if max_value >= classification_threshold:
            print(f'\nPerson {i+1} is {action_list[predicted.item()]}')
            print(outputs)
            if i == 0:
                actions_occuring = [action_list[predicted.item()]]
            else:
                actions_occuring.append(action_list[predicted.item()])
        else:
            print(f'\nNo prediction met the required threshold for person {i+1}')
            if i ==0:
                actions_occuring = ["Unknown Action"]
            else:
                actions_occuring.append("Unknown Action")
        if i == 0:
            scene_output = np.zeros(outputs.shape[1], dtype=np.float32)
            action_outputs = np.atleast_2d(outputs)
        else:
            action_outputs = np.append(action_outputs, np.atleast_2d(outputs), axis = 0)
            scene_output = scene_output + outputs
        
    
    return normalise(scene_output), action_outputs, actions_occuring
    print('\n\n')
    
    