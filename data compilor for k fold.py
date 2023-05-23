import numpy as np
import os
import random
import math as m

test_size = 20

def atleast_4d(x):
    if x.ndim < 4:
        y = np.expand_dims(np.atleast_3d(x), axis=3)
    else:
        y = x
    return y

def atleast_5d(x):
    if x.ndim < 5:
        y = np.expand_dims(np.atleast_3d(x), axis=4)
    else:
        y = x
    return y

def rotation(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ], [ m.sin(theta), m.cos(theta) , 0 ], [ 0, 0, 1 ]])

def delete_joint(reconstruction):  
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

    return(shortend_reconstruction)


def augament_data(shortend_reconstruction, no_rotation):
    
    #rotate about z
    random_rotation = np.array([])
    rotated_out = np.array([])
    rotated_reconstruction = np.array([])
    angle_list = []
    for k in range(no_rotation):
        angle_list.append(m.pi*2*random.random())
    
    for j in range(shortend_reconstruction.shape[0]):
        for a in range (no_rotation):
            angle = angle_list[a]
            for i in range(shortend_reconstruction.shape[1]-1):
                recon = shortend_reconstruction[j][i]            
                rotated = np.array(recon* rotation(angle))
            
                if i == 0:
                    rotated_out = np.atleast_3d(rotated)
                else:
                    rotated_out = np.append(rotated_out, np.atleast_3d(rotated), axis = 2)
            rotated_out = np.append(rotated_out, np.atleast_3d(shortend_reconstruction[j,-1]), axis = 2)   
            if a == 0:
                rotated_reconstruction = atleast_4d(rotated_out)
            else:
                rotated_reconstruction = np.append(rotated_reconstruction, atleast_4d(rotated_out), axis = 3)
            
        if j == 0:
            random_rotation = atleast_4d(rotated_reconstruction)
        else:
            random_rotation = np.append(random_rotation, atleast_4d(rotated_reconstruction), axis = 3)
        
    random_rotation = random_rotation.transpose(3, 2, 0, 1)
    
    return random_rotation


def loader(file, folder):
    x = np.load("./KFold_data/"+folder+"/"+file)
    y = x['reconstruction']
    leng = 240
    if y.shape[1] > leng:
        for i in range(y.shape[1]-leng):
            y = np.delete(y, -1, axis=1)    
    short = delete_joint(y)
    return short


#locate folders for loading
actions_available = []
for folders in os.listdir("./KFold_data"):
    if folders.endswith("_action"):
        actions_available.append(folders)

leng = 240
training_data = []
z = np.atleast_3d(np.array([]))

#open each folder consecutivly
for j in range(len(actions_available)):
    data_available = []
    #check folders for files than can be opened and add names to data_available
    for files in os.listdir("./KFold_data/"+actions_available[j]):
        if files.endswith('.npz'):                
            data_available.append(files)
        else:
            continue
    # itterate through each file in the folder and load into array
    for i in range(len(data_available)):
        print("Loading "+data_available[i])
        #load the file
        load = loader(data_available[i], actions_available[j])
        #label the file
        load = np.append(load, np.full((1,1,12,3), j), axis = 1)
        print(load.shape)
        
        
        #add file to list of already opened files
        if i == 0 and j == 0:
            z = np.atleast_3d(load)
        else:
            z = np.append(z, np.atleast_3d(load), axis = 0)

np.random.shuffle(z)

#training_data = np.atleast_3d([])
no_rotation = 100
training_data = augament_data(z, no_rotation)

training_data = np.atleast_3d(training_data.reshape([training_data.shape[0],training_data.shape[1],36]))
    #short = short.transpose(2,0,1)

    
print(training_data.shape)
        
print(f'\nTraining data shape is: {training_data.shape}')

np.savez("./KFold_data/KFoldtraining_data", training_data=training_data)