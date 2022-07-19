import pandas as pd 
import numpy as np
import torch
from torch import tensor
import json
from BinaryNet.model_utils import BinarizeLinearLayer, BinarizeConv2d, Adam_meta, Adam_bk, SignActivation
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
from datetime import datetime
from torchmetrics import F1Score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from dataset import AudioUtil


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_dataset(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    # extract inputs and tagets

    X = torch.from_numpy(np.array(data['spectrogram']))
    y = data['birds']

    return X, y

def train(model, train_dl, lr=5e-03, epochs=50, optim='', meta=0.0):
 
    criterion = nn.CrossEntropyLoss()
    if optim == 'meta':
        optimizer = Adam_bk(model.parameters(), lr=lr, meta=meta, weight_decay=1e-07)
    else :
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-07)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr= 0.005,
                                                   steps_per_epoch=int(len(train_dl)),
                                                   epochs=epochs,
                                                   anneal_strategy='linear')
    loss_vec = []
    acc_vec = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        start = datetime.now()
        model.train()
        for i, data in enumerate(train_dl):

            inputs, labels = data[0].float().to(DEVICE), data[1].long().to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            
            for p in model.parameters():  # updating the org attribute
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
                    
            optimizer.step()

            for p in model.parameters():  # updating the org attribute
                if hasattr(p,'org'):
                    p.org.copy_(p.data)
            
            scheduler.step()
            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            if i % 100 == 0 and i !=0:  
                print(f'[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                print(f'{datetime.now()-start}')
                start = datetime.now()

        num_batches = len(train_dl)
        avg_loss = running_loss/num_batches
        acc = correct_prediction/total_prediction
        acc_vec.append(acc)   
        loss_vec.append(avg_loss)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    return loss_vec, acc_vec
    
def get_data_split(data_path):
    X, y = load_dataset(data_path)
    labels= pd.unique(y).tolist()
    l = []
    for birds in labels:
        for bird in birds.split(' '):
            l.append(bird)
    labels = pd.unique(l).tolist()
        
    # create train/validation/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3)

    X_train = torch.swapaxes(X_train, 1, -1)
    X_validation = torch.swapaxes(X_validation, 1 ,-1)
    X_test = torch.swapaxes(X_test, 1, -1)

    return X_train, X_validation, X_test, y_train, y_validation, y_test, labels

def normalize(tensor):
  tensor_norm = tensor - tensor.mean()
  return tensor_norm / tensor_norm.abs().max()


class SoundDS(Dataset):
    def __init__(self, x, y, mappings, is_train=True):
        self.x = x.copy()
        self.y = y.copy()
        if is_train:
          augmented_data = []
          augmented_data_labels = []
          for data, label in zip(x, y):
            data = normalize(data)
            if label != 'nocall': 
              for _ in range(35):
                  augmented_data += torch.unsqueeze((data + (torch.randn(data.shape[1], data.shape[2])-torch.mean(data))), 0)
                  augmented_data_labels.append(label)
                  
          self.x += augmented_data
          self.y += augmented_data_labels

        self.mappings = mappings
        self.num_classes = len(mappings.keys())

    def __getitem__(self, idx):
        target = self.mappings[self.y[idx].split()[0]]
        return self.x[idx] , target

    def __len__(self):
        return len(self.x)

def inference(model, val_dl):
    correct_pred = 0
    total_pred = 0
    preds = np.empty(shape=(1,))
    targets = np.empty(shape=(1,))

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].float().to(DEVICE), data[1].long()
            outputs = model(inputs).to('cpu')
            
            _, prediction = torch.max(outputs,1)
            correct_pred += (prediction == labels).sum().item()
            
            total_pred += prediction.shape[0]
            
            preds = np.append(preds, prediction)
            targets = np.append(targets, labels)
    preds = preds[1:,...]
    targets = targets[1:,...]
    acc = correct_pred / total_pred
    print(f'Accuracy: {acc:.2f}, Total items: {total_pred}')
    return preds, targets




if __name__ == '__main__':
    tasks = {
        '1': [f'data_{i+1}.0.json' for i in range(15)], 
        '2': [f'data_{i+21}.0.json' for i in range(15)]
    }

    PATH = '/datadrive/datasets/birdclef/soundscape/'
    data = {}
    labels = []
    for task in tasks.keys():
        data[task] = {
            'X_train': [],
            'y_train': [],
            'X_test': [],
            'y_test': [],
            'X_val': [],
            'y_val': []
        }
        for path in tasks[task]:
            print(path)
            X_train, X_val, X_test, y_train, y_val, y_test, l = get_data_split(PATH + path)

            data[task]['X_train']+=X_train
            data[task]['y_train']+=y_train
            data[task]['X_test']+=X_test
            data[task]['y_test']+=y_test
            data[task]['X_val']+=X_val
            data[task]['y_val']+=y_val

            labels.append(l)

    labels = [ birds.split()[0] for subset in labels for birds in subset]
    labels = pd.unique(labels).tolist() 
    print('Labels extracted : ', labels)
    
    mappings = {}
    n_labels = 0
    
    for i, label in enumerate(labels):
        mappings[label] = i
        n_labels =1 + i

    model = AudioClassifier(n_labels=n_labels)
    modelbnn = AudioClassifierBNN(n_labels=n_labels)

    for task in data.keys():
        print(f'Training Task {task}:')
        print('Training size: ', len(data[task]['X_train']))
        train_ds = SoundDS(data[task]['X_train'], data[task]['y_train'], mappings)
        val_ds = SoundDS(data[task]['X_val'], data[task]['y_val'], mappings)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

        train(model, train_dl, epochs=10)
        torch.save(model.state_dict(), f'model_{task}.h5')
        f1_score = F1Score(num_classes=n_labels, average='macro', multiclass=True)
        
        with torch.no_grad():
            pred, targets = inference(model, train_dl)
            score = f1_score(torch.from_numpy(pred).long(), torch.from_numpy(targets).long())
            print(f'F1_score: {score}')

        f1_score = F1Score(num_classes=n_labels, average='macro', multiclass=True)
        train(modelbnn, train_dl, epochs=10, optim='meta')
        torch.save(model.state_dict(), f'model_bnn_{task}.h5')
        with torch.no_grad():
            pred, targets = inference(modelbnn, train_dl)
            score = f1_score(torch.from_numpy(pred).long(), torch.from_numpy(targets).long())
            print(f'F1_score: {score}')

    model.load_state_dict(torch.load('model_2.h5'))
    modelbnn.load_state_dict(torch.load('model_bnn_2.h5'))
    f1_score = F1Score(num_classes=n_labels, average='macro', multiclass=True)
    print('Computing scores for both datasets...')
    for task in data.keys():
        print(f'Task {task}:')
        test_ds = SoundDS(data[task]['X_val'], data[task]['y_val'], mappings)
        test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

        with torch.no_grad():
            print('Model: ')
            pred, targets = inference(model, test_dl)
            score = f1_score(torch.from_numpy(pred).long(), torch.from_numpy(targets).long())
            print(f'F1_score: {score}')

        with torch.no_grad():
            print('BNN: ')
            pred, targets = inference(modelbnn, test_dl)
            score = f1_score(torch.from_numpy(pred).long(), torch.from_numpy(targets).long())
            print(f'F1_score: {score}')
