import os,sys
import numpy as np
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CosDataset, train_tfm, test_tfm
from model import CNN_cosmo


print('torch-version: ',             torch.__version__)
print('torch.cuda.is_available: ',   torch.cuda.is_available())
print('torch.cuda.device_count(): ', torch.cuda.device_count())
print('device_name: ',               torch.cuda.get_device_name(0))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# configs
## paths
base  = './'
model_name = "CNN_cosmo"
model_param_file = 'model_param.pkl'
## model
mode = "cosmo"
## dataset
rng_seed = 114514
batch_size = 32
training_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1
## training
n_epoch = 1000
lr = 0.002

def is_save_epoch(epoch):
    return epoch % 10 == 0

def is_report_step(step):
    return step % 10 == 0

def save_net_params(net, path):
    torch.save(net.state_dict(), path)
    print('Net_params has been saved in '+path+' .')
    return

if __name__ == "__main__":
    print('>>> 1. load net')
    net = CNN_cosmo(mode)

    if torch.cuda.is_available():
        net.cuda()
    
    if os.path.exists(base+model_param_file):
        print('\t load the existed net_params to net')
        print('\t netparam_file: \n\t\t', os.path.join(base, model_param_file))
        net.load_state_dict(torch.load(os.path.join(base, model_param_file)))
    else:
        print('\t No net_params exists, use the initial net')
    
    print('>>> 2. load data_set')
    
    img_list = np.load("/home/chenze/data_gpfs02/CAMELS_multifield/dataset/compiled_img_TNG.npy")
    img_list = torch.FloatTensor(img_list)
    lab_list = np.load("/home/chenze/data_gpfs02/CAMELS_multifield/dataset/compiled_params_TNG.npy")
    lab_list = torch.FloatTensor(lab_list)
    
    np.random.seed(rng_seed)
    shuffle = np.arange(img_list.shape[0])
    np.random.shuffle(shuffle)
    
    img_list = img_list[shuffle]
    lab_list = lab_list[shuffle]
    
    len_training = int(len(img_list) * training_ratio)
    len_valid = int(len(img_list) * training_ratio) + int(len(img_list) * valid_ratio)
    
    train_set = CosDataset(img_list[:len_training], lab_list[:len_training], tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_set = CosDataset(img_list[len_training:len_valid], lab_list[len_training:len_valid], tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_set  = CosDataset(img_list[len_valid:], lab_list[len_valid:], tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    
    print('>>> 3. choose optimizer and loss-function')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
                            
    print('>>> 4. train net')
    loss_mean = [] # [epoch, train_lossmean, test_lossmean]
    for epoch in range(n_epoch):
        print('\t '+'='*20+' Epoch=', epoch, '='*20)
        
        net = net.train()
    
        train_loss = []
        for step, (batch_data, batch_targ) in enumerate(train_loader):
            time_ = time.strftime('%H%M%S', time.localtime(time.time()))
            
            batch_data = batch_data.type(torch.FloatTensor).to(device)
            batch_targ = batch_targ.type(torch.FloatTensor).to(device)
            
            out  = net(batch_data)
            loss = loss_func(out, batch_targ)
            
            if is_report_step(step):
                print('\t step-'+str(step)+': '+time_+'\n\t loss={:.4f}'.format(loss.data))
                
            if device != 'cpu':
                train_loss.append(loss.cpu().data)
            else:
                train_loss.append(loss.data)
            
            ''' >>>  clean the old gradients  <<< '''
            optimizer.zero_grad()
            ''' >>>     back -propagation     <<< '''
            loss.backward()
            ''' >>> take new gradients effect <<< '''
            optimizer.step()
            
            
        if not is_save_epoch(epoch):
            continue
            
        net = net.eval()
    
        valid_loss = []
        for step, (batch_data, batch_targ) in enumerate(valid_loader):
            batch_data = batch_data.type(torch.FloatTensor).to(device)
            batch_targ = batch_targ.type(torch.FloatTensor).to(device)
    
            out  = net(batch_data)
            loss = loss_func(out, batch_targ)
            if device != 'cpu':
                valid_loss.append(loss.cpu().data)
            else:
                valid_loss.append(loss.data)
    
        valid_loss_mean  = np.mean(valid_loss)
        train_loss_mean = np.mean(train_loss)
        loss_mean.append([epoch, train_loss_mean, valid_loss_mean])
    
        print('epoch={:>03d}'.format(epoch)+':  save net params')
        net_savename = f'{model_name}_epoch-{epoch:>03d}'+'_model_params.pkl'
        save_net_params(net, os.path.join(base,net_savename))
    
    print('>>> 5. save the trained net and loss-evolution')
    loss_mean = np.array(loss_mean)
    np.savetxt(os.path.join(base,f'{model_name}_epoch-{epoch:>03d}'+'_TrainLossMean_TestLossMean.txt'), loss_mean)
    print('\t Training finished !!!')


    