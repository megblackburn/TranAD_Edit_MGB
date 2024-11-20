import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

'''
def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size: 
            w = data[i-w_size:i]
        else:
            print(np.shape(data[0].repeat(w_size-i, 1)))
            print(np.shape(data[0:i]))
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
        
    return torch.stack(windows)
'''

def load_dataset(dataset, oneFileFlag): # dataset is the name of the saved dataset, oneFileFlag=True means one file will be loaded 
    folder = os.path.join(output_folder, dataset)
    if not os.path.join(folder):
        raise Exception('Processed Data not found')
    loader = []
    if oneFileFlag == True:
        for file in ['train', 'test', 'labels']:
            if dataset == 'GAUSS': 
                file = '1_GAUSS_Anomaly_Gaussnoise_' + file
                print(file)
       # if dataset == 'GAUSS':
            #train = '1_GAUSS_Anomaly_Gaussnoise_train.npy'
           # labels = '1_GAUSS_Anomaly_Gaussnoise_labels.npy'
          #  test = '1_GAUSS_Anomaly_Gaussnoise_test.npy'
            
            loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    else:
        filelist = os.listdir(folder)
        print(np.arange(1, 10, 1))
        
        for i in np.arange(1, 10, 1):
            
            if dataset == 'GAUSS': # this needs to seperate into labels, train and test and save seperately in array like before
                for file in ['train', 'test', 'labels']:
                    print(i)
                    file = f'{i}_GAUSS_Anomaly_Gaussnoise_'+file
                file = np.load(os.path.join(folder, f'{file}.npy'))
                loader.append(file)
        
    if args.less:
        loader[0] = cut_array(0.2, loader[0])
    test_loader = DataLoader(loader[1], batch_size = loader[1]].shape[0])
    train_loader = DataLoader(loader[0], batch_size = loader[0].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels

def save_model(model, optimizer, schedule, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/nodel.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
    

def load_model(modelname, dims):
    import src.models 
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, data0, optimizer, scheduler, training = True):
    l = nn.MSELoss(reduction = 'mean' if training else 'none')
    feats = data0.shape[1]
    if 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size = bs)
        n = epoch +1;
        w_size = model.n_window
        l1s = []
        l2s = []
        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                if not isinstance(z, tuple):
                    l1 = l(z, elem)
                else:
                    l1 = (1/n) * l(z[0], elem) + (1-1/n) * l(z[1], elem)
                
                if isinstance(z, tuple):
                    z = z[1]
                
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple):
                    z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
        
        
if __name__ == '__main__':
    train_loader, test_loader, labels = load_dataset(args.dataset, False)
   #plt.plot(np.arange(0,len(test_loader)), test_loader)
   # print('shape of labels = ', np.shape(labels))
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])
   
   ## Prepare Data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
   
    if 'TranAD' in model.name:
       trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    ## Training Phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = 15; e = epoch + 1; start = time()
        
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
            
        print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
        
        save_model(model, optimizer, scheduler, e, accuracy_list)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ## Testing Phase
    torch.zero_grad = True
    model.eval()
    
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    
    ## Plot Curves
    if not args.test:
        if 'TranAD' in model.name:
            testO = torch.roll(testO, 1, 0)
        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
        
    df = pd.DataFrame()
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    
    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        
        df = df._append(result, ignore_index=True)
        
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    pprint(result)