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
# from beepy import beep

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	print('Shape of data in Convert to Windows', np.shape(data))
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset, oneFileFlag):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	training_loader = []
	testing_loader = []
	labels_loader = []
 
	if oneFileFlag == True:
		for file in ['train', 'test', 'labels']:
			if dataset == 'GAUSS': file = '1_GAUSS_Anomaly_Gaussnoise_' + file
			loader.append(np.load(os.path.join(folder, f'{file}.npy')))

	else:
		trainnums = np.arange(1,4,1)

		for i in np.arange(1,4,1):

			for file in ['train']:
				if dataset == 'GAUSS':
					file = f'{i}_GAUSS_Anomaly_Gaussnoise_'+file

					file = np.load(os.path.join(folder, f'{file}.npy'))
					training_loader.append(file)
			for file in ['test']:
				if dataset == 'GAUSS':
					file = f'{i}_GAUSS_Anomaly_Gaussnoise_'+file

					file = np.load(os.path.join(folder, f'{file}.npy'))
					testing_loader.append(file)
			for file in ['labels']:
				if dataset == 'GAUSS':
					file = f'{i}_GAUSS_Anomaly_Gaussnoise_'+file

					file = np.load(os.path.join(folder, f'{file}.npy'))
					labels_loader.append(file)
		
	print('shape of loader = ', np.shape(loader))
		#loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(0.2, loader[0])

	training_data = np.concatenate((training_loader[0], training_loader[3], training_loader[6])) #, loader[3], loader[4], loader[5], loader[6], loader[7], loader[8])) #loader[0:trainnums[-1]]
	testing_data =  np.concatenate((testing_loader[1], testing_loader[4], testing_loader[7]))
	labels_data = np.concatenate((labels_loader[2],labels_loader[5], labels_loader[8]))
	
	train_loader = DataLoader(training_data, batch_size=training_data.shape[0]) #:trainnums[-1] # loader2[0]
	test_loader = DataLoader(testing_data, batch_size=testing_data.shape[0])
	labels = labels_data
 
	print('shape of test_loader = ', np.shape(testing_data))
	print('shape of train_loader = ', np.shape(training_data))
	print('shape of labels = ', np.shape(labels_data))

      
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
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
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
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
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]

	if 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
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
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset, False)
	#print('labels = ', np.shape(labels.shape))
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	print('shape of train data before window convert = ', np.shape(trainD), 'shape of test data before window convert = ', np.shape(testD))
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
		print('Shape of training data', np.shape(trainD))
		print('Shape of testing data', np.shape(testD))

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 15; e = epoch + 1; start = time() # num_epochs
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
	#print('line 325')

	### Plot curves
	if not args.test:
		if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
	#print('line 331')
	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	#print('line 335  before for loop')
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		#print(l)
		result, pred = pot_eval(lt, l, ls); preds.append(pred)
	#	df = df.append(result, ignore_index=True)
		df = df._append(result, ignore_index=True)
		break
	# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
	# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
	#print('after for loop line 343')
	#print('lossT = ', lossT)
	#print('loss = ', loss)
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	#print('after loss')
	#lossTfinal, lossFinal = np.mean(lossT), np.mean(loss)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	#print('after labels')
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	#print('after result')
#	print('labels = ', labels)
	result.update(hit_att(loss, labels))
	#print('after result.update')
	result.update(ndcg(loss, labels))
	#print('after result.update2')
	#print(df)
	pprint(result)
	# pprint(getresults2(df, result))
	# beep(4)
