import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
import time

class Training():
	def __init__(self, model, device, X, y, X_val=None, y_val=None,
		loss_function=None, optimizer=None):

		# Dataset as numpy array
		self.setTrain(X,y)
		if X_val is not None:
			self.setVal(X_val, y_val)
		else:
			self.X_val = None
			self.y_val = None

		# Model stuff
		self.model = model
		self.device = device
		self.history = {
			'train' : [],
			'val' : []
		}

		self.loss_function = loss_function
		if self.loss_function is None:
			self.loss_function = nn.L1Loss()

		self.optimizer = optimizer 		# improve -> pass optim to class and initialize inside
		if self.optimizer is None:
			self.optimizer = optim.Adamax(model.parameters())

		# Load model on GPU
		self.model.to(self.device)

	# Returns batch as pytorch tensor on device.
	def getBatch(self, offset, batch_size, val=False):
		if val is True:
			X = self.X_val
			y = self.y_val
		else:
			X = self.X
			y = self.y

		input = torch.autograd.Variable(
			torch.tensor( X[ offset:offset + batch_size ], dtype=torch.float )
		)
		target = torch.autograd.Variable(
			torch.tensor( y[ offset:offset + batch_size ], dtype=torch.float )
		)
		return input.to(self.device), target.to(self.device)

	def fit(self, batch_size, n_epochs, val=False):
		
		#Print all of the hyperparameters of the training iteration:
		print("====== HYPERPARAMETERS ======")
		print("batch_size :", batch_size)
		print("epochs :", n_epochs)
		print("loss function :", self.loss_function)
		print("optimizer :", self.optimizer)
		print("device :",self.device)
		print("=" * 29)
		
		n_batch = self.X.shape[0] // batch_size
		
		start_T = int(time.time())
		
		for epoch in range(1,n_epochs+1):
			print("===> Epoch[{}]".format(epoch), end='', flush=True)
			epoch_T = time.time()
			epoch_loss = 0
			
			for it in range(n_batch):
				input, target = self.getBatch(it*batch_size, batch_size)
				self.optimizer.zero_grad()
				
				output = self.model(input)

				O = torch.cat((output,output,output),1).to(self.device)
				T = torch.cat((target,target,target),1).to(self.device)

				loss = self.loss_function(O, T)
				loss.backward()
				self.optimizer.step()
				
				loss_train = loss.item()
				epoch_loss += loss_train
				
				tick_T = time.time()
				print("\r", end='')
				print("===> Epoch[{}]({}/{}): Loss: {:.4f}\tETA {}\tEpoch Loss: {:.4f}"
					  .format(epoch, it + 1, n_batch, loss_train,
					  self.formatTime((tick_T - epoch_T) / (it + 1) * (n_batch - it + 1)),
					  epoch_loss / (it+1)), end='', flush=True)
				
			epoch_loss /= n_batch
			self.history['train'].append(epoch_loss)
			torch.save(self.model.state_dict(), "weights"+str(epoch))
			print("\nEpoch[{}] finished in {} with loss {}".format(epoch, self.formatTime(tick_T - epoch_T), epoch_loss))
			
			if val is True:
				self.validate(batch_size)
			
			print("\n----------------------------\n")
		print("Finished training of {} epochs in {}.".format(n_epochs, self.formatTime(int(time.time())-start_T)))
			
		return self.history

	def validate(self, batch_size):
		if self.X_val is None:
			print("Cannot validate, no validation dataset given.")
			return None

		loss_val = 0
		n_batch_val = self.X_val.shape[0] // batch_size

		print("Validating on {} samples.".format(n_batch_val * batch_size))

		start_T = int(time.time())
		for it in range(n_batch_val):
			input, target = self.getBatch(it*batch_size, batch_size, val=True)

			output = self.model(input)
			loss = self.loss_function(output, target)
			loss_val += loss.item()

			tick_T = time.time()
			print("\r", end='')
			print("===> Validating ({}/{}):\tETA {}\tValidation Loss: {:.4f}"
					  .format(it + 1, n_batch_val,
					  self.formatTime((tick_T - start_T) / (it + 1) * (n_batch_val - it + 1)),
					  loss_val / (it+1)), end='', flush=True)

		print("\nValidation loss = {:.4f}".format(loss_val / n_batch_val))
		self.history['val'].append( loss_val / n_batch_val )

		return loss_val / n_batch_val

	def setTrain(self, X, y):
		assert type(X) == type(y) == np.ndarray
		assert X.shape[0] == y.shape[0]
		assert X.shape[2:4] == y.shape[2:4]

		self.X = X
		self.y = y

	def setVal(self, X_val, y_val):
		assert type(X_val) == type(y_val) == np.ndarray
		assert X_val.shape[0] == y_val.shape[0]
		assert X_val.shape[2:4] == y_val.shape[2:4]

		self.X_val = X_val
		self.y_val = y_val

	# Takes t as number of seconds, returns formatted string as HH:MM:SS
	@staticmethod
	def formatTime(t):
		t = int(t)
		s = t % 60
		m = (t // 60) % 60
		h = t // 3600
		return str(h) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2)