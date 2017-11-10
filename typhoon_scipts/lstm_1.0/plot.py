import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import config
hist_path = config.hist_path
loss_image_path = config.loss_image_path
acc_image_path = config.acc_image_path
with open(hist_path,'r') as f:
	hist = json.load(f)
train_acc = hist['acc']
train_loss = hist['loss']
val_acc = hist['val_acc']
val_loss = hist['val_loss']
fig = plt.figure()
plt.title('train and validation loss')
plt.plot(train_loss,'g^',label = 'train')
plt.plot(val_loss,'r--', label = 'validation')
plt.legend(loc = 'upper left', shadow =True)
plt.savefig(loss_image_path)
plt.close(fig)
plt.title('train and validation acc')
plt.plot(train_acc,'g^', label = 'train')
plt.plot(val_acc,'r--', label = 'validation')
plt.legend(loc = 'upper left', shadow  =  True)
plt.savefig (acc_image_path)