import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def plot_metrics():

	history = np.load('../g_model_history.npy',allow_pickle='TRUE').item()
	g_train_loss = history['loss']
	g_train_acc = history['accuracy']
	g_val_loss = history['val_loss']
	g_val_acc = history['val_accuracy']


	fig1, ax1 = plt.subplots()
	ax1.plot(g_train_loss, '-b', label='loss')
	ax1.plot(g_val_loss,  '-r', label='val_loss')
	ax1.set_title("Loss")
	ax1.legend(loc='upper left')
	ax1.set_xlabel("Epochs")
	fig1.savefig('../pic/g_loss.png')

	fig2, ax2 = plt.subplots()
	ax2.plot(g_train_acc, '-b', label='acc')
	ax2.plot(g_val_acc,  '-r', label='val_acc')
	ax2.legend(loc='upper left')
	ax2.set_title("Accuracy")
	ax2.set_xlabel("Epochs")
	fig2.savefig('../pic/g_acc.png')


	history2 = np.load('../d_model_history.npy',allow_pickle='TRUE').item()
	d_train_loss = history2['loss']
	d_train_acc = history2['accuracy']
	d_val_loss = history2['val_loss']
	d_val_acc = history2['val_accuracy']

	fig3, ax3 = plt.subplots()
	ax3.plot(d_train_loss, '-b', label='loss')
	ax3.plot(d_val_loss,  '-r', label='val_loss')
	ax3.set_title("Loss")
	ax3.legend(loc='upper left')
	ax3.set_xlabel("Epochs")
	fig3.savefig('../pic/d_loss.png')

	fig4, ax4 = plt.subplots()
	ax4.plot(d_train_acc, '-b', label='acc')
	ax4.plot(d_val_acc,  '-r', label='val_acc')
	ax4.legend(loc='upper left')
	ax4.set_title("Accuracy")
	ax4.set_xlabel("Epochs")
	fig4.savefig('../pic/d_acc.png')

	plt.show()

def confusion_metrix():

	y_true_c = np.load('/home/miranda/Documents/code/3DCNN/g_model_true.npy')
	y_true = []
	for x in y_true_c:
		max_index_col = np.argmax(x, axis=0)
		y_true.append(max_index_col)
	y_true = np.asarray(y_true)
	print(y_true)
	y_pred_c = np.load('/home/miranda/Documents/code/3DCNN/g_model_pred.npy')
	y_pred = []
	for x in y_pred_c:
		max_index_col = np.argmax(x, axis=0)
		y_pred.append(max_index_col)

	y_pred = np.asarray(y_pred)
	print(y_pred)
	# Print the confusion matrix
	print(metrics.confusion_matrix(y_true, y_pred))

	# Print the precision and recall, among other metrics
	print(metrics.classification_report(y_true, y_pred, digits=3))

	fpr, tpr, _ = roc_curve(y_true,  y_pred)
	auc = roc_auc_score(y_true, y_pred)

	fig = plt.figure(figsize=(8,6))

	plt.plot(fpr, tpr, 
			 label="{}, AUC={:.3f}".format('3dcnn', auc))

	plt.plot([0,1], [0,1], color='orange', linestyle='--')

	plt.xticks(np.arange(0.0, 1.1, step=0.1))
	plt.xlabel("Flase Positive Rate", fontsize=15)

	plt.yticks(np.arange(0.0, 1.1, step=0.1))
	plt.ylabel("True Positive Rate", fontsize=15)

	plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
	plt.legend(prop={'size':13}, loc='lower right')

	# plt.show()
	fig.savefig('/home/miranda/Documents/code/3DCNN/pic/3dcnn_roc_curve.png')

# plot_metrics()
confusion_metrix()