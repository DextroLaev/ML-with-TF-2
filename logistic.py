import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset
import sys

class logistic_regression:
	def __init__(self,trainable_weights,trainable_bias):
		self.trainable_weights = tf.Variable(trainable_weights,dtype=tf.float32)
		self.trainable_bias = tf.Variable(trainable_bias,dtype=tf.float32)
		self.optimizer = tf.keras.optimizers.Adam()

	
	def predict(self,data):
		out = tf.linalg.matmul(data,self.trainable_weights)+self.trainable_bias
		return tf.sigmoid(out)

	@tf.function	
	def update_parameter(self,inputs,label):
		self.loss = lambda : tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict(inputs),labels=label))
		self.optimizer.minimize(self.loss,var_list=[self.trainable_weights,self.trainable_bias])

	def train(self,epochs,data,output,test_data,test_label):
		for i in range(epochs):
				inputs = data
				label = output
				self.update_parameter(inputs,label)
				if i%10 == 0:
					# tf.print('\r loss = {}, testing accuracy = {} %'.format(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict(inputs),labels=label)),self.test(test_data,test_label)),end='')
					tf.print('\r loss = {}'.format(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict(inputs),labels=label))))
					sys.stdout.flush()
		print()		
		self.ROC_CURVE(self.predict(data),label)

	def test(self,data,label):
		predict = self.predict(data)
		count = 0
		for i in range(len(predict)):
			if predict[i] < self.threshold:
				temp = 0
			else:
				temp = 1.0
			if temp == label[i]:
				count += 1 		
		return (count/len(predict))*100

	def confuson_matrix(self,threshold,pred,label):
		true_positive=false_positive=true_negative=false_negative = 0
		for i in range(len(pred)):
			if pred[i] >= threshold and label[i] == 1:
				true_positive += 1
			elif pred[i] >= threshold and label[i] == 0:
				false_positive += 1
			elif pred[i] <= threshold and label[i] == 0:
				true_negative += 1
			else:
				false_negative += 1

		tpr = true_positive/(true_positive+false_negative)
		fpr = false_positive/(false_positive+true_negative)
		return tpr,fpr
			

	def ROC_CURVE(self,prediction,labels):
		threashold_vals = np.linspace(0,1,5)
		tprs = []
		fprs = []
		for i in range(len(threashold_vals)):
			tpr,fpr = self.confuson_matrix(threashold_vals[i],prediction,labels)
			tprs.append(tpr)
			fprs.append(fpr)
		self.threshold = max(fprs)	
		plt.scatter(fprs,tprs,s=100,color='#0F9D58')
		plt.show()			

def init_parameters(data):
	trainable_weights = tf.ones(shape=[data.shape[1],1],dtype=tf.float32)
	trainable_bias = 1
	return (trainable_weights,trainable_bias)

if __name__ == '__main__':
	(train_data,train_label),(test_data,test_label) = dataset.load_data('logistic_regression')
	train_label = tf.reshape(train_label,shape=(-1,1))
	(trainable_weights,trainable_bias) = init_parameters(train_data)
	nn = logistic_regression(trainable_weights,trainable_bias)
	nn.train(1200,train_data,train_label,test_data,test_label)
	print('Testing acc = {} %'.format(nn.test(test_data,test_label)))
	