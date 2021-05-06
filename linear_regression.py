import dataset
import tensorflow as tf
import matplotlib.pyplot as plt
from argparse import ArgumentParser


class linear_estimator:
	def __init__(self,trainable_weights,trainable_bias):
		self.trainable_weights=tf.Variable(trainable_weights,dtype=tf.float32)
		self.trainable_bias=tf.Variable(trainable_bias,dtype=tf.float32)
		self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.2)

	def predict(self,input_data):
		return tf.linalg.matmul(input_data,self.trainable_weights)+self.trainable_bias

	@tf.function	
	def update_params(self,input,output):
		loss=lambda:tf.reduce_mean((self.predict(input)-output)**2)
		self.optimizer.minimize(loss,[self.trainable_weights,self.trainable_bias])

	def train(self,epochs,train_data,train_label):
		
		for ep in range(epochs):
			input=train_data
			output=train_label
			self.update_params(input,output)
			if ep%10==0:
				print("\rloss {}".format(tf.reduce_mean((self.predict(input)-output)**2)),end='')
		print()		
	def  __plot(self,predictions,test_data,test_label):
		fig=plt.figure()
		ax=fig.add_subplot(111,projection='3d')
		ax.set_xlabel('param1')
		ax.set_ylabel('param2')
		ax.set_zlabel('targets')
		ax.scatter(test_data[:,:1],test_data[:,1:],test_label,c='b',label='ground truth')
		ax.legend()
		ax.scatter(test_data[:,:1],test_data[:,1:],predictions,c='r',label='predictions')
		ax.legend()
		plt.title('Linear regression')
		plt.show()

	def test(self,test_data,test_label):
		predictions=self.predict(test_data)
		self.__plot(predictions,test_data,test_label)

def init__model_params():
	trainable_weights=tf.ones(shape=[2,1])
	trainable_bias=1
	return (trainable_weights,trainable_bias)

if __name__=='__main__':
	epochs=3000
	(train_data,train_label),(test_data,test_label)=dataset.load_data('lg_dataset')
	(trainable_weights,trainable_bias)=init__model_params()
	estimator=linear_estimator(trainable_weights,trainable_bias)
	estimator.train(epochs,train_data,train_label)
	estimator.test(test_data,test_label)

