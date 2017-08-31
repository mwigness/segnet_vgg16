import segnet_train_top_1layer as N1
import segnet_train_top_2layer as N2
import segnet_train_top_3layer as N3
import segnet_train_top_4layer as N4
import segnet_train_top_5layer as N5
import segnet_train_top_6layer as N6
import segnet_train_top_7layer as N7
import segnet_train_top_8layer as N8

import tensorflow as tf
import numpy as np
import os
from image_reader import *
from utils_mod import *
from argparse import ArgumentParser
import fnmatch
from color_map import *
import shutil
import sys
import fnmatch
try:
  import h5py
  import matplotlib.pyplot as plt
  from color_map import *
except:
  pass

def plot_generator():
	accuracies=[]
	for trial in range(1,6):
		path=os.path.join(BASE_DIR,'all_models')
		li=['retrain-'+str(i+1)+'layer-output' for i in range(6)]
		rt=dict()
		for name in li:
			path1=os.path.join(path,name)
			net_num=name.split('-')[1][0]

			for name1 in os.listdir(path1):
				path2=os.path.join(path1,name1,'trial'+str(trial))
				rt[name.split('-')[1],name1.split('-')[0]]=np.load(os.path.join(path2,name.split('-')[1]+'_'+name1.split('-')[0]+'_'+'confmat.npy'))
		for i in rt.keys():
			print i
		k1=['full','half','third','quarter']
		k2=['%dlayer'%(i+1) for i in range(6)]
		accuracies_trial=[]
		for n in k1:
			temp=[]
			for p in k2:
				
				temp.append(np.array(compute_acc(rt[p,n])).reshape(1,-1))
			accuracies_trial.append(np.array(temp))
		#print np.stack(accuracies_trial)
		accuracies.append(np.stack(accuracies_trial))
	print np.stack(accuracies)
	print np.stack(accuracies).shape
	np.save('accuracies.npy',accuracies)



def logfile_parser():
	accuracies=[]
        for trial in range(1,2):
                path=os.path.join(BASE_DIR,'all_models')
                li=['retrain-'+str(i+1)+'layer-output' for i in range(6)]
                rt=dict()
                for name in li:
                        path1=os.path.join(path,name)
                        net_num=name.split('-')[1][0]
			for name1 in os.listdir(path1):
                                path2=os.path.join(path1,name1,'trial'+str(trial))
				plts=file_parser(os.path.join(path2,'logfile_'+name1.split('-')[0]+name.split('-')[1]),name1.split('-')[0])
				print name.split('-')[1],name1.split('-')[0]
				print 'trial',trial,'list stat',plts.shape
                                rt[name.split('-')[1],name1.split('-')[0]]=plts            
                k1=['full','half','third','quarter']
                k2=['%dlayer'%(i+1) for i in range(6)]
                accuracies_trial=[]
                for n in k1:
                        temp=[]
                        for p in k2:
                                temp.append(rt[p,n].reshape([2,-1]))
                        accuracies_trial.append(np.array(temp))
                #print np.stack(accuracies_trial)
                accuracies.append(np.stack(accuracies_trial))
        #print np.stack(accuracies)
        print np.stack(accuracies).shape
        np.save('acc_plots.npy',np.stack(accuracies))

def file_parser(filename,amt):
	f=open(filename,'r')
	train_accs=[]
	val_accs=[]
	content=f.readlines()
	f.close()
	i=0
	while(int(content[i].split(' ')[2][-1])==1):
		i+=1
		#print i
	val=i
	total_len=len(content)
	rl=[val,2*val,3*val,3*val+100]
	while(rl[-1]<total_len):
		for index,i in enumerate(rl):
			if(index==3):
				val_accs.append(float(content[i-1].split(' ')[-1][13:]))
			else:
				train_accs.append(float(content[i-1].split(' ')[-1][13:]))
		rl=[i+3*val+100 for i in rl]
	val_accs=val_accs+[0]*(len(train_accs)-len(val_accs))
	print train_accs
	print val_accs
        return np.array([train_accs,val_accs])

	
def compute_acc(conf_mat):
	d=np.diag(conf_mat)
	total_acc=np.sum(d)/np.sum(conf_mat)
	er=np.zeros(conf_mat.shape)
	for i in range(conf_mat.shape[0]):
		er[i]=conf_mat[i,i]/np.sum(conf_mat[i,:])
	mean_per_class=np.mean(er)
	return [total_acc,mean_per_class]


def call_segnet(net_num,num_classes):
	return {
		'1': N1.Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,desktop=False,weights_path=os.path.join(BASE_DIR,'segnet_road.npy')),
		'2': N2.Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,desktop=False,weights_path=os.path.join(BASE_DIR,'segnet_road.npy')),
		'3': N3.Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,desktop=False,weights_path=os.path.join(BASE_DIR,'segnet_road.npy')),
		'4': N4.Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,desktop=False,weights_path=os.path.join(BASE_DIR,'segnet_road.npy')),
		'5': N5.Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,desktop=False,weights_path=os.path.join(BASE_DIR,'segnet_road.npy')),
		'6': N6.Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,desktop=False,weights_path=os.path.join(BASE_DIR,'segnet_road.npy')),
		'7': N7.Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,desktop=False,weights_path=os.path.join(BASE_DIR,'segnet_road.npy')),
		'8': N8.Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,desktop=False,weights_path=os.path.join(BASE_DIR,'segnet_road.npy'))
		}[net_num]

def test_models(trial=1):

	path=os.path.join(BASE_DIR,'all_models')
	li=['retrain-'+str(i)+'layer-output' for i in [7,8]]	
	for name in li:
		print name
		path1=os.path.join(path,name)
		net_num=name.split('-')[1][0]
		num_classes=8
		batch_size_test=1
		test_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/','trial'+str(trial),'val_set/images/')
		test_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/','trial'+str(trial),'val_set/new_labels/')
		#test_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/b507/images/')
                #test_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/b507/new_labels/')

		reader_test=image_reader(test_data_dir,test_label_dir,batch_size_test,image_size=[360,480,3])
		image_size=reader_test.image_size
		sess=tf.Session()
		test_data = tf.placeholder(tf.float32,shape=[batch_size_test, image_size[0], image_size[1], image_size[2]])
		test_labels = tf.placeholder(tf.int64, shape=[batch_size_test, image_size[0], image_size[1]])
		print 'Loading Segnet with %s layers'%net_num
		# net=Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,weights_path=os.path.join(BASE_DIR,'segnet_road.h5'))
		net=call_segnet(net_num,num_classes)
		test_logits=net.inference(test_data,is_training=False,reuse=False)
		print 'built network'
		prediction=tf.argmax(test_logits,axis=3)
		#
		print 'built loss graph'
		sess.run(tf.global_variables_initializer())
		print 'initialized vars'
		moment_vars=[]
		for var in tf.global_variables():
			if('moving_mean' in var.name or 'moving_variance' in var.name):
				moment_vars.append(var)
		saver=tf.train.Saver(tf.trainable_variables()+moment_vars)
		for name1 in [i for i in os.listdir(path1)]:
			print name1
			path2=os.path.join(path1,name1)
			path2=os.path.join(path2,'trial'+str(trial))
			print os.listdir(path2)
			epoch_number=98
			modelfile_name=os.path.join(path2,'modelfile_'+name1.split('-')[0]+name.split('-')[1]+'-'+str(epoch_number))
			print modelfile_name
			saver.restore(sess,modelfile_name)
			conf_mat=np.zeros([num_classes,num_classes])
			s_test=0;count_test=1
			while(reader_test.batch_num<reader_test.n_batches):
				[test_data_batch,test_label_batch]=reader_test.next_batch();
				feed_dict={test_data:test_data_batch,test_labels:test_label_batch};
				pred=sess.run(prediction,feed_dict=feed_dict)
				[corr,total_pix,matr]=transform_labels(pred,test_label_batch,num_classes)
				# viz=np.zeros([pred.shape[0]]+image_size)
				# for cl in range(num_classes):
				# 	t=np.where(pred==cl)
				# 	viz[t]=colors[cl,:]
				conf_mat=conf_mat+matr
				acc=corr*1.0/total_pix
				s_test=s_test+acc
				agg_acc=s_test/count_test
				count_test+=1
				print name,name1,'epoch:',reader_test.epoch+1,', Batch:',reader_test.batch_num, ', correct pixels:', corr, ', Accuracy:',acc,'Aggregate_acc:',agg_acc
				print '\n'
			reader_test.reset_reader()
			np.save(os.path.join(path2,name.split('-')[1]+'_'+name1.split('-')[0]+'_'+'confmat.npy'),conf_mat)
				# sp.imsave('outimgs-'+'retrain_4layer_moment'+'-%d-%f.png'%(reader_test.batch_num,acc),viz[0,:])
				
				# sp.imsave('outimgs_real-'+'retrain_4layer_moment'+'-%d-%f.png'%(reader_test.batch_num,acc),test_data_batch[0,:])
			
		tf.reset_default_graph()

if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument('-devbox',type=int,default=1)
	parser.add_argument('-ngpu',type=int,default=0)
	parser.add_argument('-trial',type=int,default=-1)
	args = parser.parse_args()
	# if args.trial == -1:
	# 	print "Enter -trial option"
	# 	sys.exit()

	# n_layers=7	
	if args.devbox:
	  BASE_DIR = '/root/segnet_vgg16'
	  os.environ['CUDA_VISIBLE_DEVICES']=str(args.ngpu)
	  print os.system('echo CUDA_VISBLE_DEVICES')
	else:
	  BASE_DIR = '/home/sriram/intern'
	  os.environ['CUDA_VISIBLE_DEVICES']="0"
	#for i in range(5):
	#	test_models(i+1)
	logfile_parser()
	#plot_generator()
