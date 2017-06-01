import tensorflow as tf
import json
import math
import sys
import os
import pickle as pkl
import random
import numpy as np
import nltk
from network import *
from read_data import *

def feeding_dict(model, inputs_text, inputs_attn_wts):
	feed_dict = {}
	for input_sent_embedding, input_text in zip(model.input_sent_embedding, inputs_text):
		for input_sent_embedding_i, input_text_i in zip(input_sent_embedding, input_text):
			feed_dict[input_sent_embedding_i] = input_text_i
	feed_dict[model.input_attn] = inputs_attn_wts
	return feed_dict

def run_training(param):
	def get_train_loss(model, batch_dict):
		train_batch_text, train_batch_attn_wts = get_batch_data(param['num_word_dim'], param['max_steps'], param['max_contexts'],  param['batch_size'], param['num_types'], batch_dict)
		feed_dict = feeding_dict(model, train_batch_text, train_batch_attn_wts)
		loss, predicted_attn, _ = sess.run([losses, pred_attn, train_op], feed_dict=feed_dict)				
		predicted_attn = np.array(predicted_attn)
		return loss, predicted_attn

	def get_valid_loss(model, batch_dict):
		valid_batch_text, valid_batch_attn_wts = get_batch_data(param['num_word_dim'], param['max_steps'], param['max_contexts'], param['batch_size'], param['num_types'], batch_dict)
		feed_dict = feeding_dict(model, valid_batch_text, valid_batch_attn_wts)
		loss, predicted_attn, mean_predicted_attn, mean_gold_attn = sess.run([losses, pred_attn, mean_pred_attn, mean_input_attn], feed_dict=feed_dict)
		predicted_attn = np.array(predicted_attn)
		return loss, predicted_attn, mean_predicted_attn, mean_gold_attn

	def evaluate(model, valid_data, batch_size):
		print 'Validation started'
		sys.stdout.flush()
		valid_loss = 0.
		n_batches = int(math.ceil(float(len(valid_data))/float(param['batch_size'])))
		overall_mean_pred_attn = 0.
		overall_mean_gold_attn = 0.
		for i in range(n_batches):
			batch_dict = valid_data[i*batch_size:(i+1)*batch_size]
			valid_batch_loss, mean_predicted_attn, mean_gold_attn = perform_validation(model, batch_dict)
			valid_loss = valid_loss + valid_batch_loss
			overall_mean_pred_attn = overall_mean_pred_attn + mean_predicted_attn
			overall_mean_gold_attn = overall_mean_gold_attn + mean_gold_attn
		overall_mean_pred_attn = overall_mean_pred_attn/float(n_batches)
		overall_mean_gold_attn = overall_mean_gold_attn/float(n_batches)
		valid_loss = valid_loss/float(n_batches)
		return valid_loss, overall_mean_pred_attn, overall_mean_gold_attn
	
	def perform_validation(model, batch_dict):
		batch_valid_loss, pred_attn, mean_predicted_attn, mean_gold_attn = get_valid_loss(model, batch_dict)
		return batch_valid_loss, mean_predicted_attn, mean_gold_attn
			
	def perform_training(model, batch_dict):
		batch_train_loss, pred_attn = get_train_loss(model, batch_dict)
		return batch_train_loss

	descs = np.load(param['descendants_file_prepro'])
	train_data = pkl.load(open(param['train_data_file']))
	print 'Train Data loaded'
	sys.stdout.flush()
	valid_data = pkl.load(open(param['valid_data_file']))
	print 'Valid Data loaded'
	sys.stdout.flush()
	n_batches = int(math.ceil(float(len(train_data))/float(param['batch_size'])))
	if not os.path.exists(param['model_path']):
		os.makedirs(param['model_path'])
	model_file = os.path.join(param['model_path'],"best_model")
	attention_type = param['attention_type']
	if attention_type=="TypeAttention.CHANGE_OF_VARIABLES_MAX":
		attention_type = TypeAttention.CHANGE_OF_VARIABLES_MAX
	if attention_type=="TypeAttention.CHANGE_OF_VARIABLES_SUM":
                attention_type = TypeAttention.CHANGE_OF_VARIABLES_SUM
	elif attention_type=="TypeAttention.STANDARD_ATTENTION":
		attention_type = TypeAttention.STANDARD_ATTENTION
	elif attention_type=="TypeAttention.STRUCTURED_ATTENTION":
		attention_type = TypeAttention.STRUCTURED_ATTENTION
	else:
		print 'wrong attention type provided'
		sys.exit(1)
	with tf.Graph().as_default():
		model = TypeAttention(param['max_steps'], param['max_contexts'], param['hidden_size'], param['num_word_dim'], param['num_type_dim'], param['num_types'], param['learning_rate'], param['max_gradient_norm'], descs, attention_type)
		model.create_placeholder()
		pred_attn = model.inference()
		losses, mean_pred_attn, mean_input_attn = model.loss(pred_attn)
		train_op, gradients = model.train(losses)
		print 'model created'
		sys.stdout.flush()
		saver = tf.train.Saver()
		init = tf.initialize_all_variables()
		sess = tf.Session()
		if os.path.isfile(model_file):
			print 'best model exists .. restoring from that point'
			saver.restore(sess, model_file)
		else:
			print 'initializing fresh variables'
			sess.run(init)
		best_valid_loss = float("inf")
		best_valid_epoch=0
		all_var = tf.all_variables()
		print 'printing all ', len(all_var), 'TF variables '
		for var in all_var:
			print var.name, var.get_shape()
		print 'training started'
		sys.stdout.flush()
		overall_step_count = 0
		last_overall_avg_train_loss = float("inf")
		for epoch in range(param['max_epochs']):
			random.shuffle(train_data)
			train_loss = 0
			for i in range(n_batches):
				overall_step_count = overall_step_count+1
				train_batch_dict = train_data[i*param['batch_size']:(i+1)*param['batch_size']]
				train_batch_loss = perform_training(model, train_batch_dict)
				print ('Epoch %d Step %d train loss (avg over batch) = %.6f' %(epoch, i, train_loss))
				sys.stdout.flush()
				train_loss = train_loss + train_batch_loss
				if overall_step_count>0 and (overall_step_count%param['valid_freq']==0 or i==n_batches-1):
					overall_avg_valid_loss, overall_mean_pred_attn, overall_mean_gold_attn = evaluate(model, valid_data, param['batch_size'])
					print ('Epoch %d Step %d valid loss (avg overall) = %.6f (mean_pred_attn=%.6f mean_gold_attn=%.6f)' %(epoch, i, overall_avg_valid_loss, overall_mean_pred_attn, overall_mean_gold_attn))
					sys.stdout.flush()
					if best_valid_loss>overall_avg_valid_loss:
						saver.save(sess, model_file)
						best_valid_loss = overall_avg_valid_loss
					else:
						continue
			overall_avg_train_loss = train_loss/float(n_batches)
			print('Epoch  %d training completed, train loss (avg overall) = %.6f' %(epoch, overall_avg_train_loss))
			if last_overall_avg_train_loss is not None and overall_avg_train_loss > last_overall_avg_train_loss:
				diff = overall_avg_train_loss - last_overall_avg_train_loss
				if diff>param['train_loss_incremenet_tolerance']:
					print 'WARNING: training loss (%.6f) has increased by %.6f since last epoch, has exceed tolerance of %f ' %(overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance'])
		                else:
        		                print 'WARNING: training loss (%.6f) has increased by %.6f since last epoch, but still within tolerance of %f ' %(overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance'])		  				
			last_overall_avg_train_loss = overall_avg_train_loss
	            	sys.stdout.flush()
	        print 'Training over'
	        print 'Evaluating on test data'
	f_out.close()

def main():
    param = json.load(open(sys.argv[1]))
    print param
    if os.path.exists(param['train_data_file_prepro']) and os.path.exists(param['valid_data_file_prepro']) and os.path.exists(param['test_data_file_prepro']):
        print 'preprocessed data already exists'
        sys.stdout.flush()
    else:
        read_data_to_numpy(param['train_data_file'], param['train_data_file_prepro'], param['max_contexts'], param['max_steps'], param['num_word_dim'], param['num_types'])
	read_data_to_numpy(param['valid_data_file'], param['valid_data_file_prepro'], param['max_contexts'], param['max_steps'], param['num_word_dim'], param['num_types'])
	read_data_to_numpy(param['test_data_file'], param['test_data_file_prepro'], param['max_contexts'], param['max_steps'], param['num_word_dim'], param['num_types'])
	read_structure_to_numpy(param['descendants_file'], param['descendants_file_prepro'], param['num_types'])	
        print 'preprocessed the data'
        sys.stdout.flush()
    run_training(param)

if __name__=="__main__":
    main()
			
