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
from read_data_Wiki import *

def feeding_dict(model, input_left_contexts, input_right_contexts, mentions, labels, decoder_inputs, sequence_weights, feed_prev):
	feed_dict = {}
	for model_input_left_context, input_left_context in zip(model.input_left_contexts, input_left_contexts):
		feed_dict[model_input_left_context] = input_left_context
	for model_input_right_context, input_right_context in zip(model.input_right_contexts, input_right_contexts):
		feed_dict[model_input_right_context] = input_right_context
	for model_input_mention, mention in zip(model.input_mentions, mentions):
		feed_dict[model_input_mention] = mention
	for model_decoder_label, decoder_label in zip(model.decoder_label,labels):
		feed_dict[model_decoder_label] = decoder_label
	for model_decoder_input, decoder_input in zip(model.decoder_input, decoder_inputs):
		feed_dict[model_decoder_input] = decoder_input
	for model_decoder_sequence_weight, sequence_weight in zip(model.decoder_sequence_weight, sequence_weights):
		feed_dict[model_decoder_sequence_weight] = sequence_weight		
	feed_dict[model.feed_previous] = feed_prev	
	return feed_dict

def run_training(param):
	def perform_training(model, batch_dict, feed_prev, batch_size, step, show_grad_freq):
		train_batch_left_context, train_batch_right_context, train_batch_mention, train_batch_label, train_batch_decoder_input, train_batch_sequence_weight = get_batch_data(batch_dict, batch_size)
		feed_dict = feeding_dict(model, train_batch_left_context, train_batch_right_context, train_batch_mention, train_batch_label, train_batch_decoder_input, train_batch_sequence_weight, feed_prev)
		loss, dec_op, grads, _ = sess.run([losses, logits, gradients, train_op], feed_dict=feed_dict)		
		loss = np.array(loss)
		sum_loss = np.sum(loss)	
		if step % show_grad_freq == 0:
	                grad_vals = sess.run(gradients, feed_dict=feed_dict)
        	        var_to_grad = {}
                	for grad_val, var in zip(grad_vals, gradients):
                        	if type(grad_val).__module__ == np.__name__:
                                	var_to_grad[var.name] = grad_val
	                                sys.stdout.flush()
        	                        print 'var.name ', var.name, 'shape(grad) ',grad_val.shape, 'mean(grad) ',np.mean(grad_val)
                	                sys.stdout.flush()  
		return sum_loss

	def get_valid_loss(model, batch_dict, batch_size):
		valid_batch_left_context, valid_batch_right_context, valid_batch_mention, valid_batch_label, valid_batch_decoder_input, valid_batch_sequence_weight = get_batch_data(batch_dict, batch_size)
		feed_dict = feeding_dict(model, valid_batch_left_context, valid_batch_right_context, valid_batch_mention, valid_batch_label, valid_batch_decoder_input, valid_batch_sequence_weight, True)
		loss, dec_op = sess.run([losses, logits], feed_dict=feed_dict)
		loss = np.array(loss)
		#print 'shape of valid_batch_label ',valid_batch_label.shape
		return loss, dec_op, valid_batch_label

	def evaluate(model, valid_data, batch_size, type_vocab, step, epoch):
		print 'Validation started'
		sys.stdout.flush()
		valid_loss = 0.
		batch_predicted_labels = []	
		n_batches = int(math.ceil(float(len(valid_data))/float(param['batch_size'])))
		for i in range(n_batches):
			batch_dict = valid_data[i*batch_size:(i+1)*batch_size]
			valid_batch_loss = perform_validation(model, batch_dict, batch_size, type_vocab, step, epoch)
			valid_loss = valid_loss + valid_batch_loss
		valid_loss = valid_loss/float(n_batches)
		return valid_loss
	
	def perform_validation(model, batch_dict, batch_size, type_vocab, step, epoch):
		batch_valid_loss, dec_op, valid_batch_label_ids = get_valid_loss(model, batch_dict, batch_size)
		batch_gold_labels =  map_id_to_word(np.transpose(valid_batch_label_ids), type_vocab)
		batch_predicted_labels, batch_predicted_prob, batch_true_prob = get_labels_prob(dec_op, valid_batch_label_ids, type_vocab)	
		print_pred_true_op(batch_predicted_labels, batch_predicted_prob, batch_true_prob, batch_gold_labels, step, epoch, batch_valid_loss)	
		return np.sum(batch_valid_loss)

	def map_id_to_word(ids_list, vocab):
		labels_list = []
		for ids in ids_list:
			labels = " ".join([ vocab[i] for i in ids if vocab[i]!='</pad>'])
			labels_list.append(labels)
		return labels_list
	
	def get_labels_prob(pred_op, true_op, type_vocab):
		true_op = np.asarray(true_op)
		#true_op is of shape (max_len_labels * batch_size)
		#pred_op is a list of max_len_labels size, of tensors of dimension (batch_size * num_types)	
		max_probs_index = []
		max_probs = []
		#print 'true_op.shape ', true_op.shape
		#print 'len(pred_op) ', len(pred_op)
		#print 'pred_op[0].shape ',pred_op[0].shape
		if true_op is not None:
			true_op_prob = []
		i=0
		for op in pred_op:
			#op is a tensor of dimension (batch_size * num_types) 
			sys.stdout.flush()
			max_index = np.argmax(op, axis=1)
			#max_index is a tensor of dimension batch_size
			max_prob = np.max(op, axis=1)
			#print 'op.shape ',op.shape, 'max_prob.shape ',max_prob.shape, 'max_index.shape ',max_index.shape
			#max_prob is a tensor of dimension batch_size
			max_probs.append(max_prob)
			max_probs_index.append(max_index)
			if true_op is not None:
				op = op.tolist()
				#op is a list of batch_size tensors of shape num_types
				#true_op[i] is a tensor of shape batch_size
				#t_ij is an integer (between 0 and num_types-1)
				#v_ij is a tensor of shape num_types
				true_prob = [v_ij[t_ij] for v_ij,t_ij in zip(op, true_op[i].tolist())]
				#true_prob is a list of length batch_size
				true_op_prob.append(true_prob)
				i=i+1
		#true_op_prob is a 2d list of shape (max_len_labels, batch_size)
		#max_probs_index is a list of max_len size; each of tensors of dimension batch_size
		max_probs_index = np.asarray(max_probs_index)
		max_probs_index = np.transpose(max_probs_index)
		#max_probs_index is a tensor of dimension batch_size * max_len	

		#max_probs is a list of max_len size; each of tensors of dimension batch_size   
		max_probs = np.asarray(max_probs)
		max_probs = np.transpose(max_probs)
		#print 'max_prob.shape ',max_probs.shape, 'max_index.shape ',max_probs_index.shape	
		#max_probs is a tensor of dimension batch_size * max_len
		if true_op is not None:
			true_op_prob = np.asarray(true_op_prob)
			true_op_prob = np.transpose(true_op_prob)
			#print ' true_op_prob.shape ',true_op_prob.shape
			#true_op_prob is a tensor of shape (batch_size, max_len_labels)
			if true_op_prob.shape[0]!=max_probs.shape[0] or true_op_prob.shape[1]!=max_probs.shape[1]:
				raise Exception('some problem with shape match')
		pred_labels_list = map_id_to_word(max_probs_index, type_vocab)
		return pred_labels_list, max_probs, true_op_prob
	
	def print_pred_true_op(pred_op, prob_pred, prob_true, true_op, step, epoch, batch_valid_loss):
		for i in range(0, len(true_op),2):
			print 'true sentence in step '+str(step)+' of epoch '+str(epoch)+' is'
			sys.stdout.flush()
			print true_op[i]
			print "\n"
		        print "predicted sentence in step "+str(step)+" of epoch "+str(epoch)+" is:"
	            	sys.stdout.flush()
           	 	print pred_op[i]
	            	print "\n"
	            	print "prob of predicted words in step "+str(step)+" of epoch "+str(epoch)+" is:"
	            	sys.stdout.flush()
	            	print prob_pred[i]
	            	print "\n"
	            	print "prob of true words in step "+str(step)+" of epoch "+str(epoch)+" is:"
	            	sys.stdout.flush()
	            	print prob_true[i]
            		print "\n"
            		#print "crossent  in step "+str(step)+" of epoch "+str(epoch)+" is:"
            		#sys.stdout.flush()
            		#print sum([math.log(x+1e-12) for x in prob_true[i]])               
            		#print "\n"
            		sys.stdout.flush()
           		print "loss for the pair of true and predicted sentences", str(batch_valid_loss[i])
            		print "\n"	
			
	descs = np.load(param['descendants_file_prepro'])
	train_data = pkl.load(open(param['train_data_file_prepro']))
	random.shuffle(train_data)
	print 'Train Data loaded'
	sys.stdout.flush()
	valid_data = pkl.load(open(param['valid_data_file_prepro']))
	random.shuffle(valid_data)	
	print 'Valid Data loaded'
	types_per_level = pkl.load(open(param['types_per_level_file']))
	types_per_level = [np.asarray(i) for i in types_per_level]	
	num_types_per_level = [i.shape[0] for i in types_per_level]
	sys.stdout.flush()	
	words_vocab = pkl.load(open(param['vocab_file']))
	type_vocab = pkl.load(open(param['type_vocab_file']))
	mention_vocab = pkl.load(open(param['ent_vocab_file']))
	num_words = len(words_vocab)
	num_types = len(type_vocab)
	num_mention_words = len(mention_vocab)
	num_levels = len(num_types_per_level)
	param['num_words'] = num_words
	param['num_types'] = num_types
	param['num_mention_words'] = num_mention_words	
	n_batches = int(math.ceil(float(len(train_data))/float(param['batch_size'])))
	model_file = os.path.join(param['model_path'],"best_model")
	attention_type = param['attention_type']
	if attention_type=="TypeAttention.CHANGE_OF_VARIABLES":
		attention_type = TypeAttention.CHANGE_OF_VARIABLES
	elif attention_type=="TypeAttention.CHANGE_OF_VARIABLES_MAX":
                attention_type = TypeAttention.CHANGE_OF_VARIABLES_MAX
	elif attention_type=="TypeAttention.STANDARD_ATTENTION":
		attention_type = TypeAttention.STANDARD_ATTENTION
	elif attention_type=="TypeAttention.STRUCTURED_ATTENTION":
		attention_type = TypeAttention.STRUCTURED_ATTENTION
	else:
		print 'wrong attention type provided'
		sys.exit(1)
	with tf.Graph().as_default():
		print 'attention_type' ,attention_type
		model = TypeAttention(param['max_len'], param['max_len_mention'], param['max_len_labels'], param['hidden_size'], param['num_type_dim'], param['num_word_dim'], param['num_mention_dim'], param['num_types'], num_types_per_level, types_per_level, num_levels, param['num_words'], param['num_mention_words'], param['learning_rate'], param['batch_size'],param['max_gradient_norm'], attention_type, param['output_activation'],descs)
		model.create_placeholder()
		logits = model.inference()
		losses = model.loss(logits)
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
			if epoch==0:
				feed_prev = False
			else:
				feed_prev = True
			for i in range(n_batches):
				overall_step_count = overall_step_count+1
				train_batch_dict = train_data[i*param['batch_size']:(i+1)*param['batch_size']]
				train_batch_loss = perform_training(model, train_batch_dict, feed_prev, param['batch_size'], overall_step_count, param['show_grad_freq'])
				if overall_step_count%param['show_freq']==0:
					print ('Epoch %d Step %d train loss (avg over batch) = %.6f' %(epoch, i, train_batch_loss))
				sys.stdout.flush()
				train_loss = train_loss + train_batch_loss
				if overall_step_count>0 and (overall_step_count%param['valid_freq']==0 or i==n_batches-1):
					overall_avg_valid_loss = evaluate(model, valid_data, param['batch_size'], type_vocab, i, epoch)
					print ('Epoch %d Step %d valid loss (avg overall) = %.6f' %(epoch, i, overall_avg_valid_loss))
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
	get_data_dict(param)
        print 'preprocessed the data'
        sys.stdout.flush()
    run_training(param)

if __name__=="__main__":
    main()
			
