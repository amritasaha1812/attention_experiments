import sys
import numpy as np
import cPickle as pkl
import os

def get_batch_data(num_word_dim, max_steps, max_contexts, batch_size, num_types, data_dict):
	data_dict = np.array(data_dict)
	len_data_dict = data_dict.shape[0]
	batch_text_dict = np.concatenate(np.array(data_dict[:,0]),0).reshape(len_data_dict, max_contexts, max_steps, num_word_dim)
	batch_attn_dict = np.concatenate(np.array(data_dict[:,1]),0).reshape(len_data_dict, num_types)	
	if len(batch_text_dict)%batch_size!=0:
		batch_text_dict, batch_attn_dict = check_padding(batch_text_dict, batch_attn_dict, max_steps, max_contexts, num_word_dim, num_types, batch_size)
	#batch_text_dict is of shape (batch_size, max_contexts, max_steps, num_word_dim)
	batch_text_dict = np.asarray(batch_text_dict)
	batch_attn_dict = np.asarray(batch_attn_dict)		
	#print "batch_text_dict.shape ",batch_text_dict.shape
	batch_text_dict = batch_text_dict.transpose((1,2,0,3))
	return batch_text_dict, batch_attn_dict		

def check_padding(batch_text_dict, batch_attn_dict, max_steps, max_contexts, num_word_dim, num_types, batch_size):
	pad_size = batch_size - len(batch_text_dict)%batch_size	
	empty_data = [0.]*num_word_dim
	empty_data = [empty_data]*max_steps
	empty_data = [empty_data]*max_contexts
	empty_data = [empty_data]*pad_size
	batch_text_dict = batch_text_dict.tolist()
	batch_text_dict.extend(empty_data)
	empty_data = [0.]*num_types
	empty_data = [empty_data]*pad_size
	batch_attn_dict = batch_attn_dict.tolist()
	batch_attn_dict.extend(empty_data)	
	return batch_text_dict, batch_attn_dict

def clip_or_pad(data, max_contexts, max_steps, num_word_dim):
	if len(data)<max_contexts:
		pad_size = max_contexts - len(data)
		pad_data = [[[0.]*num_word_dim]*max_steps]*pad_size
		data.extend(pad_data)
	else:
		data = data[:max_contexts]
	for data_i in data:
		if len(data_i)<max_steps:
			pad_size = max_steps - len(data_i)
			pad_data = [[0.]*num_word_dim]*pad_size
			data_i.extend(pad_data)
		else:
			data_i = data_i[:max_steps]
	return data				

def read_data_to_numpy(data_file, data_processed_file, max_contexts, max_steps, num_word_dim, num_types):
	#each line is a training instance, each training instance [X,Y] where X is num_sequences*num_words*num_word_dim and Y is num_types 
	data = pkl.load(open(data_file))
	data_processed = []
	for data_i in data:
		data_ix = data_i[0]
		data_iy = data_i[1]
		data_ix = clip_or_pad(data_ix, max_contexts, max_steps, num_word_dim)
		#print len(data_ix), len(data_ix[0]), data_ix[0][0].shape
		data_processed.append([np.array(data_ix), np.array(data_iy)])
	if not os.path.exists(os.path.dirname(data_processed_file)):
		os.makedirs(os.path.dirname(data_processed_file))
	pkl.dump(data_processed, open(data_processed_file,'w'))

def read_structure_to_numpy(descendants_file, desc_processed_file, num_types):
	desc_dict = pkl.load(open(descendants_file))
	num_types = len(desc_dict)
	descs = [[0.]*num_types]*num_types
	for i in range(num_types):
		#descs[i][i]=1.
		for desc in desc_dict[i]:
			descs[i][desc] = 1.
		descs[i][i]=0.
	descs = np.array(descs, dtype=np.float32)
	if not os.path.exists(os.path.dirname(desc_processed_file)):
		os.makedirs(os.path.dirname(desc_processed_file))
	np.save(open(desc_processed_file,'w'), descs)
	
