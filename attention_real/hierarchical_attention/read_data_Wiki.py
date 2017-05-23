import json
import sys
import numpy as np
import os
import itertools
from prepare_data_Wiki import PrepareDataWiki
start_symbol_index = 0
end_symbol_index = 1
unk_symbol_index = 2
pad_symbol_index = 3
import os
import cPickle as pkl

def get_data_dict(param):
	train_file = param['train_data_file']
	valid_file = param['valid_data_file']
	test_file = param['test_data_file']
	train_file_prepro = param['train_data_file_prepro']
	valid_file_prepro = param['valid_data_file_prepro']
	test_file_prepro = param['test_data_file_prepro']
	vocab_file = param['vocab_file']
	vocab_stats_file = param['vocab_stats_file']
	ent_vocab_file = param['ent_vocab_file']
	ent_vocab_stats_file = param['ent_vocab_stats_file']
	type_vocab_file = param['type_vocab_file']
	type_vocab_stats_file = param['type_vocab_stats_file']
	types_file = param['types_file']
	types_per_level_file = param['types_per_level_file']
	types_hierarchy_file = param['descendants_file_prepro']
	dump_dir_loc = param['dump_dir_loc']
	max_len = param['max_len']
	max_len_mention = param['max_len_mention']
	max_len_labels = param['max_len_labels']
	type_order = param['type_order']
	cutoff = param['cutoff']
	print 'going to preparedata'
	
	preparedata = PrepareDataWiki(max_len, max_len_mention, max_len_labels, start_symbol_index, end_symbol_index, unk_symbol_index, pad_symbol_index, type_order, cutoff)
	preparedata.build_all_vocabs(ent_vocab_file, ent_vocab_stats_file, type_vocab_file, type_vocab_stats_file, types_file, types_per_level_file, types_hierarchy_file, train_file, valid_file, test_file)
	preparedata.prepare_data_from_file(train_file, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "train"), train_file_prepro)
	preparedata.prepare_data_from_file(valid_file, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "valid"), valid_file_prepro)
	preparedata.prepare_data_from_file(test_file, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "test"), test_file_prepro)

def pad(lists, size):
	pad_size = size - len(lists)
	width = len(lists[0])
	lists.extend([[pad_symbol_index]*width]*pad_size)
	return lists

def get_batch_data(data, batch_size):
	left_contextids = np.array(data)[:,0]
	right_contextids = np.array(data)[:,1]
	mention_wordids = np.array(data)[:,2]
	mention_labelids = np.array(data)[:,3]
	len_data = len(data)
	if len(left_contextids)<batch_size:
		left_contextids = pad(left_contextids.tolist(), batch_size)
	if len(right_contextids)<batch_size:
		right_contextids = pad(right_contextids.tolist(), batch_size)
	if len(mention_wordids)<batch_size:
		mention_wordids = pad(mention_wordids.tolist(), batch_size)
	if len(mention_labelids)<batch_size:
		mention_labelids = pad(mention_labelids.tolist(), batch_size)
	left_contextids = np.array([np.array(xi) for xi in left_contextids])
	#left_contextids is of dimension (batch_size * num_contexts * max_len)	
	right_contextids = np.array([np.array(xi) for xi in right_contextids])
	#right_contextids is of dimension (batch_size * num_contexts * max_len)   
	mention_wordids = np.array([np.array(xi) for xi in mention_wordids])
	#mention_wordids is of dimension (batch_size * num_contexts * max_len_mention)
	#print mention_wordids
	decoder_input = np.array([np.array(list(itertools.chain([pad_symbol_index],xi[1:]))) for xi in mention_labelids])
	#decoder_input is of dimension (batch_size * max_len_labels)
	sequence_weights = np.array([np.array([0 if xij==pad_symbol_index else 1 for xij in xi]) for xi in  mention_labelids])
	mention_labelids = np.array([np.array(xi) for xi in mention_labelids])
	#mention_labelids is of dimension (batch_size * max_len_labels)
	#print decoder_input[0], '::',mention_labelids[0], '::',sequence_weights[0]
	#print decoder_input[1], '::',mention_labelids[1], '::',sequence_weights[1]
	#print decoder_input[2], '::',mention_labelids[2], '::',sequence_weights[2]
	#print decoder_input[3], '::',mention_labelids[3], '::',sequence_weights[3]
	#print [mention_labelids[i] for i in range(batch_size)]
	left_contextids = np.transpose(left_contextids, (1,0))
	right_contextids = np.transpose(right_contextids, (1,0))
	mention_wordids = np.transpose(mention_wordids, (1,0))
	decoder_input = np.transpose(decoder_input, (1,0))
	sequence_weights = np.transpose(sequence_weights, (1,0))
	mention_labelids = np.transpose(mention_labelids, (1,0))

	#print 'shape of mention_labelids ', mention_labelids.shape	
	return left_contextids, right_contextids, mention_wordids, mention_labelids, decoder_input, sequence_weights	

if __name__=="__main__":
	params = json.load(open('params.json'))
	get_data_dict(params)
	train_data = pkl.load(open(params['train_data_file_prepro']))
	batch_size = params['batch_size']
	get_batch_data(train_data[:batch_size])	 					
