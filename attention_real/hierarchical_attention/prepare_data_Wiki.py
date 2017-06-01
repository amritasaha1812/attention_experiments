import os
import json
import logging
import collections
from collections import Counter
import cPickle as pkl
import numpy as np
class PrepareDataWiki:	
	

	def __init__(self, max_len, max_len_mention, max_len_labels, start_symbol_index, end_symbol_index, unk_symbol_index, pad_symbol_index, type_order, cutoff=-1):
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger('prepare_data_wiki')
		self.max_len = max_len
		self.max_len_mention = max_len_mention
		self.max_len_labels = max_len_labels
		self.unk_word_id = unk_symbol_index
		self.start_word_id = start_symbol_index
		self.pad_word_id = pad_symbol_index
		self.end_word_id = end_symbol_index
		self.start_word_symbol = '</s>'
		self.end_word_symbol = '</e>'
		self.pad_symbol = '</pad>'
		self.unk_symbol = '<unk>'
		self.cutoff = cutoff
		self.input = None
		self.output = None
		self.vocab_file = None
		self.vocab_dict = None
		self.ent_vocab_file = None
		self.ent_vocab_dict = None
		self.type_vocab_file = None
		self.type_vocab_dict = None
		self.word_counter = None
		self.ent_counter = None
		self.type_word_counter = None			
		self.types = set([])
		self.types_per_level = {}
		self.type_hierarchy = {}
		self.types_file = None
		self.types_per_level_file = None
		self.types_hierarchy_file = None
		self.type_order = type_order

	def safe_pickle(self, obj, filename):
		if os.path.isfile(filename):
			self.logger.info("Overwriting %s." % filename)
		else:
			self.logger.info("Saving to %s." % filename)
		with open(filename, 'wb') as f:
			pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

        def retain_most_specific_labels(self, labels):
                if isinstance(labels, basestring):
                       labels = [labels]
                if isinstance(labels, set):
                        labels = list(labels)
                labels = [l.strip('/') for l in labels]
                labels.sort(lambda x,y: cmp(len(x),len(y)), reverse=True)
                to_remove = set([''])
                for index,l in enumerate(labels):
                        if any([l in x for x in labels[:index]]):
                                to_remove.add(l)
                path = list(set(labels) - to_remove)
                return path

	def create_type_hierarchy(self):
		#self.types = set(self.retain_most_specific_labels(self.types))
		self.types_per_level = {}
		self.type_hierarchy = np.zeros((len(self.type_vocab_dict), len(self.type_vocab_dict)))
		for type_ in self.types:
			type_=type_.split("/")
			print type_	
			for i,t in enumerate(type_):
				level = i
				if level not in self.types_per_level:
					self.types_per_level[level] = set([])
				self.types_per_level[level].add(self.type_vocab_dict[t])
			 
			for i,t in enumerate(type_):
				if len(t.strip())==0:
					continue	
				indices = [self.type_vocab_dict[type_i] for type_i in type_[i+1:]]
				index = self.type_vocab_dict[t]
				self.type_hierarchy[index][indices] = 1.0	
		self.types_per_level = collections.OrderedDict(sorted(self.types_per_level.items())).values()
		self.types_per_level = [np.asarray(list(i)) for i in self.types_per_level] 
		return 						

	def build_all_vocabs(self, ent_vocab_file, ent_vocab_stats_file, type_vocab_file, type_vocab_stats_file, types_file, types_per_level_file, type_hierarchy_file, train_file, valid_file, test_file):
		self.ent_vocab_file = ent_vocab_file
                self.ent_vocab_stats_file = ent_vocab_stats_file
                self.type_vocab_file = type_vocab_file
                self.type_vocab_stats_file = type_vocab_stats_file
                self.types_file = types_file
		self.types_per_level_file = types_per_level_file
                self.type_hierarchy_file = type_hierarchy_file
		self.ent_word_counter = Counter()
		self.type_word_counter = Counter()	
		for file in [train_file, valid_file, test_file]:
			for line in open(file).readlines():
        	                data = json.loads(line)	
				tokens = [x.lower().strip() for x in data['tokens']]
				mentions = data['mentions']
				for mention in mentions:
                                	mention_start = mention['start']
	                                mention_end = mention['end']
        	                        mention_words = tokens[mention_start:mention_end]
                	                self.ent_word_counter.update(mention_words)
                        	        mention_labels = []
                                	for x in self.retain_most_specific_labels(mention['labels']):		
						labels = [xi.strip() for xi in x.lower().split("/") if len(xi.strip())>0]
						if self.type_order=="bottom_up":
							labels = list(reversed(labels))
						self.types.add("/".join(labels))	
	                                        if len(labels)<self.max_len_labels-2:
        	                                        labels = [self.start_word_symbol] + labels + [self.end_word_symbol] + [self.pad_symbol]*(self.max_len_labels-2-len(labels))
                	                        elif len(labels)>self.max_len_labels-2:
                        	                        labels = [self.start_word_symbol] + labels[:self.max_len_labels-2] + [self.end_word_symbol]
                                	        else:
                                        	        labels = [self.start_word_symbol] + labels + [self.end_word_symbol]
	                                        mention_labels.append(labels)
                	                for mention_label in mention_labels:
                        	                self.type_word_counter.update(mention_label)
		self.ent_vocab_dict = self.build_vocab('entity',self.ent_word_counter, self.ent_vocab_file, self.ent_vocab_dict, self.cutoff)
                self.type_vocab_dict = self.build_vocab('type',self.type_word_counter, self.type_vocab_file, self.type_vocab_dict, 0)
		self.safe_pickle(self.ent_word_counter, self.ent_vocab_stats_file)
                inv_ent_vocab_dict = {word_id:word for word,word_id in self.ent_vocab_dict.iteritems()}
                self.safe_pickle(inv_ent_vocab_dict, self.ent_vocab_file)
                print 'len(ent_vocab_dict) ', len(self.ent_vocab_dict)
                print 'len(type_vocab_dict) ', len(self.type_vocab_dict)
                self.safe_pickle(self.type_word_counter, self.type_vocab_stats_file)
                inv_type_vocab_dict = {word_id:word for word,word_id in self.type_vocab_dict.iteritems()}
		self.create_type_hierarchy()
                self.safe_pickle(self.types_per_level, self.types_per_level_file)
                np.save(open(self.type_hierarchy_file, 'w'), self.type_hierarchy)
                self.safe_pickle(self.types, self.types_file)
                self.safe_pickle(inv_type_vocab_dict, self.type_vocab_file)	
	
	def prepare_data_from_file(self, file, vocab_file, vocab_stats_file, output, data_pkl_file):
		self.vocab_file = vocab_file
		self.vocab_stats_file = vocab_stats_file
		self.output = output	
		self.output_string_file = self.output+"_string.pkl"
		if os.path.isfile(self.types_file):
                        self.types = set(pkl.load(open(self.types_file)))
		if os.path.isfile(vocab_file):	
			print 'found pre-existing vocab file .. reusing it'
			create_vocab = False
		else:
			create_vocab = True
			self.word_counter = Counter()
		data_to_dump = {}
		for line in open(file).readlines():
			data = json.loads(line)
			tokens = [x.lower().strip() for x in data['tokens']]
			if create_vocab:
				self.word_counter.update(tokens)
			senid = data['senid']
			mentions = data['mentions']
			fileid = data['fileid']
			for mention in mentions:
				mention_start = mention['start']
				mention_end = mention['end']
				mention_words = tokens[mention_start:mention_end]
				#self.ent_word_counter.update(mention_words)	
				mention_labels = []
				for x in mention['labels']:
					labels = [xi.strip() for xi in x.lower().split("/") if len(xi.strip())>0]
					if self.type_order=="bottom_up":
                                                labels = list(reversed(labels))
					if len(labels)<self.max_len_labels-2:
						labels = [self.start_word_symbol] + labels + [self.end_word_symbol] + [self.pad_symbol]*(self.max_len_labels-2-len(labels))
					elif len(labels)>self.max_len_labels-2:
						labels = [self.start_word_symbol] + labels[:self.max_len_labels-2] + [self.end_word_symbol]
					else:
						labels = [self.start_word_symbol] + labels + [self.end_word_symbol]
					mention_labels.append(labels)
					#self.types.add(x)
				#for mention_label in mention_labels:	
				#	self.type_word_counter.update(mention_label)	 
				left_context_start = mention_start - self.max_len + 2 if mention_start - self.max_len + 2 > 0  else 0
				right_context_end = mention_end + self.max_len - 2 if mention_end + self.max_len - 2 <= len(tokens) else len(tokens)
				mention_left_context = tokens[left_context_start:mention_start]
				mention_right_context = tokens[mention_end:right_context_end]
				if len(mention_left_context)>(self.max_len - 2) or len(mention_right_context)>(self.max_len-2):
					raise Exception('len(mention_context) is not max_len-2 '+str(len(mention_left_context)))
				mention_left_context = [self.start_word_symbol] + mention_left_context + [self.end_word_symbol]
				mention_right_context = [self.start_word_symbol] + mention_right_context + [self.end_word_symbol]	
				if len(mention_left_context)<self.max_len:
					mention_left_context=mention_left_context+[self.pad_symbol]*(self.max_len - len(mention_left_context))
				elif len(mention_left_context)>self.max_len:
					raise Exception('len(mention_left_context) should not be greater than self.max_len')
				if len(mention_right_context)<self.max_len:
					mention_right_context=mention_right_context+[self.pad_symbol]*(self.max_len - len(mention_right_context))
				elif len(mention_right_context)>self.max_len:
					raise Exception('len(mention_right_context) should not be greater than self.max_len')
				if len(mention_words)<(self.max_len_mention-2):
					mention_words = [self.start_word_symbol] + mention_words + [self.end_word_symbol] + [self.pad_symbol]*(self.max_len_mention-2-len(mention_words))
				elif len(mention_words)>(self.max_len_mention-2):
					mention_words = [self.start_word_symbol] + mention_words[:self.max_len_mention-2] + [self.end_word_symbol]
				else:
					mention_words = [self.start_word_symbol] + mention_words + [self.end_word_symbol]
				if len(mention_left_context)!=self.max_len:
					raise Exception('len(mention_left_context)!=self.max_len')
				if len(mention_right_context)!=self.max_len:
					raise Exception('len(mention_right_context)!=self.max_len')
				if len(mention_words)!=self.max_len_mention:
					raise Exception('len(mention_words)!=self.max_len_mention')	
				for mention_label in mention_labels:
					mention_label_str  = '|'.join(mention_label)
					if mention_label_str in data_to_dump:
						data_to_dump[mention_label_str].append([mention_left_context,mention_right_context, mention_words])
					else:
						data_to_dump[mention_label_str] = [[mention_left_context,mention_right_context, mention_words]]	
		pkl.dump(data_to_dump, 	open(self.output_string_file,'w'))
		if create_vocab:
			self.vocab_dict = self.build_vocab('words',self.word_counter,self.vocab_file, self.vocab_dict, self.cutoff)
		else:
			self.vocab_dict = self.read_vocab(self.vocab_file, self.vocab_dict)
		self.ent_vocab_dict = self.read_vocab(self.ent_vocab_file, self.ent_vocab_dict)
		self.type_vocab_dict = self.read_vocab(self.type_vocab_file, self.type_vocab_dict)
		if create_vocab or not os.path.exists(data_pkl_file):
			self.binarize_corpus(self.output_string_file, data_pkl_file)
	
								
	def read_vocab(self, vocab_file, vocab_dict):
		assert os.path.isfile(vocab_file)
		vocab_dict = {word:word_id for word_id, word in pkl.load(open(vocab_file, "r")).iteritems()}
		assert self.start_word_symbol in vocab_dict
		assert self.end_word_symbol in vocab_dict
		assert self.pad_symbol in vocab_dict
		return vocab_dict

	def build_vocab(self,type_of_vocab, word_counter, vocab_file,  vocab_dict, cutoff):
		total_freq = sum(word_counter.values())
		self.logger.info('Total %s frequency in dictionary %d ',type_of_vocab, total_freq)
		self.logger.info('For %s, Cutoff %d',type_of_vocab, self.cutoff)						
		vocab_count = [x for x in word_counter.most_common() if x[1]>cutoff]
		if vocab_dict is None:
			vocab_dict = {}
		self.safe_pickle(vocab_count, vocab_file.replace('.pkl','_counter.pkl'))
		vocab_dict = {self.unk_symbol:self.unk_word_id, self.start_word_symbol:self.start_word_id, self.pad_symbol:self.pad_word_id, self.end_word_symbol:self.end_word_id}
                i = 4
                for (word, count) in vocab_count:
                        if not word in vocab_dict:
                                vocab_dict[word] = i
                                i += 1
                self.logger.info('For %s, Vocab size %d', type_of_vocab, len(vocab_dict))
		return vocab_dict
	
	def binarize_corpus(self, output_string_file, data_pkl_file):
		binarized_corpus = []
		num_instances = 0
		data = pkl.load(open(output_string_file))
		for mention_label_str,data_instances in data.iteritems():
			mention_label = mention_label_str.split('|')
			mention_labelids = [self.type_vocab_dict[x] if x in self.type_vocab_dict else self.unk_word_id for x in mention_label]
			for data_instance in data_instances:
				mention_left_context = data_instance[0]	
				mention_right_context = data_instance[1]	
				mention_words = data_instance[2]
				mention_left_contextids = [self.vocab_dict[x] if x in self.vocab_dict else self.unk_word_id for x in mention_left_context]
				mention_right_contextids = [self.vocab_dict[x] if x in self.vocab_dict else self.unk_word_id for x in mention_right_context]	
				mention_wordids = [self.ent_vocab_dict[x] if x in self.ent_vocab_dict else self.unk_word_id for x in mention_words]
				if len(mention_left_context)!=len(mention_left_contextids) or len(mention_right_context)!=len(mention_right_contextids) or len(mention_words)!=len(mention_wordids):
					raise Exception('len(mention_left_context)!=len(mention_left_contextids) or len(mention_right_context)!=len(mention_right_contextids) or len(mention_words)!=len(mention_wordids) ---> ',len(mention_left_context),len(mention_left_contextids), len(mention_right_context),len(mention_right_contextids),len(mention_words),len(mention_wordids))
				binarized_corpus.append([mention_left_contextids, mention_right_contextids, mention_wordids, mention_labelids])
		self.safe_pickle(binarized_corpus, data_pkl_file)	
		if not os.path.isfile(self.vocab_file):
			self.safe_pickle(self.word_counter, self.vocab_stats_file)
			inv_vocab_dict = {word_id:word for word,word_id in self.vocab_dict.iteritems()}
			self.safe_pickle(inv_vocab_dict, self.vocab_file)
