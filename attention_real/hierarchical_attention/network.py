import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn,rnn_cell
import math
import os
import sys
from tensorflow.python.ops import control_flow_ops
import seq2seq
from seq2seq import *
from tensorflow.python.ops import variable_scope

class TypeAttention:
	CHANGE_OF_VARIABLES = "CHANGE_OF_VARIABLES"
	CHANGE_OF_VARIABLES_MAX = "CHANGE_OF_VARIABLES_MAX"
        STRUCTURED_ATTENTION = "STRUCTURED_ATTENTION"
        STANDARD_ATTENTION = "STANDARD_ATTENTION"

	def __init__(self, num_steps, num_steps_mention, num_steps_label, hidden_size, num_type_dim, num_word_dim, num_mention_dim, num_types, num_types_per_level, types_per_level, num_levels, num_words, num_mention_words, learning_rate, batch_size, max_gradient_norm, attention_type, output_activation, descs):
		self.CHANGE_OF_VARIABLES = "CHANGE_OF_VARIABLES"
		self.CHANGE_OF_VARIABLES_MAX = "CHANGE_OF_VARIABLES_MAX"
		self.STRUCTURED_ATTENTION = "STRUCTURED_ATTENTION"
		self.STANDARD_ATTENTION = "STANDARD_ATTENTION"		
		self.num_steps = num_steps
		self.num_steps_mention = num_steps_mention
		self.num_steps_label = num_steps_label	
		self.hidden_size = hidden_size
		self.num_word_dim = num_word_dim
		self.num_type_dim = num_type_dim
		self.num_mention_dim = num_mention_dim
		self.num_levels = num_levels
		self.num_types = num_types
		self.num_words = num_words	
		self.num_mention_words = num_mention_words
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.max_gradient_norm = max_gradient_norm
		self.sigmoid_activation = tf.nn.sigmoid
		self.softmax_activation = tf.nn.softmax
		if output_activation=="softmax":
			self.output_activation = self.softmax_activation
		else:
			self.output_activation = None	
		self.relu_activation = tf.nn.relu
		self.attention_type = attention_type
		self.num_types_per_level = [tf.constant(i) for i in num_types_per_level]
		self.types_per_level = [tf.constant(i) for i in types_per_level]
		def create_scopes():
			self.descs = tf.constant(descs, dtype=tf.float32, shape=[self.num_types, self.num_types], name='descs')
			max_val = np.sqrt(6. / (self.num_types + self.num_types))
			if self.attention_type ==self.CHANGE_OF_VARIABLES:
				self.attn_aggrW = tf.Variable(tf.random_uniform([self.num_types, self.num_types], -1.*max_val, max_val), name="attn_aggrW")	
			self.type_embedding = tf.get_variable('type_embedding',shape=[self.num_types, self.num_type_dim], initializer=tf.truncated_normal_initializer(0.0,1.0))
			max_val = np.sqrt(6. / (self.hidden_size + self.num_type_dim))
			self.type_context_scope = "type_context_scope"
        	        self.type_context_W = tf.Variable(tf.random_uniform([self.hidden_size, self.num_type_dim], -1.*max_val, max_val), name="W_type_context")
			self.type_context_attnW = tf.Variable(tf.random_uniform([self.hidden_size, self.num_type_dim], -1.*max_val, max_val), name="attnW_type_context")
			self.type_mention_scope = "type_mention_scope"
			self.type_mention_W = tf.Variable(tf.random_uniform([self.hidden_size, self.num_type_dim], -1.*max_val, max_val), name="W_type_mention")
			self.type_mention_attnW = tf.Variable(tf.random_uniform([self.hidden_size, self.num_type_dim], -1.*max_val, max_val), name="attnW_type_mention")
			self.enc_rnn_left_context = rnn_cell.EmbeddingWrapper(rnn_cell.GRUCell(self.hidden_size), self.num_words, self.num_word_dim)
			self.enc_scope_left_context = "encoder_left_context"
			self.enc_rnn_right_context = rnn_cell.EmbeddingWrapper(rnn_cell.GRUCell(self.hidden_size), self.num_words, self.num_word_dim)
			self.enc_scope_right_context = "encoder_right_context"
			self.enc_rnn_mention = rnn_cell.EmbeddingWrapper(rnn_cell.GRUCell(self.hidden_size), self.num_mention_words, self.num_mention_dim)
			self.enc_scope_mention = "encoder_mention"	
			self.dec_cells = rnn_cell.GRUCell(self.hidden_size)
			self.dec_scope = "decoder"
			max_val = np.sqrt(6. / (self.num_types + self.hidden_size))
                        self.dec_weights = tf.get_variable("dec_weights",[self.hidden_size, self.num_type_dim],initializer=tf.random_uniform_initializer(-1.*max_val,max_val))
                        self.dec_biases = tf.get_variable("dec_biases",[self.num_type_dim], initializer=tf.constant_initializer(0.0))	
		create_scopes()
		self.input_left_contexts = None
		self.input_right_contexts = None
		self.input_mentions = None
		self.decoder_label = None	
		self.decoder_input = None
		self.decoder_sequence_weight = None
		self.feed_previous = None
			
	def create_placeholder(self):
		self.input_left_contexts = [tf.placeholder(tf.int32,[None], name="input_left_contexts") for i in range(self.num_steps)]
		self.input_right_contexts = [tf.placeholder(tf.int32,[None], name="input_right_contexts") for i in range(self.num_steps)]
		self.input_mentions = [tf.placeholder(tf.int32,[None], name="input_mentions") for i in range(self.num_steps_mention)]
		self.decoder_label = [tf.placeholder(tf.int32,[None], name="decoder_label") for i in range(self.num_steps_label)]
		self.decoder_input = [tf.placeholder(tf.int32,[None], name="decoder_input") for i in range(self.num_steps_label)]
		self.decoder_sequence_weight = [tf.placeholder(tf.float32,[None], name="decoder_sequence_weight") for i in range(self.num_steps_label)]
		self.feed_previous = tf.placeholder(tf.bool, name='feed_previous')
	
	def sentence_encoder(self):
		with tf.variable_scope(self.enc_scope_left_context) as scope:
			sentence_left_outputs, sentence_left_states = rnn.rnn(self.enc_rnn_left_context, self.input_left_contexts, scope=scope, dtype=tf.float32)
		#sentence_left_states is a tensor of dimension (#batch_size * #hidden_size)
		with tf.variable_scope(self.enc_scope_right_context) as scope:
			sentence_right_outputs, sentence_right_states = rnn.rnn(self.enc_rnn_right_context, self.input_right_contexts, scope=scope, dtype=tf.float32)
		#sentence_right_states is a tensor of dimension (#batch_size * #hidden_size)
		with tf.variable_scope(self.enc_scope_mention) as scope:
			sentence_mention_outputs, sentence_mention_states = rnn.rnn(self.enc_rnn_mention, self.input_mentions, scope=scope, dtype=tf.float32)
		#sentence_mention_states is a tensor of dimension (#batch_size * #hidden_size)
		sentence_states = tf.concat(1,[sentence_left_states, sentence_right_states])	
		sentence_outputs = []
		sentence_outputs.extend(sentence_left_outputs)
		sentence_outputs.extend(sentence_right_outputs)
		#sentence_states is a tensor of dimension (#batch_size * 3*#hidden_size)
		return sentence_states, sentence_outputs, sentence_mention_states, sentence_mention_outputs
	
	def type_context_matmul(self, sentence_states):
		#after doing the matmul for every instance in the stack the output will be a tensor of dimension (#batch_size * #num_types_in_level)
		sentence_type_context_W = tf.matmul(sentence_states, self.type_context_W)
		#sentence_type_context_W is a list of #num_steps tensors of dimension (#batch_size * #num_type_dim)
		sentence_type_context_W_type_embedding = tf.matmul(sentence_type_context_W, self.type_embedding, transpose_b=True)
		#sentence_type_context_W_type_embedding is a list of #num_steps tensors of dimension (#batch_size * #num_types_in_level)
		return sentence_type_context_W_type_embedding
	
	def attn_on_types_given_context(self, sentence_states, types_in_level, num_types_in_level, num_heads=1):
		#sentence_states is a list of #num_steps dimension tensor 
		with tf.variable_scope(self.type_context_scope) as scope:
			#type_embedding_in_level = tf.gather(self.type_embedding, indices=types_in_level)
			sentence_states = tf.stack(sentence_states)
			#type_embedding_in_level = tf.tile(tf.expand_dims(type_embedding_in_level, axis=0), [2*self.num_steps,1,1])
                        sentence_type_attn = tf.map_fn(self.type_context_matmul, sentence_states)
			#sentence_type_attn is a list of #num_steps tensors of dimension (#batch_size * #num_types_in_level)
			sentence_type_attn = tf.concat(0, sentence_type_attn)
			#sentence_type_attn is a tensor of dimension (#num_steps * #batch_size * #num_types_in_level)
			sentence_type_attn = tf.nn.softmax(sentence_type_attn, dim=0)
			sentence_type_attn = tf.transpose(sentence_type_attn)
			sentence_type_attn = tf.gather(sentence_type_attn,indices=types_in_level)	
			#sentence_type_attn is a tensor of dimension (#num_types_in_level * #batch_size * #num_steps)
			sentence_type_attn = tf.expand_dims(sentence_type_attn, axis=3)
			#sentence_type_attn is a tensor of dimension (#num_types_in_level * #batch_size * #num_steps * 1)
			sentence_states = tf.transpose(sentence_states, perm=[1,0,2])
			sentence_states_replicated = tf.expand_dims(sentence_states, axis=0)
			#sentence_states_replicated is a tensor of dimension (1 * #batch_size * #num_steps * hidden_size)
			sentence_states_replicated = tf.tile(sentence_states_replicated, [num_types_in_level,1,1,1])
			#sentence_states_replicated is a tensor of dimension (#num_types_in_level * #batch_size * #num_steps * hidden_size)
			attn = tf.multiply(sentence_type_attn,  sentence_states_replicated)
			attn = tf.reduce_sum(attn, 2)
			#attn is a tensor of dimension (#num_types_in_level * #batch_size * #hidden_size)
			attn = tf.transpose(attn, perm=[1,0,2])
			#attn is a tensor of dimension (#batch_size * #num_types_in_level * #hidden_size)
			attn = self.softmax_activation(attn, dim=1)
			#we want softmax to be over the #num_types_in_level 
			attn.set_shape([None, attn.get_shape()[1].value, attn.get_shape()[2].value])	
                return attn
	
	def type_mention_matmul(self, sentence_mention_states):
                #after doing the matmul for every instance in the stack the output will be a tensor of dimension (#batch_size * #num_types_in_level)
                sentence_type_mention_W = tf.matmul(sentence_mention_states, self.type_mention_W)
                #sentence_type_mention_W is a list of #num_steps_mention tensors of dimension (#batch_size * #num_type_dim)
                sentence_type_mention_W_type_embedding = tf.matmul(sentence_type_mention_W, self.type_embedding, transpose_b=True)
                #sentence_type_mention_W_type_embedding is a list of #num_steps_mention tensors of dimension (#batch_size * #num_types_in_level)
                return sentence_type_mention_W_type_embedding

        def attn_on_types_given_mention(self, sentence_mention_states, types_in_level, num_types_in_level, num_heads=1):
                #sentence_mention_states is a list of #num_steps_mention dimension tensor 
                with tf.variable_scope(self.type_mention_scope) as scope:
                        #type_embedding_in_level = tf.gather(self.type_embedding, indices=types_in_level)	
                        sentence_mention_states = tf.stack(sentence_mention_states)
			#type_embedding_in_level = tf.tile(tf.expand_dims(type_embedding_in_level, axis=0), [self.num_steps_mention,1,1])
                        #print type_embedding_in_level
                        sentence_type_attn = tf.map_fn(self.type_mention_matmul, sentence_mention_states)
                        #sentence_type_attn is a list of #num_states_mention tensors of dimension (#batch_size * #num_types_in_level)
                        sentence_type_attn = tf.concat(0, sentence_type_attn)
                        #sentence_type_attn is a tensor of dimension (#num_states_mention * #batch_size * #num_types_in_level)
                        sentence_type_attn = tf.nn.softmax(sentence_type_attn, dim=0)
                        sentence_type_attn = tf.transpose(sentence_type_attn)
			sentence_type_attn = tf.gather(sentence_type_attn, indices=types_in_level)
                        #sentence_type_attn is a tensor of dimension (#num_types_in_level * #batch_size * #num_steps_mention)
                        sentence_type_attn = tf.expand_dims(sentence_type_attn, axis=3)
                        #sentence_type_attn is a tensor of dimension (#num_types_in_level * #batch_size * #num_steps_mention * 1)
                        sentence_mention_states = tf.transpose(sentence_mention_states, perm=[1,0,2])
                        sentence_states_replicated = tf.expand_dims(sentence_mention_states, axis=0)
                        #sentence_states_replicated is a tensor of dimension (1 * #batch_size * #num_steps_mention * hidden_size)
                        sentence_states_replicated = tf.tile(sentence_states_replicated, [num_types_in_level,1,1,1])
                        #sentence_states_replicated is a tensor of dimension (#num_types_in_level * #batch_size * #num_steps_mention * hidden_size)
                        attn = tf.multiply(sentence_type_attn,  sentence_states_replicated)
                        attn = tf.reduce_sum(attn, 2)
                        #attn is a tensor of dimension (#num_types_in_level * #batch_size * #hidden_size)
                        attn = tf.transpose(attn, perm=[1,0,2])
                        #attn is a tensor of dimension (#batch_size * #num_types_in_level * #hidden_size)
                        attn = self.softmax_activation(attn, dim=1)
			#we want the softmax to be over #num_types_in_level
			attn.set_shape([None, attn.get_shape()[1].value, attn.get_shape()[2].value])
                return attn		
	
	def diag(self, matrix):
		diagonal = tf.diag_part(matrix)
		return diagonal

	def attention_aggregation(self, attn_t):
		attn_t = attn_t + tf.matmul(attn_t, tf.multiply(self.attn_aggrW, self.descs))
		return attn_t
	

	def attention_aggregation_max(self, attn_t):
		attn_t_t = tf.expand_dims(attn_t, axis=1)
		attn_t_t = tf.tile(attn_t_t, [1,self.num_types,1])
		attn_t = attn_t + tf.reduce_max(tf.multiply(attn_t_t,  tf.multiply(self.attn_aggrW, self.descs)), axis=2)
		return attn_t

	def decode(self, concatenated_input, loop_fn, init_state, sentence_states, sentence_outputs, sentence_mention_states, sentence_mention_outputs, dec_scope, attn_len, attn_size, attn_t_given_context_per_level, attn_t_given_mention_per_level, initial_state_attention=False):
		state = init_state
		outputs = []
		prev = None
		output_size = self.dec_cells.output_size
		def attention(query, attn_t_given_context_per_level, attn_t_given_mention_per_level):
			with variable_scope.variable_scope("Attention"):
				hidden_t = None
				overall_hidden_t = None
				for i in range(self.num_levels):
					types_per_level_i = self.types_per_level[i]
					attn_t_given_context = attn_t_given_context_per_level[i]	
					attn_t_given_mention = attn_t_given_mention_per_level[i]
					type_embedding_in_level = tf.gather(self.type_embedding, indices=types_per_level_i)			
					y = tf.tile(array_ops.reshape(query, [-1, 1, self.hidden_size]), [1, self.num_types_per_level[i], 1])
					#y is of dimension batch_size * num_types_in_level * hidden_size
					#attn_t_given_context is of dimension batch_size * num_types_in_level * hidden_size
					#attn_t_given_mention is of dimension batch_size * num_types_in_level * hidden_size
					#type_embedding is of dimension num_types_in_level * num_type_dim
					#type_context_attnW is of dimension hidden_size * num_type_dim
					#type_mention_attnW is of dimension hidden_size * num_type_dim
					attn_t_given_context = tf.multiply(attn_t_given_context, y)
					attn_t_given_mention = tf.multiply(attn_t_given_mention, y)
					#attn_t_given_context is of dimension batch_size * num_types_in_level * hidden_size
	                                #attn_t_given_mention is of dimension batch_size * num_types_in_level * hidden_size
					if hidden_t is None:		
						type_context_attn = tf.matmul(type_embedding_in_level, self.type_context_attnW, transpose_b=True)
						type_mention_attn = tf.matmul(type_embedding_in_level, self.type_mention_attnW, transpose_b=True)
						type_context_attn = tf.tile(tf.expand_dims(type_context_attn, axis=0), [self.batch_size,1,1])
						type_mention_attn = tf.tile(tf.expand_dims(type_mention_attn, axis=0), [self.batch_size,1,1])
					else:
						hidden_t = tf.tile(tf.expand_dims(hidden_t, axis=1), [1,self.num_types_per_level[i],1])
						#hidden_t is of dimension (batch_size * num_types_in_level * num_type_dim)
						hidden_type_embedding_in_level = tf.multiply(hidden_t, type_embedding_in_level)
						#type_embedding_in_level is of dimension (batch_size * num_types_in_level * num_type_dim)
						#self.type_context_attnW is of dimension (hidden_size * num_type_dim)
						type_context_attnW = tf.tile(tf.expand_dims(self.type_context_attnW, axis=0), [self.batch_size, 1, 1])
						type_mention_attnW = tf.tile(tf.expand_dims(self.type_mention_attnW, axis=0), [self.batch_size, 1, 1])
						type_context_attn = tf.matmul(hidden_type_embedding_in_level, type_context_attnW, transpose_b=True)
						type_mention_attn = tf.matmul(hidden_type_embedding_in_level, type_mention_attnW, transpose_b=True)
					#type_context_attn is of dimension batch_size * num_types_in_level * hidden_size
					#type_mention_attn is of dimension batch_size * num_types_in_level * hidden_size
					attn_t_given_context = tf.reduce_sum(tf.multiply(attn_t_given_context, type_context_attn), 2)
					attn_t_given_mention = tf.reduce_sum(tf.multiply(attn_t_given_mention, type_mention_attn), 2)
					#attn_t_given_context is of dimension batch_size * num_types_in_level
					#attn_t_given_mention is of dimension batch_size * num_types_in_level
					attn_t = attn_t_given_context + attn_t_given_mention
					attn_t = nn_ops.softmax(attn_t)
					if self.attention_type==self.CHANGE_OF_VARIABLES:
						attn_t = self.attention_aggregation(attn_t)
					#attn_t is of dimension batch_size * num_types_in_level
					hidden_t = tf.matmul(attn_t, type_embedding_in_level)
					#hidden_t is of dimension batch_size * num_type_dim
					if overall_hidden_t is None:
						overall_hidden_t = hidden_t
					else:	
						overall_hidden_t = overall_hidden_t + hidden_t	
					#hidden_t is of dimension batch_size * self.num_type_dim
			return overall_hidden_t	

		batch_attn_size = array_ops.pack([self.batch_size, attn_size])
		attns = array_ops.zeros(batch_attn_size, dtype=tf.float32)
		attns.set_shape([None, attn_size])
		#atns is a tensor of dimension batch_size * self.hidden_size
		if initial_state_attention:
			#initial_state is a tensor of dimension batch_size * self.hidden_size
			attns = attention(initial_state)
		for i,inp in enumerate(concatenated_input):
			if loop_fn is not None and prev is not None:
				with tf.variable_scope("loop_function", reuse=True):
					inp = loop_fn(prev, i)
					inp = tf.concat(1, [sentence_states, sentence_mention_states, inp])
			if i>0:
				dec_scope.reuse_variables()
			input_size = inp.get_shape().with_rank(2)[1]
			x = linear([inp]+[attns], input_size, True)
			output, state = self.dec_cells(x, state, scope=dec_scope)
			if i==0 and initial_state_attention:	
				with tf.variable_scope(dec_scope, reuse=True):
					attns = attention(state, attn_t_given_context_per_level, attn_t_given_mention_per_level)
			else:
				attns = attention(state, attn_t_given_context_per_level, attn_t_given_mention_per_level)
			with tf.variable_scope("AttnOutputProjection"):
				output = linear([output]+[attns], output_size, True)
			outputs.append(output)
			if loop_fn is not None:
				prev = output
		return outputs, state						


	def decoder(self, sentence_states, sentence_outputs, sentence_mention_states, sentence_mention_outputs, initial_state_attention=False):
		#self.decoder_input is of dimension max_len * batch_size
		
		with tf.variable_scope(self.dec_scope) as scope:
			attn_t_given_context_per_level = []
			attn_t_given_mention_per_level = []	
			init_state = self.dec_cells.zero_state(self.batch_size, tf.float32)
			attn_len = self.type_embedding.get_shape()[0].value
			attn_size = self.type_embedding.get_shape()[1].value
			for i in range(self.num_levels):
				attn_t_given_context = self.attn_on_types_given_context(sentence_outputs, self.types_per_level[i], self.num_types_per_level[i])
				attn_t_given_mention = self.attn_on_types_given_mention(sentence_mention_outputs, self.types_per_level[i], self.num_types_per_level[i])
				attn_t_given_context_per_level.append(attn_t_given_context)
				attn_t_given_mention_per_level.append(attn_t_given_mention)
			def feed_previous_decode(feed_previous_bool):
				weights = tf.matmul(self.dec_weights, self.type_embedding, transpose_b=True)
				biases = tf.matmul(tf.expand_dims(self.dec_biases,0), self.type_embedding, transpose_b=True)
				biases = array_ops.reshape(biases,[self.num_types])
				dec_embed, loop_fn = seq2seq.get_decoder_embedding(self.decoder_input, self.num_types, self.num_type_dim, output_projection=(weights, biases), feed_previous=feed_previous_bool)
				#dec_embed is of dimension self.max_steps_label * batch_size * self.num_type_dim
				concatenated_input = self.dec_concat_ip(dec_embed, sentence_states, sentence_mention_states)
				#concat_dec_inputs is a list of self.max_steps_label tensors of dimension batch_size * (self.num_type_dim + self.hidden_size)				
				dec_output, _ = self.decode(concatenated_input, loop_fn, init_state, sentence_states, sentence_outputs, sentence_mention_states, sentence_mention_outputs, scope, attn_len, attn_size, attn_t_given_context_per_level, attn_t_given_mention_per_level, initial_state_attention)
				return dec_output
		
			weights = tf.matmul(self.dec_weights, self.type_embedding, transpose_b=True)
			biases = tf.matmul(tf.expand_dims(self.dec_biases,0), self.type_embedding, transpose_b=True)
			biases = array_ops.reshape(biases,[self.num_types])	
			dec_output = control_flow_ops.cond(self.feed_previous, lambda: feed_previous_decode(True), lambda: feed_previous_decode(False))
			for i in range(len(dec_output)):
				dec_output[i] = tf.matmul(dec_output[i], weights)+biases
				if self.output_activation is not None:
					dec_output[i] = self.output_activation(dec_output[i])
		return dec_output				
			
	def dec_concat_ip(self, dec_embed, sentence_states, sentence_mention_states):
		concat_dec_inputs = []
		for (i, inp) in enumerate(dec_embed):
			concat_dec_inputs.append(tf.concat(1, [sentence_states, sentence_mention_states, inp]))
			#inp is of dimension batch_size * self.num_type_dim
			#sentence_states is of dimension batch_size * self.hidden_size
			#concat_dec_inputs[i] is of dimension batch_size * (self.num_type_dim + self.hidden_size)
		#concat_dec_inputs is a list of self.max_steps_label tensors of dimension batch_size * (self.num_type_dim + self.hidden_size)
		return concat_dec_inputs
			
	def loss(self, logits):
		#self.decoder_label is a tensor of dimension (self.num_steps_label, batch_size)
		losses = seq2seq.sequence_loss_by_example(logits, self.decoder_label, self.decoder_sequence_weight)
		return losses
	
	def inference(self):
		sentence_states, sentence_outputs, sentence_mention_states, sentence_mention_outputs = self.sentence_encoder()
		predicted = self.decoder(sentence_states, sentence_outputs, sentence_mention_states, sentence_mention_outputs)
		return predicted
	
	def train(self, losses):
		parameters=tf.trainable_variables()
	        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
        	gradients=tf.gradients(losses,parameters)
	        #print tf.get_default_graph().as_graph_def()
        	clipped_gradients,norm=tf.clip_by_global_norm(gradients,self.max_gradient_norm)
	        global_step=tf.Variable(0,name="global_step",trainable='False')
        	#train_op=optimizer.minimize(losses,global_step=global_step)
	        train_op=optimizer.apply_gradients(zip(clipped_gradients,parameters),global_step=global_step)
        	return train_op, clipped_gradients

