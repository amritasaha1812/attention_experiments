import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn,rnn_cell
import math
import os
import sys
from tensorflow.python.ops import control_flow_ops

class TypeAttention:
	CHANGE_OF_VARIABLES = "CHANGE_OF_VARIABLES"
        STRUCTURED_ATTENTION = "STRUCTURED_ATTENTION"
        STANDARD_ATTENTION = "STANDARD_ATTENTION"

	def __init__(self, num_steps, num_contexts, hidden_size, num_word_dim, num_type_dim, num_types, learning_rate, max_gradient_norm, descs, attention_type):
		self.CHANGE_OF_VARIABLES = "CHANGE_OF_VARIABLES"
		self.STRUCTURED_ATTENTION = "STRUCTURED_ATTENTION"
		self.STANDARD_ATTENTION = "STANDARD_ATTENTION"
		self.num_steps = num_steps
		self.num_contexts = num_contexts
		self.hidden_size = hidden_size
		self.num_word_dim = num_word_dim
		self.num_type_dim = num_type_dim
		self.num_types = num_types
		self.learning_rate = learning_rate
		self.max_gradient_norm = max_gradient_norm
		self.sigmoid_activation = tf.nn.sigmoid
		self.softmax_activation = tf.nn.softmax
		self.relu_activation = tf.nn.relu
		self.attention_type = attention_type
		def create_scopes():
			max_val = np.sqrt(6. / (self.num_types + self.num_type_dim))
			self.type_embedding = tf.get_variable('type_embedding',shape=[self.num_types, self.num_type_dim], initializer=tf.truncated_normal_initializer(0.0,1.0)) 
			self.type_context_scope = "type_context_scope"
			max_val = np.sqrt(6. / (self.hidden_size + self.num_type_dim))
        	        self.type_context_W = tf.Variable(tf.random_uniform([self.hidden_size, self.num_type_dim], -1.*max_val, max_val), name="W_type_context")
			self.enc_rnn_sent = rnn_cell.GRUCell(self.hidden_size)
			self.enc_scope_sent = "encoder_sent"
			max_val = np.sqrt(6. / (self.num_types + self.hidden_size))
                        self.attn_dec_W = tf.Variable(tf.random_uniform([self.num_types, self.hidden_size], -1.*max_val, max_val), name="W_attn_dec")
			self.attn_dec_W = tf.expand_dims(self.attn_dec_W, axis=0)
			self.attn_dec_b = tf.Variable(tf.constant(0., shape=[self.num_types]), name="b_attn_dec")
			self.descs = tf.get_variable('descs', initializer=descs)
		create_scopes()
		self.input_sent_embedding = None
		self.input_attn = None	
			
	def create_placeholder(self):
		self.input_sent_embedding = [[tf.placeholder(tf.float32,[None, self.num_word_dim], name="input_sent_embedding") for i in range(self.num_steps)] for j in range(self.num_contexts)]
		self.input_attn = tf.placeholder(dtype=tf.float32, shape=[None, self.num_types], name="input_attn")

	def sentence_encoder(self):
		enc_inputs = self.input_sent_embedding
		sentence_states = []
		with tf.variable_scope(self.enc_scope_sent) as scope:
			for i in range(0, len(enc_inputs)):
				if i>0:
					scope.reuse_variables()
				_, states = rnn.rnn(self.enc_rnn_sent, enc_inputs[i], scope=scope, dtype=tf.float32)
				sentence_states.append(states)
		#sentence_states is a list of #num_contexts tensors of dimension (#batch_size * #hidden_size)
		return sentence_states
	
	def type_context_matmul(self, sentence_states):
		#after doing the matmul for every instance in the stack the output will be a tensor of dimension (#batch_size * #num_types)
		sentence_type_context_W = tf.matmul(sentence_states, self.type_context_W)
		#sentence_type_context_W is a list of #num_contexts tensors of dimension (#batch_size * #num_type_dim)
		sentence_type_context_W_type_embedding = tf.matmul(sentence_type_context_W, self.type_embedding, transpose_b=True)
		#sentence_type_context_W_type_embedding is a list of #num_contexts tensors of dimension (#batch_size * #num_types)
		return sentence_type_context_W_type_embedding
	
	def attn_on_types(self, sentence_states, num_heads=1):
		with tf.variable_scope(self.type_context_scope) as scope:
			sentence_states = tf.stack(sentence_states)	
			sentence_type_attn = tf.map_fn(self.type_context_matmul, sentence_states)
			#sentence_type_attn is a list of #num_context tensors of dimension (#batch_size * #num_types)
			sentence_type_attn = tf.concat(0, sentence_type_attn)
			#sentence_type_attn is a tensor of dimension (#num_context * #batch_size * #num_types)
			sentence_type_attn = tf.nn.softmax(sentence_type_attn, dim=0)
			sentence_type_attn = tf.transpose(sentence_type_attn)
			#sentence_type_attn is a tensor of dimension (#num_types * #batch_size * #num_context)
			sentence_type_attn = tf.expand_dims(sentence_type_attn, axis=3)
			#sentence_type_attn is a tensor of dimension (#num_types * #batch_size * #num_context * 1)
			sentence_states = tf.transpose(sentence_states, perm=[1,0,2])
			sentence_states_replicated = tf.expand_dims(sentence_states, axis=0)
			#sentence_states_replicated is a tensor of dimension (1 * #batch_size * #num_context * hidden_size)
			sentence_states_replicated = tf.tile(sentence_states_replicated, [self.num_types,1,1,1])
			#sentence_states_replicated is a tensor of dimension (#num_types * #batch_size * #num_context * hidden_size)
			attn = tf.multiply(sentence_type_attn,  sentence_states_replicated)
			attn = tf.reduce_sum(attn, 2)
			#attn is a tensor of dimension (#num_types * #batch_size * #hidden_size)
			attn = tf.transpose(attn, perm=[1,0,2])
			#attn is a tensor of dimension (#batch_size * #num_types * #hidden_size)
		attn = tf.reduce_sum(tf.multiply(attn, self.attn_dec_W),2)
		attn = attn + self.attn_dec_b
                #attn is a tensor of dimension (#batch_size * #num_types)
		if self.attention_type==self.CHANGE_OF_VARIABLES:
			attn_matrix = tf.expand_dims(attn, axis=1)
			#attn_matrix is a tensor of dimension (#batch_size * 1 * #num_types)
			attn_matrix = tf.tile(attn_matrix, [1,self.num_types,1])
			#attn_matrix is a tensor of dimension (#batch_size * #num_types * #num_types)
			attn_matrix = tf.multiply(attn_matrix, self.descs)
			#attn_matrix is a tensor of dimension (#batch_size * #num_types * #num_types)	
			#V2:attn_matrix = self.softmax_activation(attn_matrix)
			#V2:attn = attn + tf.reduce_sum(attn_matrix, 2)
			#V2:#attn is a tensor of dimension (#batch_size * #num_types) 
			attn_matrix = self.softmax_activation(attn_matrix)
			attn = attn + tf.reduce_max(attn_matrix, axis=2)
			#attn is a tensor of dimension (#batch_size * #num_types)
		attn = self.softmax_activation(attn, dim=1)
                return attn
	
	def loss(self, pred_attn):
		l = tf.reduce_mean(tf.squared_difference(pred_attn, self.input_attn))
		return l, tf.reduce_mean(pred_attn), tf.reduce_mean(self.input_attn)
			
	def inference(self):
		sentence_states = self.sentence_encoder()
		pred_attn = self.attn_on_types(sentence_states)
		return pred_attn	
	
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

