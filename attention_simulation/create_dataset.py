import json
import sys
from scipy import random as scipyrand
from scipy import linalg
import random
import numpy as np
import math
from sklearn.mixture import GaussianMixture
import cPickle as pkl

class Node():
	#each object of type node has a value and a set of node type objects as its immediate children
    	def __init__(self, value, children):
        	self.value = value
	        self.children = children

	def printnode(self, children_dict):
		if len(self.children)==0:
	            return children_dict
		print self.value, '->',
		if self.value not in children_dict:
			children_dict[self.value] = set([])
		for child in self.children:
			if child is not None:
				print child.value,
				children_dict[self.value].add(child.value)
		print ''
		for child in self.children:
			if child is not None:
				children_dict=child.printnode(children_dict)
		return children_dict

	def get_all_paths(self):
		#will return all paths from the self node as root; note paths originate at the root and go all the way till the leaf	
		paths=self.value
		if len(self.children)==0:
                    return [paths]
		all_paths_child = []
		for child in self.children:
			if child is not None:
				paths_child = child.get_all_paths()
				for i in paths_child:
					if isinstance(i, list):
						i[0:0] = [paths]
					else:
						paths_child[0:0]=[paths]
						all_paths_child.append(paths_child)
						break
					all_paths_child.append(i)
		return all_paths_child
				 
	def get_desc(self, descs):
		if self.value not in descs:
			descs[self.value] = set([])
		descs[self.value].add(self.value)
		for child in self.children:
			if child is not None:
				descs[self.value].add(child.value)	
				descs = child.get_desc(descs)
				descs[self.value].update(descs[child.value])
		return descs
ary=2

def heapify(l, heap):
	#create a heap, with the ary given as input currently hardcoded to 2 in the above line
	len_l = len(l)
	if len_l==0:
		return None	
	root = l[0]
	if len_l==1:
		node = Node(root, [])
		if root not in heap:
			heap[root]= node
		return node
	partition = int(math.pow(ary,int(math.log(len_l-1)/math.log(ary))))
	check_l = math.log(len_l)/math.log(ary)
	if int(check_l)==check_l:
		partition = partition+1
	num_children = math.ceil((float(len_l))/(float(partition)))
	i=partition
	children = []
	last_i=1
	while last_i<=len_l:
		child = l[last_i:min(len_l,i)]
		children.append(heapify(child, heap))
		last_i=i
		i=i+partition
	print root, ':: ->',[x.value for x in children if x is not None]
	node = Node(root, children)
	if root not in heap:
		heap[root] = node
	return node
	
def sample_n(n):
	rand = set([])
	while len(rand)<n:
		rand.add(random.random())
	return list(rand)

def sample_mean_vector(dim):
	return sample_n(dim)

def sample_cov_matrix(dim):
	A = np.matrix(scipyrand.rand(dim,dim))
	cov = np.dot(A,A.transpose())
	return cov	

def sample_params(nodes, dimension):
	num_nodes = len(nodes)
	mu_i = [sample_mean_vector(dimension) for i in range(num_nodes)]
	sigma_i = [sample_cov_matrix(dimension) for i in range(num_nodes)]
	params = {}
	for i in nodes:
		dist = np.random.multivariate_normal(mu_i[i], sigma_i[i], 3)
		params[i] = [mu_i[i],sigma_i[i], dist]
	return params

def create_dataset(nodes, paths, params, num_data, num_contexts, num_words):
	#this is the main function where the dataset gets created. Currently it is a simplistic setting where the following steps are observed	
	num_paths = len(paths)
	num_nodes = len(nodes)
	dataset = []
	for i in range(num_data):
		#first sample a path from all the paths in the given heap
		sample_path = paths[random.sample(xrange(num_paths),1)[0]]
		#then sample weights for each node in the path
		sample_attn_weights = sample_n(len(sample_path))
		#then normalize the weights (because they would be treated as explicit supervision of attention weights which are themselves probabilities
		sample_attn_weights = normalize(sample_attn_weights)
		#then sort the attention weights, (because we want to simulate the effect that attention weights on the higher-up nodes is higher than on its children (reverse=True) since path is root to leaf)
		sample_attn_weights.sort(reverse=True)
		if len(sample_attn_weights)!=len(sample_path):
			raise Exception('len(sample_attn_weights)!=len(sample_path)')
		gaussians = None
		sample_attn_weights_arr = [0.]*num_nodes
		for index,path_nodeid in enumerate(sample_path):
			sample_attn_weights_arr[path_nodeid] = sample_attn_weights[index]
			if gaussians is None:
				gaussians = params[path_nodeid][2]
			else:
				gaussians = np.concatenate((gaussians, params[path_nodeid][2]), axis=0)
		#create a gaussian mixture model with the nodes in the path as the component gaussians and the attention weights as the mixture weights
		gmm = GaussianMixture(n_components=len(sample_path), covariance_type="full",tol=0.001, weights_init=sample_attn_weights)
		gmm = gmm.fit(X=gaussians)
		xs = []
		ys = []
		for j in range(num_contexts):
			x,y = gmm.sample(num_words)		 	
			xs.append(x)
			ys.append(y)
		#sample n different context words from that gaussian mixture model	
		dataset.append([xs, sample_attn_weights_arr, ys])
		if i%100==0:
			print 'created ',i, ' data points'
	print 'length of dataset ', len(dataset)	
	return dataset

def normalize(x):
	x = [math.exp(x_i) for x_i in x]
	sum_x = sum(x)
	x = [x_i/sum_x for x_i in x]
	return x

def simulate_data():
	dataset_params = json.load(open(sys.argv[1]))
	num_types = dataset_params['num_types']
	num_instances = dataset_params['num_instances']
	num_test_instances = int(0.15*num_instances)
	num_valid_instances = int(0.15*num_instances)
	num_training_instances = num_instances - num_test_instances - num_valid_instances
	num_word_dim = dataset_params['num_word_dim']
	num_type_dim = dataset_params['num_type_dim']
	if num_word_dim!=num_type_dim:
		print 'currently words are sampled from GMM over types, so word dimension and type dimension are the same'	
		sys.exit(1)
	num_contexts = dataset_params['num_contexts']
	num_words = dataset_params['num_words']
    	myfloat = np.float64
    	l = range(num_types)
	heap={}	
	root = heapify(l, heap)
	print 'Printing Immediate Children'
	children_dict = {}
	children_dict = root.printnode(children_dict)
	print 'Printing Descendants'
	descs = {}
	descs = root.get_desc(descs)
	print 'got descendants'
	#for k,v in descs.items():
        #	print k, '::', v
	paths = []
    	paths = root.get_all_paths()    	
	print 'got all paths'
	for path in paths:
		print path
	sys.exit(1)	
	nodes = descs.keys()
    	params_per_node = sample_params(nodes, num_type_dim)
	print 'sampled parameters'	
    	training_dataset = create_dataset(nodes, paths, params_per_node, num_training_instances, num_contexts, num_words)
	test_dataset = create_dataset(nodes, paths, params_per_node, num_test_instances, num_contexts, num_words)
	valid_dataset = create_dataset(nodes, paths, params_per_node, num_valid_instances, num_contexts, num_words)
	pkl.dump(children_dict, open('data/children.pkl','w'))
	pkl.dump(descs, open('data/descendants.pkl','w'))
	pkl.dump(training_dataset, open('data/train_dataset.pkl','w'))
	pkl.dump(valid_dataset, open('data/valid_dataset.pkl','w'))
	pkl.dump(test_dataset, open('data/test_dataset.pkl','w'))
	

if __name__ == "__main__":
        simulate_data()
