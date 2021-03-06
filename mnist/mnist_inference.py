import tensorflow as tf

# define parameters of network
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularize):
	weights = tf.get_variable("weights", shape,
														initializer=tf.truncated_normal_initializer(stddev=0.1))
	
	if regularize != None:
		tf.add_to_collection('losses', regularize(weights))
	return weights
		

def inference(input_tensors, regularize):
	with tf.variable_scope('layer1'):
	  weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularize)
	  biases = tf.get_variable("biases", [LAYER1_NODE], 
	  													initializer=tf.constant_initializer(0.0))
	  
	  layer1 = tf.nn.relu(tf.matmul(input_tensors, weights) + biases)
	  
	
	with tf.variable_scope('layer2'):
		weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularize)
		biases = tf.get_variable("biases", [OUTPUT_NODE],
															initializer=tf.constant_initializer(0.0))
															
		layer2 = tf.matmul(layer1, weights) + biases
		
	return layer2
	

