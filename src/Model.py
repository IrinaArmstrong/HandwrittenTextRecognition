from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os

class ModelFilePaths:
	""" Filenames and paths to data. """
	modelDir = '../model/model'  # dir, where to store saved model file (.zip)
	dumpDir = '../dump/'  # dir, where to dump(save) the output of the RNN in CSV file(s)
	snapshotsDir = '../model/snapshot'  # file, where to save the NN configuration during training


class DecoderType:
	""" Types of CTC search technique """
	BestPath = 0
	BeamSearch = 1
	# WordBeamSearch = 2


class Model: 
	""" The simpliest TF model for HTR.
	 	Architecture: TODO: Enter here model.sunmmary()
	"""

	# Model constants
	batchSize = 50
	imgSize = (128, 32)# for TF =(w, h)
	maxTextLen = 32

	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, dump=False):
		""" Configure NN structure: add CNN, RNN and CTC layers, set params for training and make session.
			Arguments:
				charList - symbols of dictionary (e.t. dictionary, numbers, punctuation marks),
				decoderType- chosen type of CTC search technique, by default = BestPath,
				mustRestore - True, if model must be restored (for inference), otherwise False,
				dump - True, if write output of NN to CSV file, otherwise False.
			Return:

		"""
		self.dump = dump
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0

		# Whether to use normalization over a batch or a population
		self.is_train = tf.placeholder(tf.bool, name='is_train')

		# Input image batch: [batchSize, W=128, H=32]
		self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

		# Setup CNN, RNN and CTC
		self.setupCNN()
		self.setupRNN()
		self.setupCTC()

		# Setup optimizer to train NN
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[]) # variable for list of learning rates
		# get_collection - Returns a list of values in the collection with the given name.
		# GraphKeys class contains many standard names for collections.
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # UPDATE_OPS - ???

		# control_dependencies - Returns a context manager that specifies control dependencies.
		# So, only when self.update_ops have been executed ops under 'with' will be executed
		with tf.control_dependencies(self.update_ops):
			# minimize - combines calls compute_gradients() and apply_gradients().
			# loss - CTC loss function for batch
			# compute_gradients() - Compute gradients of loss for the variable. It returns a list of (gradient, variable) pairs
			# apply_gradients() - Apply gradients to variables.  It returns an Operation that applies gradients.
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

			# TODO: Change RMSPropOptimizer to Adam 
			
		# initialize TF
		(self.sess, self.saver) = self.setupTF()

			
	def setupCNN(self):
		""" Create CNN layers and return output of these layers.
		 	Architecture * 5:
		 		- Convolutional layer with 'Same' padding
		 		- Batch normalization layer,
		 		- ReLu finction layer,
		 		- MaxPooling layer with 'Valid' padding.
		 	"""
		# expand dimensions of image [?, W, H] -> [?, W, H, 1]
		cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

		# list of parameters for the layers
		kernelVals = [5, 5, 3, 3, 3] # kernel sizes
		featureVals = [1, 32, 64, 128, 128, 256] # number of kernels in each layer
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)] # for max-pooling layer
		numLayers = len(strideVals)

		# create layers
		pool = cnnIn4d # input to first CNN layer [batchSize, W, H, 1]
		for i in range(numLayers):
			# init kernel with random numbers from a truncated normal distribution, 
			# shape=[kernelWidth, kernelHeight, kernelDepth, numOfKernels]
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			# Normalizes a tensor by mean and variance
			# training - Whether to return the output in training mode (normalized with statistics of the current batch) 
			# or in inference mode (normalized with moving statistics).
			conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
			relu = tf.nn.relu(conv_norm)
			# (2)ksize: list of ints that has length 1, 2 or 4. The size of the window for each dimension of the input tensor.
			# (3)strides: list of ints that has length 1, 2 or 4. The stride of the sliding window for each dimension of the input tensor.
			# (4)padding: A string, either 'VALID' or 'SAME'. The padding algorithm.
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
		# Returns: The max pooled output tensor, size = [batchSize, W=32, H=1, D=256]
		self.cnnOut4d = pool


	def setupRNN(self):
		""" Create Bidirectional RNN (LSTM) layers and return output of these layers.
			Architecture:
				- LSTM layer with 256 units, * 2 (stacked on each other for BLSTM),
				- Covered by dynamic RNN.
		"""
		# squeeze - Removes dimensions of size 1 from the shape of a tensor.
		# From CNN output [batchSize, W=32, H=1, D=256] -> [batchSize, 32, 256]
		rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

		# basic cells which is used to build RNN
		numHidden = 256
		# Create list of 2 LSTM recurrent network simple cells
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		# Stack basic cells into one MultiRNNCell
		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		# Creates a dynamic version of bidirectional recurrent neural network
		# Returns a tuple (output_state_fw, output_state_bw) containing the forward and the backward final states of bidirectional rnn.
		# BxTxF -> BxTx2H
		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
									
		# BatchxTimexH + BxTxH -> BxTx2H -> BxTx1X2H
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
									
		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		# Atrous convolution (a.k.a. convolution with holes or dilated convolution) 
		# Returns [batch, height, width, num_classes].
		# Output as a matrix of size [batchSize, 32, 80]
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def setupCTC(self):
		"""Create CTC loss and decoder and return them. """

		# BxTxC -> TxBxC = [32, batchSize, 80]
		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
		# Give ground truth text as sparse tensor
		# Sparse tensor as three separate dense tensors: indices [?, 2], values [?], and dense_shape [?]
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

		# Computes the CTC (Connectionist Temporal Classification) loss for batch
		# ctc_loss: labels - must take on values in [0, num_labels), must be a SparseTensor type,
		# inputs - a Tensor shaped: [max_time, batch_size, num_classes],
		# sequence_length - vector, size [batch_size],
		# Returns a 1-D float Tensor, size [batch], containing the negative log probabilities.
		self.seqLen = tf.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

		# Calc loss for each element to compute label probability
		# shape = [32, ?, 80]
		self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

		# Decoder selected: either best path decoding or beam search decoding
		if self.decoderType == DecoderType.BestPath:
			# Performs greedy decoding on the logits given in input (best path).
			# Returns: A tuple (decoded, neg_sum_logits)
			self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
		elif self.decoderType == DecoderType.BeamSearch:
			# Performs beam search decoding on the logits given in input.
			# Returns: A tuple (decoded, log_probabilities)
			self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)

			# TODO: Use WordBeamSearch or not???
		# elif self.decoderType == DecoderType.WordBeamSearch:
		# 	# import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
		# 	word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')
		# 	# prepare information about language (dictionary, characters in dataset, characters forming words)
		# 	chars = str().join(self.charList)
		# 	wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
		# 	corpus = open('../data/corpus.txt').read()
		# 	# decode using the "Words" mode of word beam search
		# 	self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))


	def setupTF(self):
		""" Initialize TF.
		 	Return:
		 		(sess, saver) list:
		 		sess - created TF session,
		 		saver - tf.train.Saver object, that saves model to file.
		"""
		print('Python: ' + sys.version)
		print('Tensorflow: ' + tf.__version__)

		sess = tf.Session() # create TF session
		# Saver - saves model to file. 
		# max_to_keep - indicates the maximum number of recent checkpoint files to keep. As new files are created, older files are deleted.
		saver = tf.train.Saver(max_to_keep=1)

		# latest_checkpoint - Finds the filename of latest saved checkpoint file.
		latestSnapshot = tf.train.latest_checkpoint(ModelFilePaths.modelDir)

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in %s: ' % ModelFilePaths.modelDir)

		# load saved model if available
		if latestSnapshot:
			print('Init with stored values from %s' % latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			# init all variables
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess, saver)


	def toSparse(self, texts):
		""" Put ground truth texts into sparse tensor for ctc_loss.
			Arguments:
				texts - all ground truth texts.
			Returns:
				(indices, values, shape) - a list containing elements of sparse tensor.

		"""
		# A 2-D int64 tensor, which specifies the indices of the elements in the sparse tensor that contain nonzero values.
		indices = []
		# A 1-D tensor of any type, which supplies the values for each element in indices. 
		values = []
		# A 1-D int64 tensor, which specifies the dense_shape of the sparse tensor. 
		# Takes a list indicating the number of elements in each dimension.
		# Size: [batchSize, maxLenthOfText]
		shape = [len(texts), 0]

		# Go over all texts
		# enumerate - Returns a list containing a pair: (index, element)
		for (batchElement, text) in enumerate(texts):
			# Convert texts to string of labels (i.e. indexes of chars)
			labelStr = [self.charList.index(c) for c in text]
			# Sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# Put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput, batchSize):
		""" Extract texts from output of CTC decoder.
			Arguments:
				ctcOutput - decoded strings (as numbers/labels) from output of CTC decoder to convert them to chars,
				batchSize - number of elements in single batch.
			Returns:
				String, containing processed and decoded appropriate way (depends on decoder type) characters.
		"""
		
		# Contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(batchSize)]

		# TODO: Use WordBeamSearch or not???
		# Word beam search: label strings terminated by blank
		# if self.decoderType == DecoderType.WordBeamSearch:
		# 	blank = len(self.charList)
		# 	for b in range(batchSize):
		# 		for label in ctcOutput[b]:
		# 			# If element is 'blank' then don't decode it
		# 			if label==blank:
		# 				break
		# 			encodedLabelStrs[b].append(label)
		# else:

		# TF decoders: label strings are contained in sparse tensor
		# CTC returns tuple, first element is SparseTensor
		decoded = ctcOutput[0][0]

		# Go over all indices and save mapping: batch -> values into dictionary
		idxDict = { b : [] for b in range(batchSize) }
		# decoded: A list of length top_paths, where decoded[j] is a SparseTensor containing the decoded outputs:
		# decoded[j].indices: Indices matrix (total_decoded_outputs[j] x 2) The rows store: [batch, time].
		# decoded[j].values: Values vector, size (total_decoded_outputs[j]). The vector stores the decoded classes for beam j.
		for (idx, idx2d) in enumerate(decoded.indices):
			label = decoded.values[idx]
			batchElement = idx2d[0] # index according to [batch, time]
			encodedLabelStrs[batchElement].append(label)

		# Map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		""" Feed a batch into the NN to train it.
		 	Returns:
		 		lossVal - loss value of loss function computed on this batch.
		"""
		numBatchElements = len(batch.imgs)
		sparse = self.toSparse(batch.gtTexts)
		# Decay learning rate (If Adam, then don't needed)
		# TODO: Set Adam optimizer
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001)
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		(_, lossVal) = self.sess.run(evalList, feedDict)
		# Every batch counted
		self.batchesTrained += 1
		return lossVal


	def dumpNNOutput(self, rnnOutput):
		""" Dump(save) the output of the RNN to CSV file(s)"""
		# All files in "../dump/" is in .gitignore !!!

		# If path do not exist create it
		if not os.path.isdir(ModelFilePaths.dumpDir):
			os.mkdir(ModelFilePaths.dumpDir)

		# RNN Output is a matrix of size [32, batchSize, 80]
		# Iterate over all batch elements and create a CSV file for each one
		maxT, maxB, maxC = rnnOutput.shape
		for b in range(maxB):
			csv = ''
			for t in range(maxT):
				for c in range(maxC):
					# Forming each string
					csv += str(rnnOutput[t, b, c]) + ';'
				csv += '\n'
			# Forming filename
			fn = ModelFilePaths.dumpDir + 'rnnOutput_' + str(b)+'.csv'
			print('Write dump of NN to file: ' + fn)

			with open(fn, 'w') as f:
				f.write(csv)


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		""" Feed a batch into the NN to recognize the texts.
		 Arguments:
		 	batch - batch to be feeded,
		 	calcProbability - True,if needed to compute labeling probability, False otherwise,
		 	probabilityOfGT -  True,if needed to compute labeling probability of general truth texts,
		 		False, if needed to compute labeling probability of NN output texts.
		Return:
			(texts, probs):
			texts - decoded texts,
			probs - labeling probabilities. If there was no need to compute them, then 'None' type.
		"""
		# Decode 
		numBatchElements = len(batch.imgs)
		# Optionally save RNN output
		evalRnnOutput = self.dump or calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		# Collect and compile feed dictionary for TF session
		feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		# Run session
		# Returns: [decoded, ?]
		evalRes = self.sess.run(evalList, feedDict)
		# Take decoded strings and convert them to chars
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		# Feed RNN output and recognized text into CTC loss to compute labeling probability
		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			# Get input of CTC layer
			ctcInput = evalRes[1]
			# And compute CTC loss function per element 
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			# Session Returns: 1-D float Tensor, size [batch], containing the negative log probabilities.
			lossVals = self.sess.run(evalList, feedDict)
			# Calc probabilitie (see paper: Maximum Likehood)
			probs = np.exp(-lossVals)

		# Dump the output of the NN to CSV file(s)
		if self.dump:
			self.dumpNNOutput(evalRes[1])

		return (texts, probs)
	

	def save(self):
		""" Save model to file. """
		self.snapID += 1
		self.saver.save(self.sess, ModelFilePaths.snapshotsDir, global_step=self.snapID)
 
