import numpy as np
import arff
import time

from hoeffdingtree import *    
from onlineMLR import AdaOLMR
import utils

def main():
	seed = np.random.randint(1, 999)
	# Read params.csv file and parse the options
	params = utils.read_params()
	loss = params['loss']
	data_source = params['data_source']
	num_wls = int(params['num_wls'])
	num_covs = int(params['num_covs'])
	M = int(params['M'])
	gamma = params['gamma']

	# Load the train data
	fp = utils.get_filepath(data_source, 'train')
	data = arff.load(open(fp, 'rb'))
	class_index, _, _ = utils.parse_attributes(data)
	train_rows = data['data']

	# Load the test data
	fp = utils.get_filepath(data_source, 'test')
	data = arff.load(open(fp, 'rb'))
	test_rows = data['data']
	
	start = time.time()

	model = AdaOLMR(data_source, loss=loss,
								num_covs=num_covs, gamma=gamma)
	model.M = M
	model.gen_weaklearners(num_wls,
	                       min_grace=5, max_grace=20,
	                       min_tie=0.01, max_tie=0.9,
	                       min_conf=0.01, max_conf=0.9,
	                       min_weight=3, max_weight=10,
	                       seed=seed) 

	for i, row in enumerate(train_rows):
	    X = row[:class_index]
	    Y = row[class_index:]
	    pred = model.predict(X)
	    model.update(Y)

	cum_error = 0

	for i, row in enumerate(test_rows):
	    X = row[:class_index]
	    Y = row[class_index:]
	    pred = model.predict(X)
	    model.update(Y)
	    cum_error += utils.rank_loss(pred, model.Y)

	end = time.time()
	runtime = round(end - start, 2)
	avg_loss = round(cum_error / float(len(test_rows)), 4)

	print 'data_source', data_source
	print 'loss', loss
	print 'gamma', gamma
	print 'num_wls', num_wls
	print 'num_covs', num_covs
	print 'M', M
	print 'seed', seed
	print 'runtime', runtime
	print 'avg_loss', avg_loss
	

if __name__ == '__main__':
	main()