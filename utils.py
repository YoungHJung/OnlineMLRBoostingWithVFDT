import os
import csv
import numpy as np
import copy
from hoeffdingtree import * 
import arff   

DATA_DIR = os.path.join(os.getcwd(), 'data')

def get_filepath(dataname, key):
    ''' Locate the data file and return the path
    Args:
        dataname (string): folder name that contains the data
        key (string): keyword that is a part of the file
    Returns:
        (string): the file path
    '''
    file_dir = os.path.join(DATA_DIR, dataname)
    for filename in os.listdir(file_dir):
        if key in filename:
            return os.path.join(file_dir, filename)
    return None

def parse_attributes(data):
    ''' Parse the attribute information out of arff data
    Args:
        data (dictionary): data loaded by arff.load()
    Returns:
        class_index (int): the index that contains label
        header (list): covariate names
        data_types (list): data types
    '''
    class_index = 0
    data_types = []
    header = []
    for i, attribute in enumerate(data['attributes']):
        if attribute[1] == ['0','1']:
            class_index = i
            break
        header.append(attribute[0])
        data_types.append(attribute[1].capitalize())
    return class_index, header, data_types

def open_dataset(filepath):
    ''' Load the data and output Dataset 
    Args:
        filepath (string): the path that contains the data file
    Returns:
        (dataset): the data in Dataset format
    '''
    
    data = arff.load(open(filepath, 'rb'))
    class_index, header, data_types = parse_attributes(data)
    nominal_indices = []
    for i, data_type in enumerate(data_types):
        if data_type != 'Numeric':
            nominal_indices.append(i)
    class_values = []
    instances = []
    for row in data['data']:
        inst = row[:class_index]
        label = reduce(lambda x,y:x+y, row[class_index:])
        class_values.append(label)
        inst.insert(0, label)
        instances.append(inst)
    
    class_values = list(set(class_values))
    attributes = []
    attributes.append(Attribute('Class', class_values, att_type='Nominal'))
    for i, h in enumerate(header):
        data_type = data_types[i]
        if data_type == 'Numeric':
            attributes.append(Attribute(h, att_type='Numeric'))
        else:
            att_values = data['attributes'][i][1]
            attributes.append(Attribute(h, att_values, att_type='Nominal'))
    
    dataset = Dataset(attributes, 0)
    for inst in instances:
        inst[0] = int(attributes[0].index_of_value(str(inst[0])))
        for i in nominal_indices:
            inst[i+1] = int(attributes[i+1].index_of_value(str(inst[i+1])))
        dataset.add(Instance(att_values=inst))

    return dataset

def open_slc_dataset(filepath, num_covs=None):
    ''' Load the data and output single-label Dataset 
    Args:
        filepath (string): the path that contains the data file
        num_covs (int): the number of covariates used by the learner
    Returns:
        (dataset): the data in Dataset format (with single-label structure)
    '''
    data = arff.load(open(filepath, 'rb'))
    class_index, header, data_types = parse_attributes(data)
    if num_covs is None or num_covs > class_index:
        num_covs = class_index
    nominal_indices = []
    for i, data_type in enumerate(data_types):
        if data_type != 'Numeric':
            nominal_indices.append(i)
    instances = []
    for row in data['data']:
        inst = row[:class_index]
        label = '0'
        inst.insert(0, label)
        inst = inst[:num_covs+1]
        instances.append(inst)
    
    num_classes = len(data['data'][0]) - class_index
    class_values = map(str, range(num_classes))
    attributes = []
    attributes.append(Attribute('Class', class_values, att_type='Nominal'))
    for i, h in enumerate(header):
        data_type = data_types[i]
        if data_type == 'Numeric':
            attributes.append(Attribute(h, att_type='Numeric'))
        else:
            att_values = data['attributes'][i][1]
            attributes.append(Attribute(h, att_values, att_type='Nominal'))
    attributes = attributes[:num_covs+1]
    dataset = Dataset(attributes, 0)
    for inst in instances:
        inst[0] = int(attributes[0].index_of_value(str(inst[0])))
        for i in nominal_indices:
            if i > num_covs:
                break
            inst[i+1] = int(attributes[i+1].index_of_value(str(inst[i+1])))
        dataset.add(Instance(att_values=inst))

    return dataset

def convert_to_slc_dataset(dataset):
    ''' Convert a Dataset to a single-label one
    Args:
        dataset (dataset): the input dataset
    Returns:
        (dataset): dataset with single-label structure
    '''
    n = dataset.num_attributes()
    attributes = []
    for i in xrange(n):
        attributes.append(dataset.attribute(i))
    sample_label = dataset.class_attribute().value(0)
    att_values = map(str, xrange(len(sample_label)))
    attributes[0] = Attribute('Class', att_values, att_type='Nominal')
    new_dataset = Dataset(attributes, 0)
    return new_dataset

def rank_loss(s, Y):
    ''' The rank loss
    Args:
        s (list): cumulative votes
        Y (set): true label
    Returns:
        (float): the rank loss
    '''
    k = len(s)
    normalize_const = float(len(Y)*(k-len(Y)))
    cnt = 0
    Y_complement = set(range(k)).difference(Y)
    if normalize_const == 0:
        return 0
    for l in Y:
        for r in Y_complement:
            if s[r] > s[l]:
                cnt += 1
            elif s[r] == s[l]:
                cnt += 0.5
    return cnt/normalize_const

def hinge_loss(s, Y):
    ''' The hinge loss
    Args:
        s (list): cumulative votes
        Y (set): true label
    Returns:
        (float): the hinge loss
    '''
    k = len(s)
    normalize_const = float(len(Y)*(k-len(Y)))
    _sum = 0
    Y_complement = set(range(k)).difference(Y)
    if normalize_const == 0:
        return 0
    for l in Y:
        for r in Y_complement:
            if s[r] >= s[l]:
                _sum += s[r]-s[l]
    return _sum/normalize_const

def mc_potential(t, b, Y=[0], M=10000, s=None, loss=rank_loss):
    '''Approximate potential via Monte Carlo simulation
    Args:
        t (int)         : number of weak learners until final decision
        b (list)        : baseline distribution
        Y (list)        : true label
        M (int)         : number of simulations
        s (list)        : current state
        loss (function) : loss function
    Returns:
        potential value (float)
    '''
    k = len(b)
    r = 0
    if s is None:
        s = np.zeros(k)
    _sum = 0.0
    for _ in xrange(M):
        x = np.random.multinomial(t, b)
        x = x + s
        _sum += loss(x, Y)
    return _sum / M

def str_to_set(_str):
    ''' Convert a binary string to a multi-label label 
    Args:
        _str (string): binary string (1: positive label 0: negatvie label)
    Returns:
        (list): the list of positive labels
    '''
    ret = []
    for i, letter in enumerate(_str):
        if letter == '1':
            ret.append(i)
    return set(ret)

def read_params():
    ''' Read the params file 
    Returns:
        (dictionary): parsed information
    '''
    filename = 'params.csv'
    filepath = os.path.join(DATA_DIR, filename)
    ret = {}
    with open(filepath, 'rb') as f:
        r = csv.reader(f)
        for row in r:
            key = row[0]
            val = row[1]
            try: 
                val = float(val)
            except:
                pass
            ret[key] = val
    return ret