import csv
import numpy as np
import copy
from hoeffdingtree import *
import utils

class AdaOLMR:
    '''
    Main class for Online Multi-label Ranking Boosting Algorithm using VFDT.

    Notation conversion table: 

    v = expert_weights
    alpha =  wl_weights
    sVec = expert_votes
    yHat_t = expert_preds
    '''

    def __init__(self, data_source, loss='logistic', 
                    num_covs=20, gamma=0.1):

        # Initializing computational elements of the algorithm

        self.num_wls = None
        self.num_classes = None
        self.num_data = 0
        self.dataset = None
        self.class_index = 0
        self.cum_error = 0
        self.exp_step_size = 1
        self.loss = loss
        self.gamma = gamma
        self.M = 100
        self.num_covs = num_covs

        self.wl_edges = None
        self.weaklearners = None
        self.expert_weights = None
        self.wl_weights = None
        self.wl_preds = None
        self.expert_preds = None
        
        # Initializing data states
        self.X = None
        self.Yhat_index = None
        self.Y_index = None
        self.Yhat = None
        self.Y = None
        self.pred_conf = None

        self.initialize_dataset(data_source)

    ########################################################################

    # Helper functions
    def make_cov_instance(self, X):
        '''Turns a list of covariates into an Instance set to self.datset 
        with None in the location of the class of interest. This is required to 
        pass to a HoeffdingTree so it can make predictions.

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset

        Returns:
            pred_instance (Instance): An Instance with the covariates X and 
                      None in the correct locations

        '''
        inst_values = list(copy.deepcopy(X))
        
        inst_values = map(float, inst_values)
        
        inst_values.insert(self.class_index, None)

        # indices = range(len(inst_values))
        # del indices[self.class_index]
        # for i in indices:
        #     if self.dataset.attribute(index=i).type() == 'Nominal':
        #         inst_values[i] = int(self.dataset.attribute(index=i)
        #             .index_of_value(str(inst_values[i])))
        #     else:
        #         inst_values[i] = float(inst_values[i])


        pred_instance = Instance(att_values = inst_values)
        pred_instance.set_dataset(self.slc_dataset)
        return pred_instance

    def make_full_instance(self, X, Y):
        '''Makes a complete Instance set to self.dataset with 
        class of interest in correct place

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset
            Y (string): the class of interest corresponding to these covariates.
        
        Returns:
            full_instance (Instance): An Instance with the covariates X and Y 
                            in the correct locations

        '''

        inst_values = list(copy.deepcopy(X))

        inst_values = map(float, inst_values)
        
        inst_values.insert(self.class_index, Y)
        i = self.class_index
        inst_values[i] = int(self.slc_dataset.attribute(index=i)
                    .index_of_value(str(inst_values[i])))
        # for i in range(len(inst_values)):
        #     if self.dataset.attribute(index=i).type() == 'Nominal':
        #         inst_values[i] = int(self.slc_dataset.attribute(index=i)
        #             .index_of_value(str(inst_values[i])))
        #     else:
        #         inst_values[i] = float(inst_values[i])

        
        full_instance = Instance(att_values=inst_values)
        full_instance.set_dataset(self.slc_dataset)
        return full_instance

    def find_Y(self, Y_index):
        '''Get class string from its index
        Args:
            Y_index (int): The index of Y
        Returns:
            Y (string): The class of Y
        '''

        Y = self.dataset.attribute(index=self.class_index).value(Y_index)
        return Y

    def find_Y_index(self, Y):
        '''Get class index from its string
        Args:
            Y (string): The class of Y
        Returns:
            Y_index (int): The index of Y
        '''

        Y_index = int(self.dataset.attribute(index=self.class_index)
                    .index_of_value(Y))
        return Y_index

    ########################################################################

    def compute_cost(self, s, i):
        ''' Compute cost matrix
        Args:
            s (list): Current state
            i (int): Weak learner index
        Return:
            (numpy.ndarray) Cost matrix
        '''
        k = self.num_classes
        s = np.array(s)
        Y_complement = set(range(k)).difference(self.Y)
        normalize_const = len(self.Y)*len(Y_complement)
        ret = np.zeros(k)
        if normalize_const == 0:
            return ret

        Ylist = list(self.Y)
        Y_complement = list(Y_complement)
        if self.loss == 'logistic':
            for l in Ylist:
                ret[l] = -sum(1/(1+np.exp(s[l] - s[Y_complement])))
            for r in Y_complement:
                ret[r] = sum(1/(1+np.exp(s[Ylist] - s[r])))
            return ret/normalize_const
        else:
            ret = np.zeros(k)
            Y = list(self.Y)
            u = np.ones(k) * (1 - self.gamma*len(Y))/float(k)
            u[Y] += self.gamma
            for l in xrange(k):
                e = np.zeros(k)
                e[l] = 1
                ret[l] = utils.mc_potential(self.num_wls-i-1, u, Y, 
                                                M=self.M, 
                                                s=s+e,
                                                loss=utils.hinge_loss)
            return ret

    def get_grad(self, c, i, alpha):
        ''' Compute gradient for differnt losses
        Args:
            c (ndarray): Cost vector
            i (int): Weak learner index
            alpha (float): Weight
        Return:
            (float): Gradient
        '''
        
        if self.loss == 'logistic':
            return np.dot(c, self.wl_preds[i,:])
        elif self.loss == 'exp':
            if self.wl_preds[i] == self.Y_index:
                tmp_zeroer = np.ones(self.num_classes)
                tmp_zeroer[self.Y_index] = 0
                tmp = np.exp(s - (alpha + s[self.Y_index]))
                ret = sum(tmp_zeroer * tmp)
            else:
                tmp = s[int(self.wl_preds[i])] + alpha - s[self.Y_index]
                ret = np.exp(tmp)
            return ret
        else:
            # Can never reach this case
            return    

    def get_lr(self, i):
        ''' Get learning rate
        Args:
            i (int): Weak learner index
        Return:
            (float): Learning rate
        '''
        if self.loss == 'zero_one':
            return 1
        else:
            ret = 1/np.sqrt(self.num_data)
            if self.loss == 'logistic':
                return ret
            else:
                return ret * np.exp(-i)
        
    def update_alpha(self, c, i, alpha):
        ''' Update the weight alpha
        Args:
            c (ndarray): Cost vector
            i (int): Weak learner index
            alpha (float): Weight
        Return:
            (float): updated alpha
        '''
        if self.loss == 'zero_one':
            return 1
        else:
            grad = self.get_grad(c, i, alpha)
            lr = self.get_lr(i)
            return max(-2, min(2, alpha - lr*grad))

    def predict(self, X, verbose=False):
        '''Runs the entire prediction procedure, updating internal tracking 
        of wl_preds and Yhat, and returns a score vector

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset

        Returns:
            Yhat (list): Score vector
        '''

        self.X = np.array(X)
    
        # Initialize values

        expert_votes = np.zeros(self.num_classes)
        expert_votes_mat = np.empty([self.num_wls, self.num_classes])

        for i in xrange(self.num_wls):
            data_indices = self.data_indices[i]
            pred_inst = self.make_cov_instance(self.X[data_indices])
            # Get our new week learner prediction and our new expert prediction
            pred_probs = \
                self.weaklearners[i].distribution_for_instance(pred_inst)
            pred_probs = np.array(pred_probs)
            self.wl_preds[i,:] = pred_probs
            if verbose is True:
                print i, pred_probs
            expert_votes += self.wl_weights[i] * pred_probs
            expert_votes_mat[i,:] = expert_votes
            
        if self.loss == 'zero_one':
            pred_index = -1
        else:
            tmp = self.expert_weights/sum(self.expert_weights)
            pred_index = np.random.choice(range(self.num_wls), p=tmp)
        self.Yhat = expert_votes_mat[pred_index,:]
        self.expert_votes_mat = expert_votes_mat
        
        return self.Yhat

    def update(self, Y, X=None, verbose=False):
        '''Runs the entire updating procedure, updating interal 
        tracking of wl_weights and expert_weights
        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset. If not given
                      the last X used for prediction will be used.
            Y (list): The true class (binary list)
        '''

        if X is None:
            X = self.X

        self.X = np.array(X)
        Ystr = reduce(lambda x,y:x+y, Y)
        Yset = utils.str_to_set(Ystr)
        self.Y = Yset
        self.num_data +=1
        self.cum_error += utils.rank_loss(self.Yhat, Yset)
        expert_votes = np.zeros(self.num_classes)
        cost_vec = self.compute_cost(expert_votes, 0)
            
        for i in xrange(self.num_wls):
            alpha = self.wl_weights[i]
            w = self.weight_consts[i]
            # if self.loss == 'zero_one':
            #     w *= 5
            data_indices = self.data_indices[i]
            _max = max(cost_vec)
            for l in Yset:
                full_inst = self.make_full_instance(self.X[data_indices], l)
                full_inst.set_weight(w*(_max - cost_vec[l]))
                self.weaklearners[i].update_classifier(full_inst) 

            if verbose is True:
                print i, _max - min(cost_vec)

            # updating the quality weights and weighted vote vector
            expert_votes = self.expert_votes_mat[i,:]
            cost_vec = self.compute_cost(expert_votes, i+1)
            self.wl_weights[i] = \
                                self.update_alpha(cost_vec, i, alpha)
            if self.loss == 'logistic':
                self.expert_weights[i] *= \
                                np.exp(-utils.rank_loss(expert_votes, Yset) \
                                * self.exp_step_size)

        self.expert_weights = self.expert_weights/sum(self.expert_weights)

    def initialize_dataset(self, data_source):
        filepath = utils.get_filepath(data_source, 'train')
        self.dataset = utils.open_dataset(filepath)
        self.slc_dataset = utils.open_slc_dataset(filepath, self.num_covs)

        self.num_classes = self.slc_dataset.num_classes()
        
    def gen_weaklearners(self, num_wls, min_conf = 0.00001, max_conf = 0.9, 
                                              min_grace = 1, max_grace = 10,
                                              min_tie = 0.001, max_tie = 1,
                                              min_weight = 10, max_weight = 200, 
                                              seed = 1):
        ''' Generate weak learners.
        Args:
            num_wls (int): Number of weak learners
            Other args (float): Range to randomly generate parameters
            seed (int): Random seed
        Returns:
            It does not return anything. Generated weak learners are stored in 
            internal variables. 
        '''
        np.random.seed(seed)
        self.num_wls = num_wls
        self.weaklearners = [HoeffdingTree() for _ in range(num_wls)]

        min_conf = np.log10(min_conf)
        max_conf = np.log10(max_conf)
        min_tie = np.log10(min_tie)
        max_tie = np.log10(max_tie)

        for wl in self.weaklearners:
            wl._header = self.slc_dataset
            conf = 10 ** np.random.uniform(low=min_conf, high=max_conf)
            grace = np.random.uniform(low=min_grace, high=max_grace)
            tie = 10**np.random.uniform(low=min_tie, high=max_tie)
            wl.set_split_confidence(conf)
            wl.set_grace_period(grace)
            wl.set_hoeffding_tie_threshold(tie)
            
        self.wl_edges = np.zeros(num_wls)
        self.expert_weights = np.ones(num_wls)/num_wls
        if self.loss == 'zero_one':
            self.wl_weights = np.ones(num_wls)   
        else:
            self.wl_weights = np.zeros(num_wls)
        self.wl_preds = np.zeros((num_wls, self.num_classes))
        self.expert_preds = np.zeros(num_wls)

        self.weight_consts = [np.random.uniform(low=min_weight, high=max_weight)
                                for _ in range(num_wls)]

        self.data_indices = []
        data_len = self.dataset.num_attributes() - 1
        if data_len <= self.num_covs:
            for _ in xrange(num_wls):
                self.data_indices.append(range(data_len))
        else:
            for _ in xrange(num_wls):
                data_indices = np.random.choice(range(data_len), 
                                                    self.num_covs, 
                                                    replace=False)
                self.data_indices.append(data_indices)