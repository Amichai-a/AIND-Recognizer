import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores
        # BIC = -2 * logL + p * logN # N = number of data points, L = likelihood of the model
        # p = free parameters =
            # probabilities in transition matrix parameters +
            # initial distribution parameters +
	        # Gaussian mean parameters+
	        # Gaussian variance parameters
            # = n*(n-1) + (n-1) + 2*d*n = n^2 + 2*d*n - 1 # d = number of features


        curr_best_model = None
        curr_best_score = float("inf")

        n_components_range = range(self.min_n_components, self.max_n_components + 1)
        for n_comp in n_components_range:
            try:
                tmp_model = self.base_model(n_comp)
                logL = tmp_model.score(self.X, self.lengths) # L = likelihood of the model
                d = self.X.shape[1] # number of features
                params = n_comp ** 2 + 2 * d * n_comp - 1 # n^2 + 2*d*n - 1
                logN = np.log(self.X.shape[0]) # N = number of data points
                bic_score = -2 * logL + params * logN # -2 * logL + p * logN
                if bic_score < curr_best_score: # Find minimize BIC score
                    curr_best_score = bic_score
                    curr_best_model = tmp_model

            except Exception as e:
                pass

        return curr_best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i)) 
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on DIC scores
        # DIC = likelihood(this word) - average likelihood(other words) = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        # M = the total words quantity
        # The maximum DIC score wins

        curr_best_model = None
        curr_best_score = float("-inf")

        n_components_range = range(self.min_n_components, self.max_n_components + 1)
        for n_comp in n_components_range:
            try:
                tmp_model = self.base_model(n_comp)
                curr_logL = tmp_model.score(self.X, self.lengths)
                others_logL_sum = 0
                for w in self.words:
                    if w != self.this_word:
                        other_X, other_lengths = self.hwords[w]
                        others_logL_sum += tmp_model.score(other_X, other_lengths)

                M = len(self.words) # total words quantity
                # DIC = likelihood(this word) - average likelihood(other words) = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                dic_score = curr_logL - others_logL_sum / (M - 1)
                if dic_score > curr_best_score:
                    curr_best_score = dic_score
                    curr_best_model = tmp_model
            except:
                pass

        return curr_best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Implement model selection using CV

        if len(self.sequences) < 2: # We don't have enough sequences for splitting to training and testing folds
            return self.base_model(self.n_constant)

        # Will Break the training set into "folds" as explained at the notebook
        split_method = KFold(n_splits = min(3, len(self.sequences)))

        curr_best_model = None
        curr_best_score = float("-inf")

        n_components_range = range(self.min_n_components, self.max_n_components + 1)
        for n_comp in n_components_range:
            tmp_scores = []
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                x_train, length_train = combine_sequences(cv_train_idx, self.sequences)
                x_test, length_test = combine_sequences(cv_test_idx, self.sequences)
                try:
                    tmp_model = GaussianHMM(n_components=n_comp, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(x_train, length_train)
                    tmp_score = tmp_model.score(x_test, length_test)
                    tmp_scores.append(tmp_score)
                except Exception as e:
                    pass

            if len(tmp_scores) > 0:
                mean_scores = np.mean(tmp_scores)
            else:
                mean_scores = float("-inf")

            if mean_scores > curr_best_score:
                curr_best_score = mean_scores
                curr_best_model = tmp_model

        return curr_best_model
