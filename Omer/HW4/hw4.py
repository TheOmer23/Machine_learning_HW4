import numpy as np
import pandas as pd


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Calculate the mean of the two columns
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sub_x_x_mean = x - x_mean
    sub_y_y_mean = y - y_mean
    
    sum = np.sum(sub_x_x_mean * sub_y_y_mean)
    
    sum_x = np.sum(sub_x_x_mean ** 2)
    sum_y = np.sum(sub_y_y_mean ** 2)
    r = sum / np.sqrt(sum_x * sum_y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if 'date' in X.columns:
        X = X.drop(columns=['date'])
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    correlations = []
    for i in range(X_arr.shape[1]):
        correlations.append(np.abs(pearson_correlation(X_arr[:,i], y_arr)))
    
    top_n_sort_indexs = np.argsort(correlations)[-n_features:]
    
    best_features = X.columns[top_n_sort_indexs].tolist()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
                # initialize weights
        self.theta = np.random.rand(X.shape[1])

        for i in range(self.n_iter):
            # Compute the linear combination of inputs and weights
            z = np.dot(X, self.theta)
            # Apply the sigmoid function
            h = 1 / (1 + np.exp(-z))
            # Compute the cost
            J = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
            # Store the cost and theta values
            self.Js.append(J)
            self.thetas.append(self.theta.copy())
            # Compute the gradient
            gradient = np.dot(X.T, (h - y)) / y.size
            # Update the weights
            self.theta -= self.eta * gradient
            # Check for convergence
            if i > 0 and abs(self.Js[-1] - self.Js[-2]) < self.eps:
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        z = np.dot(X, self.theta)
        h = 1 / (1 + np.exp(-z))
        preds = np.where(h >= 0.5, 1, 0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split data into folds
    fold_size = X.shape[0] // folds
    accuracies = []

    for i in range(folds):
        # Define the start and end of the validation fold
        start = i * fold_size
        end = start + fold_size if i != folds - 1 else X.shape[0]

        # Split the data into training and validation sets
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        X_val = X[start:end]
        y_val = y[start:end]

        # Train the model
        algo.fit(X_train, y_train)

        # Predict on validation set
        predictions = algo.predict(X_val)

        # Calculate accuracy
        accuracy = np.mean(predictions == y_val)
        accuracies.append(accuracy)

    # Calculate the average accuracy over all folds
    cv_accuracy = np.mean(accuracies)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        np.random.seed(self.random_state)
        n_samples, n_features = data.shape
        self.responsibilities = np.zeros((n_samples, self.k))
        self.weights = np.ones(self.k) / self.k
        self.mus = np.random.rand(self.k) * np.ptp(data) + np.min(data) 
        self.sigmas = np.random.random(self.k)
        self.costs = []


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        n_samples = data.shape[0]

        sum_responsibilities = np.zeros((n_samples, self.k))

        for j in range(self.k):
            pdf_res = norm_pdf(data, self.mus[j], self.sigmas[j]).reshape(-1)
            sum_responsibilities[:, j] = self.weights[j] * pdf_res
        
        # Normalize the responsibilities
        sum_responsibilities /= sum_responsibilities.sum(axis=1,keepdims=True)
        self.responsibilities = sum_responsibilities
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        num_samples = data.shape[0]
        
        #new w's
        self.weights = np.mean(self.responsibilities, axis=0)
        
        
        #new mus
        for j in range(self.k):
            resp_j = self.responsibilities[:, j]
            self.mus[j] = (resp_j @ data) / (self.weights[j] * num_samples)
                
        
        #new sigmas
        for j in range(self.k):
            diff = data - self.mus[j]
            resp_j = self.responsibilities[:, j]
            self.sigmas[j] = np.sqrt((resp_j @ (diff**2)) / (self.weights[j] * num_samples))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        for _ in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            
            checked_pdfs = np.zeros((data.shape[0], self.k))
            for j in range(self.k):
                checked_pdfs[: , j] = norm_pdf(data, self.mus[j], self.sigmas[j]).reshape(-1)
            
            cost = np.sum(-np.log(checked_pdfs @ self.weights))
            
            self.costs.append(cost)
            
            if len(self.costs) > 1 and np.abs(self.costs[-2] - self.costs[-1]) < self.eps:
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_components = len(weights)
    pdf = 0

    for i in range(n_components):
        pdf += weights[i] * norm_pdf(data, mus[i], sigmas[i])
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.classes = np.unique(y)
        self.num_features = X.shape[1]
        self.gmm_models = {}
        self.class_priors = {}

        # Compute prior probabilities for each class
        for cls in self.classes:
            self.class_priors[cls] = np.mean(y == cls)

        # Fit a GMM for each feature in each class
        for cls in self.classes:
            for feature_index in range(self.num_features):
                gmm = EM(k=self.k, random_state=self.random_state)
                data_for_class_and_feature = X[y == cls, feature_index].reshape(-1, 1)
                gmm.fit(data_for_class_and_feature)
                self.gmm_models[(cls, feature_index)] = gmm   
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            posteriors = []
            for cls in self.classes:
                likelihood = 1
                for feature_index in range(self.num_features):
                    gmm = self.gmm_models[(cls, feature_index)]
                    weights, mus, sigmas = gmm.get_dist_params()
                    likelihood *= gmm_pdf(x[feature_index].reshape(-1, 1), weights, mus, sigmas)
                posterior = self.class_priors[cls] * likelihood
                posteriors.append(posterior)
            preds[i] = self.classes[np.argmax(posteriors)]

        preds = preds.reshape(-1,1)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }