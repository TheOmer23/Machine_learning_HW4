import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    plt.show()


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
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    x_sub_mean = x - x_mean
    y_sub_mean = y - y_mean
    
    numerator = np.sum(x_sub_mean * y_sub_mean)
    
    denominator_x = np.sum(x_sub_mean**2)
    denominator_y = np.sum(y_sub_mean**2)
    final_denominator = np.sqrt(denominator_x * denominator_y)
    
    r = numerator / final_denominator
    
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
    
    #if 'date' in X.columns:
    #    X = X.drop(columns=['date'])
    X = X.select_dtypes(include=[np.number])
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
    
    
    def h_sigmoid(self, X, theta):
        exp_x = np.exp(X @ theta)
        return exp_x / (1 + exp_x)
        
    def cost_func(self, X, y, theta):
        m = len(y)
        h = self.h_sigmoid(X, theta)
        
        J = (1/m) * (np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)))
        
        return J
    
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
        X = np.c_[np.ones(len(X)),X] ##### add bias
        self.theta = np.random.random(X.shape[1]) ##
        m = len(y)
        
        for iter in range(self.n_iter):
            h = self.h_sigmoid(X, self.theta)
            gradients = (1/m) * (X.T @ (h - y))
            
            self.theta -= self.eta * gradients
            
            J = self.cost_func(X, y, self.theta)
            self.Js.append(J)
            self.thetas.append(self.theta.copy())
            
            if len(self.Js) > 1 and abs((self.Js[-2] - self.Js[-1])) < self.eps: ##does we need abs?
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
        X = np.c_[np.ones(len(X)),X] ##### add bias
        preds = self.h_sigmoid(X, self.theta)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        
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
    n_examples = len(y)
    new_indexs = np.random.permutation(n_examples)
    
    shuffled_X = X[new_indexs]
    shuffled_y = y[new_indexs]
    
    fold_size = n_examples // folds
    
    all_cv_accuracy = []
    
    for fold_num in range(folds):
        start_index = fold_num * fold_size
        end_index = start_index + fold_size
        
        X_test = shuffled_X[start_index : end_index]
        y_test = shuffled_y[start_index : end_index]
        
        X_train = np.concatenate((shuffled_X[:start_index], shuffled_X[end_index:]), axis=0)
        y_train = np.concatenate((shuffled_y[:start_index], shuffled_y[end_index:]), axis=0)
        
        algo.fit(X_train, y_train)
        y_test_pred = algo.predict(X_test)
        
        fold_accuracy = np.mean(y_test_pred == y_test)
        all_cv_accuracy.append(fold_accuracy)
        
    cv_accuracy = np.mean(all_cv_accuracy)
        
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
    p_denominator = 1 / (sigma * np.sqrt(2 * np.pi))
    
    exp_power = -(((data - mu)**2) / (2 * sigma**2))
    
    p = p_denominator * np.exp(exp_power)
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
        num_samples, num_features = data.shape
        self.weights = np.ones(self.k) / self.k #initialize as equals weights
            
        #init mus with random uniform numbers between the min to the max of the data
        self.mus = np.random.uniform(np.min(data), np.max(data), self.k)
        
        #init sigmas with ones
        self.sigmas = np.ones(self.k)
        
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
        num_samples, num_features = data.shape
                        
        responsibilities = np.zeros((num_samples, self.k))
        
        for j in range(self.k):
            pdf_res = norm_pdf(data, self.mus[j], self.sigmas[j])
            if pdf_res.ndim > 1:
                pdf_res = pdf_res.reshape(-1)
            responsibilities[:, j] = self.weights[j] * pdf_res
        
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        self.responsibilities = responsibilities
        
                
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
        num_samples, num_features = data.shape
        
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
        self.costs = []
        
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
    
    k = len(weights)
    pdf = 0
    for j in range(k):
        pdf += weights[j] * norm_pdf(data, mus[j], sigmas[j])
    return pdf
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
        self.em_models = None
        self.unique_classes = None
        self.n_features = None

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
        self.unique_classes = np.unique(y)
        self.n_features = X.shape[1]
        self.em_models = {}
        self.prior = {}
        
        for cls in self.unique_classes:
            self.prior[cls] = np.sum(y == cls) / y.shape[0] 
        
        for cls in self.unique_classes:
            for feature in range(self.n_features):
                self.em_models[(feature, cls)] = EM(self.k,random_state=self.random_state)
                self.em_models[(feature, cls)].fit(X[y==cls, feature].reshape(-1,1))
            
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
        num_examples = X.shape[0]
        preds = np.zeros(num_examples)
        
        posteriors = {}
        
        for cls in self.unique_classes:
            cls_likelihood = np.ones(num_examples)
            for feature in range(self.n_features):
                em_model = self.em_models[(feature ,cls)]
                weights, mus, sigmas = em_model.get_dist_params()
                feature_likelihood = gmm_pdf(X[:,feature], weights, mus, sigmas)
                cls_likelihood *= feature_likelihood
            cls_posterior = cls_likelihood * self.prior[cls]
            posteriors[cls] = cls_posterior
        
        for i in range(num_examples):
            preds[i] = max(posteriors, key=lambda cls:posteriors[cls][i])
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds.reshape(-1,1)
        
        
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
    #lor part
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    
    lor_train_predictions = lor.predict(x_train)
    lor_train_acc = np.mean(lor_train_predictions == y_train)
    
    lor_test_predictions = lor.predict(x_test)
    lor_test_acc = np.mean(lor_test_predictions == y_test)
    
    #em part
    bayes_model = NaiveBayesGaussian(k=k)
    bayes_model.fit(x_train, y_train)
    
    bayes_model_train_predictions = bayes_model.predict(x_train)
    bayes_train_acc = np.mean(bayes_model_train_predictions == y_train.reshape(-1,1))
    
    bayes_model_test_predictions = bayes_model.predict(x_test)
    bayes_test_acc = np.mean(bayes_model_test_predictions == y_test.reshape(-1,1))
    
    #ploting boundaries - import the func in the begining of the code
    plot_decision_regions(x_train, y_train, lor, title = "Logistic Regression decision boundary")
    plot_decision_regions(x_train, y_train, bayes_model, title = "Naive Bayes Gaussian decision boundary")

    #ploting cost vs iterations for lor
    num_of_iterations = len(lor.Js)
    iterations_array = np.arange(num_of_iterations)
    plt.plot(iterations_array, lor.Js)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost per Iteration for Logistic Regression")
    plt.show()
    
    
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
    #dataset_a
    a_mean_0 = [0,0,0]
    a_mean_1 = [0,0,0]
    a_cov_0 = np.eye(3)
    a_cov_1 = np.eye(3)*100
    
    a_gauss_0 = multivariate_normal(mean=a_mean_0, cov=a_cov_0)
    a_gauss_1 = multivariate_normal(mean=a_mean_1, cov=a_cov_1)
    
    class_0_dataset_a_features = a_gauss_0.rvs(500)
    class_1_dataset_a_features = a_gauss_1.rvs(500)
    
    dataset_a_features = np.vstack((class_0_dataset_a_features, class_1_dataset_a_features))
    dataset_a_labels = np.concatenate((np.zeros(500), np.ones(500)))
    
    #dataset_b
    b_mean_0 = [0, 0, 0]
    b_mean_1 = [0, 0, 1]
    b_cov_0 = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    b_cov_1 = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    b_gauss_0 = multivariate_normal(mean=b_mean_0, cov=b_cov_0, allow_singular=True)
    b_gauss_1 = multivariate_normal(mean=b_mean_1, cov=b_cov_1, allow_singular=True)
    
    class_0_dataset_b_features = b_gauss_0.rvs(500)
    class_1_dataset_b_features = b_gauss_1.rvs(500)
    
    dataset_b_features = np.vstack((class_0_dataset_b_features, class_1_dataset_b_features))
    dataset_b_labels = np.concatenate((np.zeros(500), np.ones(500)))
        
    # plot for dataset_a
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points with different colors for different classes
    ax.scatter(dataset_a_features[dataset_a_labels == 0, 0], dataset_a_features[dataset_a_labels == 0, 1], dataset_a_features[dataset_a_labels == 0, 2], c='r', label='Class 0')
    ax.scatter(dataset_a_features[dataset_a_labels == 1, 0], dataset_a_features[dataset_a_labels == 1, 1], dataset_a_features[dataset_a_labels == 1, 2], c='b', label='Class 1')

    # Label the axes
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Dataset_a 3d')
    ax.legend()
    fig.text(0.1, 0.02, 'Visualize Dataset_a\non this dataset Naive Bayes will work better when compared to Logisitc Regression.', fontsize=10)
    plt.show()
    
    
    # plot for dataset b
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points with different colors for different classes
    ax.scatter(dataset_b_features[dataset_b_labels == 0, 0], dataset_b_features[dataset_b_labels == 0, 1], dataset_b_features[dataset_b_labels == 0, 2], c='r', label='Class 0')
    ax.scatter(dataset_b_features[dataset_b_labels == 1, 0], dataset_b_features[dataset_b_labels == 1, 1], dataset_b_features[dataset_b_labels == 1, 2], c='b', label='Class 1')

    # Label the axes
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Dataset_b 3d')
    ax.legend()
    fig.text(0.1, 0.02, 'Visualize Dataset_b\non this dataset Logisitc Regression will work better when compared to Naive Bayes.', fontsize=10)
    plt.show()
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }
