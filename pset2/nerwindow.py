from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot, sigmoid
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)

    mean_f1 = 100*sum(f1[1:] * support[1:])/sum(support[1:])
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % mean_f1

    return mean_f1

def tanh(x):
    return 2 * sigmoid(2*x) - 1

# tanh_val is the result of the tanh function, i.e. tanh(x)
def tanh_derivative(tanh_val):
    return 1 - tanh_val**2

##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate

        dims[0] = windowsize * wv.shape[1]         # input dimension
        param_dims = dict(W1=(dims[1], dims[0]),   # 100 x 150
                          b2=(dims[1],),           # 100 x 1
                          W2=(dims[2], dims[1]),   # 5 X 100
                          b3=(dims[2],),           # 5 x 1
                          )

        param_dims_sparse = dict(L=wv.shape)       # |V| x 50

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####
        self.sparams.L = wv.copy();

        self.params.W1 = random_weight_matrix(param_dims['W1'][0], param_dims['W1'][1])

        self.params.b2 = append([], random_weight_matrix(param_dims['b2'][0], 1))
        self.params.b3 = append([], random_weight_matrix(param_dims['b3'][0], 1))
        self.params.W2 = random_weight_matrix(param_dims['W2'][0], param_dims['W2'][1])
        self.n = wv.shape[1]

        # informational
        self.windowsize = windowsize
        self.hidden_units = dims[1]

#        print 'params init-ed'
#        print 'n in ' + str(self.n)
#        print 'W1 in ' + str(self.params['W1'].shape)
#        print 'b2 in ' + str(self.params['b2'].shape)
#        print 'W2 in ' + str(self.params['W2'].shape)
#        print 'b3 in ' + str(self.params['b3'].shape)

        #### END YOUR CODE ####



    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        window_vecs = []
        for w in window:
            window_vecs.append(self.sparams.L[w])

        x = append([], window_vecs)
        z2 = dot(self.params.W1, x)
        a2 = tanh(z2 + self.params.b2)
        z3 = dot(self.params.W2, a2)
        y_hat = softmax(z3 + self.params.b3)

        #print 'acc_grads: ' + str(y_hat)
        ##
        # Backpropagation
        d3 = y_hat - make_onehot(label, y_hat.shape[0]); # label is 'y'

        self.grads.b3 += d3
        self.grads.W2 += outer(d3, a2) + self.lreg * self.params.W2

        d2 = multiply(tanh_derivative(a2), dot(self.params.W2.T, d3))

        self.grads.W1 += outer(d2, x) + self.lreg * self.params.W1
        self.grads.b2 += d2

        x_grads = dot(self.params['W1'].T, d2)

        for i in range(0, len(window)):
            self.sgrads.L[ window[i] ] = x_grads[i*self.n : (i+1)*self.n]

        #### END YOUR CODE ####


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        #### YOUR CODE HERE ####
        X = []
        for window in windows:
            window_vecs = []
            for w in window:
                window_vecs.append(self.sparams.L[w])
            x = append([], window_vecs)

            X.append(x)

        X = array(X)
        z2 = dot(X, self.params.W1.T)
        a2 = tanh(z2 + self.params.b2)
        z3 = dot(a2, self.params.W2.T)

        #print 'compute x ' + str(X[0])
        y_hat = [softmax(z + self.params.b3) for z in z3]

        P = y_hat

        #### END YOUR CODE ####

        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####

        """Predict most likely class."""
        P = self.predict_proba(windows)
        c = argmax(P, axis=1)
        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):

        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####
        # basically calculating the predictions, extracting the correct one, take the log and then sum it up
        # same as J = - sum[y_k * log(yhat_k]

        if not hasattr(labels, "__iter__"):
            labels = [labels]


        yhat = self.predict_proba(windows)

        J = 0
        for i in range(0, len(labels)):
            J -= log(yhat[i][labels[i]])

        J += (self.lreg/2) * (sum(self.params.W2**2) + sum(self.params.W1**2)) # now regularize...
        #### END YOUR CODE ####

        return J


    def grad_check_naive(self, x, y, param, eps=1e-4, tol=1e-6, verbose=False):
        """
        Generic gradient check: uses current params
        aonround a specific data point (x,y)

        This is implemented for diagnostic purposes,
        and is not optimized for speed. It is recommended
        to run this on a couple points to test a new
        neural network implementation.
        """
        # Accumulate gradients in self.grads
        self._reset_grad_acc()
        self._acc_grads(x, y)
        self.sgrads.coalesce() # combine sparse updates

        theta = self.params[param]
        grad_computed = self.grads[param]
        grad_approx = zeros(theta.shape)
        for ij, v in ndenumerate(theta):
            J  = self.compute_loss(x, y)
            tij = theta[ij]
            theta[ij] = tij + eps
            Jplus  = self.compute_loss(x, y)
            theta[ij] = tij - eps
            Jminus = self.compute_loss(x, y)
            theta[ij] = tij # reset
            grad_approx = (Jplus - Jminus)/(2*eps)

            # Compare gradients
            reldiff = abs(grad_approx - grad_computed[ij]) / max(1, abs(grad_approx), abs(grad_computed[ij]))
            print 'reldiff %f' % reldiff
            if reldiff > tol:
                print "Gradient check failed."
                print 'J+ %f; J- %s' % (Jplus, Jminus)
                print "First gradient error found at index %s" % str(ij)
                print "Your gradient: %f \t Numerical gradient: %f" % (grad_computed[ij], grad_approx)
                return
            print 'congratulations, grad[%s] check passed! with reldiff %f' % (ij, reldiff)
        self._reset_grad_acc()
