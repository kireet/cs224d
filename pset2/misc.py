##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    e = 6**.5/(m+n)**.5

    A0 = random.rand(m,n)
    A0 = A0*2*e - e
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0
