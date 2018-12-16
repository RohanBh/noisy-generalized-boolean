from scipy import stats
from scipy import special
from numpy import np

def multibeta(alpha, axis = 0, q = -1):
    if q == -1:
        return np.product(special.gamma(alpha), axis=axis) / special.gamma(np.sum(alpha, axis=axis))
    return np.power(special.gamma(alpha), q) / special.gamma(alpha * q)

def conditional_dist(stats.rv_continuous):

    def __init__(self, alpha, q):
        stats.rv_continuous.__init__(self, name="conditional Y|Xm")
        self.alpha = alpha
        self.q = q

    def _pdf(self, x):
        """x_array is the 2D array like [vx1,vx2, ..., vxk] where 
        vx is the 1D array of q elements"""
        return np.product(multibeta(x + self.alpha, axis=1)) / np.power(multibeta(self.alpha, q=self.q), x.size)

def model_prior(stats.rv_continuous):

    def __init__(self, k=np.e, l=4):
        stats.rv_continuous.__init__(self, name="model prior")
        self.k = k
        self.l = l

    def _pdf(self, model):
        """model is represented by a numpy boolean array"""
        model_size = np.count_nonzero(model)
        p = min(0, self.l - model_size)
        return self.k ** p

def posterior(x, model, alpha, q, k, l):
    a = conditional_dist(alpha, q)
    b = model_prior(k, l)
    return a.pdf(x) * b(model)

def metropolis(x, s, it = 1000, alpha = 0.1, q, k, l):
    """s is the number of predictors, x is the 2D array,
    alpha is the param for theta's prior, q is the number of 
    possible o/ps for bool fn, k is the strength param for 
    model prior and l is the cutoff (sort of) for same"""
    sample = []
    model = np.random.randint(0, 2, s)    
    sample.add(model)

    for i in xrange(it):
        next_model = np.array(model)
        next_model[np.random.randint(0, s)] ^= 1
        a = posterior(x, next_model, alpha, q, k, l) / posterior(x, model, alpha, q, k, l)
        if np.random.rand() <= a:
            model = next_model
        sample.add(model)


