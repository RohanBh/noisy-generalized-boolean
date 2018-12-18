from scipy import stats
from scipy import special
import numpy as np
import pandas as pd
from mpmath import *
import time

mp.pretty = True

def pdfYgivenXm(Xm, alpha, q, Y):
    """Y is the response of system, Xm is the
    input under model m, alpha is the parameter of
    theta's prior, q is the number of possible responses
    
                         ( gamma(q * alpha) * gamma(vx0 + alpha)*gamma(vx1 + alpha)*... )
    P(Y | Xm) = product--------------------------------------------------------------------
                                      (gamma(alpha) ** q) * gamma(nx + q*alpha)
    """
    start = time.time()
    # x is the 2D array like [vx1,vx2, ..., vxk] where 
    # vx is the 1D array of q elements
    x = parse(Xm, Y, q)
    x = matrix(x.tolist())
    alpha = mpf(alpha)
    q = mpf(q)
    x = x + alpha
    a = zeros(x.rows, x.cols)
    for i in range(x.rows):
        for j in range(x.cols):
            a[i, j] = gamma(x[i, j]) / gamma(alpha)
    a1 = ones(a.rows, 1)
    for i in range(a.rows):
        for j in range(a.cols):
            a1[i, 0] *= a[i, j]
    b = zeros(x.rows, 1)
    x = x * ones(x.cols, 1)
    for i in range(x.rows):
        b[i, 0] = gamma(q * alpha) / gamma(x[i, 0])
    # a = np.product(multibeta(x + alpha, axis=1), dtype=float)
    # b = np.power(multibeta(alpha, q=q), x.shape[0], dtype=float)
    p = mpf(1)
    for i in range(a.rows):
        p *= a1[i, 0] * b[i, 0]

    end = time.time()
    print(f"pdf computation took {end - start} seconds")
    return p

def model_prior(model, k=np.e, l=4):
    """model is represented by a numpy boolean array"""
    model_size = np.count_nonzero(model)
    p = mpf(min(0, l - model_size))
    return k ** p

def posterior(Y, X, model, alpha, q, k, l):
    Xm = X[:, model]
    a, b = pdfYgivenXm(Xm, alpha, q, Y), model_prior(model, mpf(k), mpf(l))
    print(f"P(Y|Xm) = {a}    P(model) = {b}")
    return a * b

def metropolis(X, Y, arity, q, it = 500, alpha = 0.1, k=np.e, l=4):
    """arity is the number of predictors, alpha is the param 
    for theta's prior, q is the number of possible o/ps for
    bool fn, k is the strength param for model prior and l
    is the cutoff (sort of) for same"""
    sample = []
    model = np.round(np.random.randint(0, 2, arity)) != 0
    print(np.count_nonzero(model))
    sample.append(model)

    for i in range(it):
        next_model = model.copy()
        next_model[np.random.randint(0, arity)] ^= True
        n, d = posterior(Y, X, next_model, alpha, q, k, l), posterior(Y, X, model, alpha, q, k, l)
        a = n / d
        print(f"{i:5}  size={np.count_nonzero(model):2}     next={n}     curr={d}     a={a}", end=' ')
        if np.random.rand() <= a:
            model = next_model
            print("*"*20, end = " ")
        print()
        print("-"*50)
        sample.append(model)

    return sample

def parse(X, Y, q):
    arity = X.shape[1]
    xi = np.core.defchararray.add('x', np.char.mod('%d', np.arange(1, arity + 1)))
    cols = np.append(xi, 'y')

    X = np.hstack((X, Y[:, np.newaxis]))
    
    df = pd.DataFrame(X, columns=cols)
    df = df.assign(count=-1).groupby(cols.tolist()).count().reset_index()
   
    allY = pd.DataFrame(np.arange(q), columns=['y'])

    x = []
    for name, group in df.groupby(xi.tolist()):
        vx = group.loc[:, ['y', 'count']]
        vx = pd.merge(vx, allY, how='right', on='y')
        vx.fillna(0, inplace=True) # replaces NaN with 0
        vx = vx.sort_values(by=['y']).values
        x.append(vx)

    return np.array(x)[...,1]

