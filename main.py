from scipy import special
import numpy as np
import pandas as pd
import time

def pdfYgivenXm(Xm, alpha, q, Y):
    """Y is the response of system, Xm is the
    input under model m, alpha is the parameter of
    theta's prior, q is the number of possible responses
    
                         ( gamma(q * alpha) * gamma(vx0 + alpha)*gamma(vx1 + alpha)*... )
    P(Y | Xm) = product--------------------------------------------------------------------
                                      (gamma(alpha) ** q) * gamma(nx + q*alpha)
    """
    # x is the 2D array like [vx1,vx2, ..., vxk] where 
    # vx is the 1D array of q elements
    x = parse(Xm, Y, q)
    x = x + alpha
    num = special.gamma(x)
    den = special.gamma(x.sum(axis=1))
    num = np.log(num).sum(axis=1)
    den = np.log(den)
    num -= den
    num += np.log(special.gamma(q * alpha)) - q * np.log(special.gamma(alpha))
    return num.sum()

def model_prior(model, k=np.e, l=4):
    """model is represented by a numpy boolean array"""
    model_size = np.count_nonzero(model)
    p = min(0, l - model_size)
    return p * np.log(k)

def posterior(Y, X, model, alpha, q, k, l):
    Xm = X[:, model]
    a, b = pdfYgivenXm(Xm, alpha, q, Y), model_prior(model, k, l)
    return a + b

def metropolis(X, Y, arity, q, it = 500, alpha = 0.1, k=np.e, l=4):
    """arity is the number of predictors, alpha is the param 
    for theta's prior, q is the number of possible o/ps for
    bool fn, k is the strength param for model prior and l
    is the cutoff (sort of) for same"""
    sample = []
    model = np.round(np.random.randint(0, 2, arity)) != 0
    sample.append(model)

    t = time.time()
    lt = time.time()
    for i in range(it):
        if i % 500 == 0:
            print(i, f"took {time.time()-lt} seconds")
            lt = time.time()
        start = time.time()
        next_model = model.copy()
        next_model[np.random.randint(0, arity)] ^= True
        n, d = posterior(Y, X, next_model, alpha, q, k, l), posterior(Y, X, model, alpha, q, k, l)
        a = n - d
        # print(f"{i:5}  size={np.count_nonzero(model):2}     next_log={n:.2f}     curr_log={d:.2f}     a={(np.e ** a):.2f}", end=' ')
        if np.log(np.random.rand()) <= a:
            model = next_model
            # print("*"*5, "change model", "*"*5, end = " ")
        end = time.time()
        # print()
        # print(f"loop took {end - start} secs")
        # print("-"*80)
        sample.append(model)
    print("*"*10,f"loop took {time.time() - t} seconds", "*"*10)
    return sample

def parse(X, Y, q):
    arity = X.shape[1]
    xi = np.core.defchararray.add('x', np.char.mod('%d', np.arange(1, arity + 1)))
    cols = np.append(xi, 'y')

    X = np.hstack((X, Y[:, np.newaxis]))

    t = time.time()
    df = pd.DataFrame(X, columns=cols)

    # add a count column containing vx(j)
    df = df.assign(count=-1).groupby(cols.tolist()).count().reset_index()    
    g = df.groupby(xi.tolist()).cumcount()
    # remove 'y' column
    df = df.loc[:, df.columns != 'y']

    # x is the array [vx1, vx2, ...] where vxi is qxi ndarray for state xi
    # vxi = <vxi(1), vxi(2), ..., vxi(q-1)>
    xi = xi.tolist()
    xi.append(g)
    x = df.set_index(xi)
    x = x.unstack(fill_value=0).values
    
    if x.shape[1] < q:
        x_stuffing = np.zeros((x.shape[0], q - x.shape[1]))
        x = np.hstack((x, x_stuffing))

    return x
