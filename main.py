import time
from scipy import special
import numpy as np
import pandas as pd

def pdf_y_given_x_m(x_model, alpha, q, y):
    """Y is the response of system, X_Model is the
    input under model m, alpha is the parameter of
    theta's prior, q is the number of possible responses
  
                        ( gamma(q * alpha) * gamma(vx0 + alpha)*gamma(vx1 + alpha)*... )
    P(Y|X,M) = product--------------------------------------------------------------------
                                   (gamma(alpha) ** q) * gamma(nx + q*alpha)
    vx is the 2D array like [vx1,vx2, ..., vxk] where
    vxi is the 1D array of q elements
    """
    vx = parse(x_model, y, q)
    vx = vx + alpha
    num = special.loggamma(vx).sum(axis=1)
    den = special.loggamma(vx.sum(axis=1))
    num -= den
    num += special.loggamma(q * alpha) - q * special.loggamma(alpha)
    return num.sum()

def model_prior(model, prior_strength, size_cutoff):
    """model is represented by a numpy boolean array"""
    model_size = np.count_nonzero(model)
    return min(0, size_cutoff - model_size) * np.log(prior_strength)

def posterior(y, x, model, alpha, q, prior_strength, size_cutoff):
    x_model = x[:, model]
    return pdf_y_given_x_m(x_model, alpha, q, y) + model_prior(model, prior_strength, size_cutoff)

def metropolis(x, y, q, **kwargs):
    """arity is the number of predictors, alpha is the param
    for theta's prior, q is the number of possible o/ps for
    bool fn, k is the strength param for model prior and l
    is the cutoff (sort of) for same
    iterations=10000, alpha=0.1, k=np.e, l=4, show_progress=True, interval=500
    """
    # Parse kwargs
    iterations = kwargs['iterations'] if 'iterations' in kwargs else 10000
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.1
    model_strength = kwargs['model_strength'] if 'model_strength' in kwargs else np.e
    model_size_cutoff = kwargs['model_size_cutoff'] if 'model_size_cutoff' in kwargs else 4
    show_progress = kwargs['show_progress'] if 'show_progress' in kwargs else False
    interval = kwargs['interval'] if 'interval' in kwargs else 500
   
    arity = x.shape[1]
    sample = []
    model = np.round(np.random.randint(0, 2, arity)) != 0
    sample.append(model)
   
    loop_start_time = time.time()
    loop_current_time = time.time()
    print("*"*10,"Running MCMC", "*"*10)
    for i in range(iterations):
        if show_progress and i % interval == 0 and i != 0:
            time_elapsed = time.time() - loop_current_time
            print(f"{i}: {interval} iterations took {time_elapsed:.2f} seconds", " "*40)
            loop_current_time = time.time()
        next_model = model.copy()
        next_model[np.random.randint(0, arity)] ^= True
        if np.count_nonzero(next_model) == 0:
            continue
        next_posterior = posterior(y, x, next_model, alpha, q, model_strength, model_size_cutoff)
        current_posterior = posterior(y, x, model, alpha, q, model_strength, model_size_cutoff)
        jump_probability = next_posterior - current_posterior
        if np.log(np.random.rand()) <= jump_probability:
            model = next_model
        sample.append(model)
        progress = (i % interval) * 100 // interval
        r_progress = (progress) // 2
        if show_progress:
            print(f"\rProgress: [{(progress):2d}%] [{'':#>{r_progress}}{'':.>{50 - r_progress}}]", end='\r')
    print("*"*10, f"loop took {(time.time() - loop_start_time):.2f} seconds", "*"*10, " "*30)
    return sample

def parse(x, y, q):
    arity = x.shape[1]
    xi = np.core.defchararray.add('x', np.char.mod('%d', np.arange(1, arity + 1)))
    cols = np.append(xi, 'y')

    x = np.hstack((x, y[:, np.newaxis]))

    df = pd.DataFrame(x, columns=cols)

    # add a count column containing vx(j)
    df = df.assign(count=-1).groupby(cols.tolist()).count().reset_index()    
    g = df.groupby(xi.tolist()).cumcount()
    # remove 'y' column
    df = df.loc[:, df.columns != 'y']

    # x is the array [vx1, vx2, ...] where vxi is qxi ndarray for state xi
    # vxi = <vxi(1), vxi(2), ..., vxi(q-1)>
    xi = xi.tolist()
    xi.append(g)
    vx = df.set_index(xi)
    vx = vx.unstack(fill_value=0).values
    
    if vx.shape[1] < q:
        vx_stuffing = np.zeros((vx.shape[0], q - vx.shape[1]))
        vx = np.hstack((vx, vx_stuffing))

    return vx
