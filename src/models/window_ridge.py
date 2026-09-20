"""Ridge baselines on B's exact normalized samples, with train-only intercept.

Streaming centered sufficient statistics avoid a full N x 1764 copy. Eigen
solution is equivalent to min ||y-Xb-intercept||² + alpha ||b||² (alpha>0).
Both the latest-month and full-window versions share the same fit sample IDs.
"""
import numpy as np
from scipy.linalg import eigh
from src.training.evaluation import regression_metrics


def feature_batches(dataset, mode, batch_size=2048):
    if mode not in ('latest','window'):
        raise ValueError('mode must be latest or window')
    for start in range(0,len(dataset),batch_size):
        rows=dataset.rows[start:start+batch_size]
        if mode=='latest':
            features=dataset.quant[rows]
        else:
            features=dataset.quant[rows[:,None]+np.arange(-11,1)[None,:]].reshape(len(rows),-1)
        yield start,np.asarray(features,dtype=np.float64)


def predict_ridge(dataset, model):
    values=np.empty(len(dataset))
    for start,x in feature_batches(dataset,model['mode']):
        values[start:start+len(x)]=x @ model['coef'] + model['intercept']
    return values


def fit_ridge(train,validation,*,mode,alphas,delta=1.):
    if not len(train) or not len(validation) or train.labels is None or validation.labels is None:
        raise ValueError('nonempty supervised train and validation required')
    alphas=sorted(set(float(v) for v in alphas))
    if not alphas or any(not np.isfinite(v) or v<=0 for v in alphas):
        raise ValueError('positive finite alphas required')
    width=147 if mode=='latest' else 1764
    gram=np.zeros((width,width))
    sx=np.zeros(width); xy=np.zeros(width)
    y=np.asarray(train.labels[train.rows],dtype=float)
    yv=np.asarray(validation.labels[validation.rows],dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(yv).all():
        raise ValueError('supervised labels must be finite')
    for start,x in feature_batches(train,mode):
        gram+=x.T @ x
        sx+=x.sum(axis=0)
        xy+=x.T @ y[start:start+len(x)]
    mean=sx/len(train); ymean=float(y.mean())
    gram-=len(train)*np.outer(mean,mean)
    xy-=sx*ymean
    eigenvalues,vectors=eigh(gram,check_finite=True)
    projected=vectors.T @ xy
    trials=[]; best=None; best_loss=float('inf')
    for alpha in alphas:
        coef=vectors @ (projected/(np.maximum(eigenvalues,0)+alpha))
        model={'mode':mode,'alpha':alpha,'coef':coef,'intercept':ymean-float(mean @ coef)}
        metrics=regression_metrics(yv,predict_ridge(validation,model),delta)
        trials.append({'alpha':alpha,**metrics})
        # Same selection loss as A's neural trainer; does not touch test labels.
        if metrics['huber_loss'] < best_loss:
            best,best_loss=model,metrics['huber_loss']
    return best,{'mode':mode,'n_features':width,'train_samples':len(train),'validation_samples':len(validation),
                 'selected_alpha':best['alpha'],'selection_metric':'validation_huber_loss',
                 'trials':trials,'intercept_fit':'training only','extra_standard_scaler':False}
