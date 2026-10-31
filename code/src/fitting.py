"""Continuous, identifiable reduced historical calibrations.
Adapted from upstream historical fitting, with exact segment propagation,
no observed-state resetting, and explicit fitting/holdout separation.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

def flow(y0,t,r,c):
    """Exact solution of y' = r y - c y^2 for t>=0 (also for r<0)."""
    t=np.asarray(t,float)
    if abs(r)<1e-10:
        return y0/(1+c*y0*t)
    e=np.exp(np.clip(r*t,-700,700))
    return y0*e/(1+c*y0*np.expm1(np.clip(r*t,-700,700))/r)

def predict(years,y0,theta,kind,start,break_year=2000.):
    t=np.asarray(years,float)-start
    if kind=='recruitment':
        beta,rho=theta
        return flow(y0,t,beta-rho,beta)
    if kind=='piecewise':
        b1,r1,b2,r2=theta
        dt=break_year-start
        yb=float(flow(y0,dt,b1-r1,b1))
        # Never anchor/reinitialize yb to an observation.
        return np.where(t<=dt,flow(y0,np.minimum(t,dt),b1-r1,b1),
                        flow(yb,np.maximum(0,t-dt),b2-r2,b2))
    if kind=='logistic':
        r,K=theta
        return flow(y0,t,r,r/K)
    if kind=='linear':
        return y0+theta[0]*t
    if kind=='exponential':
        return y0*np.exp(theta[0]*t)
    raise ValueError(kind)

def fit(years,y,kind,seed=1201,starts=32,break_year=2000.):
    years=np.asarray(years,float);y=np.asarray(y,float)
    if len(y)<3 or np.any(np.diff(years)<=0) or np.any((y<=0)|(y>=1)):
        raise ValueError('Invalid fitting series')
    scale=max(np.ptp(y),1e-8)
    if kind=='recruitment':
        lo=np.zeros(2);hi=np.ones(2)
        initial=[np.array([.05,.02]),np.array([0.,.01]),np.array([.02,0.])]
    elif kind=='piecewise':
        if not years[0]<break_year<years[-1]:
            raise ValueError('Breakpoint outside training period')
        lo=np.zeros(4);hi=np.ones(4)
        initial=[np.array([0.,.004,0.,.02]),np.array([.05,.02,.05,.04])]
    elif kind=='logistic':
        # Work in log K to condition tiny shares. Report K, not log K.
        lo=np.array([1e-8,np.log(max(y)*1.000001)])
        hi=np.array([1.,np.log(1.)])
        initial=[np.array([.15,np.log(max(y)*1.5)])]
    elif kind in ['linear','exponential']:
        if kind=='linear':
            slope=float(np.dot(years[1:]-years[0],y[1:]-y[0])/np.dot(years[1:]-years[0],years[1:]-years[0]))
        else:
            slope0=float(np.dot(years[1:]-years[0],np.log(y[1:]/y[0]))/np.dot(years[1:]-years[0],years[1:]-years[0]))
            opt=least_squares(lambda z: (y[0]*np.exp(z[0]*(years[1:]-years[0]))-y[1:])/scale,
                              [slope0],bounds=(-1.,1.),ftol=1e-12,xtol=1e-12,gtol=1e-12)
            if not opt.success: raise RuntimeError('Exponential fit failed')
            slope=float(opt.x[0])
        return {'kind':kind,'theta':[slope],'success':True,'n_starts':1,'jac_singular_values':[],
                'at_bound':False,'start':float(years[0]),'y0':float(y[0]),'break_year':None}
    else:
        raise ValueError(kind)
    def decode(z):
        return np.array([z[0],np.exp(z[1])]) if kind=='logistic' else z
    def fun(z):
        pred=predict(years,y[0],decode(z),kind,years[0],break_year)
        return (pred[1:]-y[1:])/scale # initialization point carries no fitting information
    rng=np.random.default_rng(seed)
    while len(initial)<starts:
        if kind=='logistic':
            initial.append(np.array([rng.uniform(.02,.4),rng.uniform(lo[1]+.001,min(hi[1]-.001,np.log(max(y)*10)))]))
        else:
            initial.append(rng.uniform(.001,.3,len(lo)))
    best=None
    for z in initial:
        res=least_squares(fun,np.maximum(lo+1e-12,np.minimum(hi-1e-12,z)),bounds=(lo,hi),
                          ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=2500)
        if res.success and (best is None or np.dot(res.fun,res.fun)<np.dot(best.fun,best.fun)):
            best=res
    if best is None:
        raise RuntimeError('All optimizer starts failed')
    singular=np.linalg.svd(best.jac,compute_uv=False)
    return {'kind':kind,'theta':decode(best.x).tolist(),'success':bool(best.success),
            'n_starts':len(initial),'jac_singular_values':singular.tolist(),
            'at_bound':bool(np.any(np.minimum(best.x-lo,hi-best.x)<1e-6)),
            'bounds_optimizer':[lo.tolist(),hi.tolist()],
            'start':float(years[0]),'y0':float(y[0]),
            'break_year':float(break_year) if kind=='piecewise' else None}

def fitted_predict(f,years):
    return predict(years,f['y0'],f['theta'],f['kind'],f['start'],f['break_year'] or 2000.)

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    e=y-p;sse=float(e@e);sst=float(np.sum((y-y.mean())**2))
    return {'n':len(y),'MAE':float(np.mean(abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),
            'R2':float(1-sse/sst) if sst>1e-16 else None,'SSE':sse}

def aicc(y,p,k):
    """Diagnostic Gaussian-iid AICc, k fitted curve parameters + variance.
    Conditional on initial value and fixed breakpoint, not causal evidence.
    """
    n=len(y);K=k+1
    if n<=K+1:
        return None
    sse=max(float(np.sum((np.asarray(y)-p)**2)),1e-30)
    return float(n*np.log(sse/n)+2*K+2*K*(K+1)/(n-K-1))
