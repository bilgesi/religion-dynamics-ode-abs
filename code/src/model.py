"""Revised multi-religion model, adapted from bilgesi/religion-dynamics-ode-abs.

The published RHS and compartment ordering are retained. This implementation
uses a shared transition matrix for the ODE and an exchangeable, count-based
implementation of the individual hazard process. MIT, original authors 2025.
"""
from __future__ import annotations
import copy
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

RATE_NAMES = ['beta', 'q', 'sigma', 'kappa', 'tauB', 'tauM', 'rhoB', 'rhoM', 'rhoP']

def validate(c: dict) -> None:
    m = len(c['rates']['beta'])
    for name in RATE_NAMES:
        v = np.asarray(c['rates'][name], float)
        if v.shape != (m,) or not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError(f'Invalid rate vector {name}')
    if np.any(np.asarray(c['rates']['q']) > 1):
        raise ValueError('q must lie in [0,1]')
    x = np.asarray(c['initial'], float)
    if len(x) != 1 + 3*m or not np.all(np.isfinite(x)) or np.any(x < 0) or x.sum() <= 0:
        raise ValueError('Invalid initial state')
    nu = np.asarray(c['nu'], float)
    if nu.shape != (m,m) or np.any(nu < 0) or np.any(np.diag(nu) != 0):
        raise ValueError('Invalid schism matrix')
    if c['b'] < 0 or c['mu'] < 0 or c['dt'] <= 0 or c['T'] <= 0:
        raise ValueError('Invalid demographic or simulation setting')
    if c.get('time_unit') != 'year':
        raise ValueError('All rates use years')
    for e in c.get('seeds', []):
        if e['time'] < 0 or e['time'] > c['T'] or not 1 <= e['religion'] <= m:
            raise ValueError('Invalid seed event')
        if len(e['counts']) != 3 or any(n < 0 or int(n) != n for n in e['counts']):
            raise ValueError('Invalid seed counts')

def rate_array(c: dict, t: float) -> np.ndarray:
    a = np.asarray([c['rates'][k] for k in RATE_NAMES], float)
    for ch in c.get('changes', []):
        if t >= ch['time'] - 1e-12:
            for key, value in ch['rates'].items():
                a[RATE_NAMES.index(key), ch['religion']-1] = value
    # Optional measured/exogenous context: beta_r = beta0_r exp(eta_r z(t)).
    # No fitted political covariates are claimed in the present experiments.
    for ctx in c.get('context', []):
        z = np.interp(t, ctx['times'], ctx['values'])
        a[0, ctx['religion']-1] *= np.exp(ctx['eta'] * z)
    return a

@njit(cache=True)
def transition_rates(x, a, nu, death):
    """Off-diagonal per-person hazards; includes death as last destination."""
    m = a.shape[1]
    d = 1 + 3*m
    Q = np.zeros((d, d+1))
    N = np.sum(x)
    if N <= 0:
        return Q
    lam = np.empty(m)
    for r in range(m):
        lam[r] = a[0,r] * x[2+3*r] / N
    for i in range(d):
        Q[i,d] = death
        source = (i-1)//3 if i > 0 else -1
        role = (i-1)%3 if i > 0 else -1
        if role != 2:
            for r in range(m):
                if r != source:
                    Q[i,1+3*r] += (1-a[1,r])*lam[r]
                    Q[i,2+3*r] += a[1,r]*lam[r]
        if i > 0:
            r = source
            if role == 0:
                Q[i,2+3*r] += a[2,r]
                Q[i,3+3*r] += a[4,r]
                Q[i,0] += a[6,r]
            elif role == 1:
                Q[i,1+3*r] += a[3,r]
                Q[i,3+3*r] += a[5,r]
                Q[i,0] += a[7,r]
                for ell in range(m):
                    if ell != r:
                        Q[i,2+3*ell] += nu[r,ell]
            else:
                Q[i,0] += a[8,r]
    return Q

def rhs(t: float, x: np.ndarray, c: dict, fixed_rates=None) -> np.ndarray:
    a = rate_array(c,t) if fixed_rates is None else fixed_rates
    Q = transition_rates(x,a,np.asarray(c['nu'],float),c['mu'])
    return x @ Q[:,:len(x)] - x * Q.sum(axis=1) + c['b'] * x

def apply_seed(x, e):
    counts = np.asarray(e['counts'], dtype=x.dtype)
    if x[0] < counts.sum():
        raise ValueError('Not enough unaffiliated agents for scheduled transfer')
    x[0] -= counts.sum()
    off = 1 + 3*(e['religion']-1)
    x[off:off+3] += counts

def solve(c: dict, times=None, rtol=1e-9, atol=1e-9) -> tuple[np.ndarray,np.ndarray]:
    """Segmented solve: carry the terminal state forward at EVERY event.

    Saved event times are right-continuous. The previous segment is integrated
    to its exact endpoint even when that endpoint is absent from output times.
    """
    validate(c)
    times = np.arange(0,c['T']+c['dt']/2,c['dt']) if times is None else np.asarray(times,float)
    if times[0] < 0 or times[-1] > c['T']+1e-8 or np.any(np.diff(times) <= 0):
        raise ValueError('Output times must be ordered and within [0,T]')
    events = sorted(set([0.,float(c['T'])] + [float(e['time']) for e in c.get('seeds',[])]
                        + [float(e['time']) for e in c.get('changes',[])]))
    out = np.empty((len(times),len(c['initial'])))
    x = np.asarray(c['initial'],float).copy()
    for j,t0 in enumerate(events):
        for e in c.get('seeds',[]):
            if abs(e['time']-t0) < 1e-9:
                apply_seed(x,e)
        ix = np.where(np.isclose(times,t0,atol=1e-10,rtol=0))[0]
        out[ix] = x
        if j == len(events)-1:
            break
        t1 = events[j+1]
        # Changes are constant within a segment unless context is supplied.
        a = None if c.get('context') else rate_array(c,(t0+t1)/2)
        sol = solve_ivp(lambda t,z: rhs(t,z,c,a),(t0,t1),x,method='DOP853',
                        rtol=rtol,atol=atol,dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        mask = (times > t0+1e-10) & (times < t1-1e-10)
        if np.any(mask):
            out[mask] = sol.sol(times[mask]).T
        x = sol.y[:,-1].copy() # critical: evolved terminal state, not stale initial state
    if np.min(out) < -1e-5 or not np.all(np.isfinite(out)):
        raise RuntimeError('ODE produced invalid state')
    return times, out

@njit(cache=True)
def _ensemble(x0, all_rates, nu, b, mu, dt, seed_schedule, seeds):
    reps = len(seeds)
    steps = all_rates.shape[0]
    d = len(x0)
    trajectories = np.empty((reps, steps+1, d),dtype=np.int64)
    for rep in range(reps):
        np.random.seed(int(seeds[rep]))
        x = x0.copy()
        for step in range(steps+1):
            for dest in range(1,d):
                add = seed_schedule[step,dest]
                if add > x[0]:
                    raise ValueError('Insufficient S for scheduled transfer')
                x[0] -= add
                x[dest] += add
            trajectories[rep,step] = x
            if step == steps:
                break
            Q = transition_rates(x,all_rates[step],nu,mu)
            z = x.copy()
            for i in range(d):
                H = np.sum(Q[i])
                if H > 0 and x[i] > 0:
                    leave = np.random.binomial(x[i], -np.expm1(-H*dt))
                    z[i] -= leave
                    rem = leave
                    weight = H
                    # Sequential conditional binomial = one multinomial draw.
                    for k in range(d+1):
                        h = Q[i,k]
                        if h <= 0:
                            continue
                        p = min(1.,max(0.,h/weight))
                        n = np.random.binomial(rem,p) if rem > 0 else 0
                        if k < d:
                            z[k] += n
                        rem -= n
                        weight = max(0.,weight-h)
                        if weight < 1e-14:
                            break
                # Composition-preserving birth intensities use START counts.
                z[i] += np.random.poisson(b*x[i]*dt)
            x = z
    return trajectories

def stochastic(c:dict,reps:int,seed:int,dt=None):
    """Frozen-hazard multinomial ABS, one competing event/agent/step + births.

    Exact sampling of this discrete-time exchangeable agent process; not an
    exact continuous-time sampler. Convergence is tested against finer dt.
    """
    validate(c)
    if reps < 1 or int(reps) != reps or np.any(np.asarray(c['initial']) != np.asarray(c['initial'],dtype=np.int64)):
        raise ValueError('Stochastic counts and replicate count must be nonnegative integers')
    dt = c['dt'] if dt is None else dt
    if dt <= 0: raise ValueError('dt must be positive')
    steps = int(round(c['T']/dt))
    if abs(steps*dt-c['T']) > 1e-8:
        raise ValueError('T must be divisible by dt')
    times = np.arange(steps+1)*dt
    for event in c.get('changes',[])+c.get('seeds',[]):
        if abs(event['time']/dt-round(event['time']/dt)) > 1e-7:
            raise ValueError('Event times must lie on stochastic grid')
    arrays = np.asarray([rate_array(c,t) for t in times[:-1]])
    schedule = np.zeros((steps+1,len(c['initial'])),dtype=np.int64)
    for e in c.get('seeds',[]):
        j = round(e['time']/dt)
        off = 1+3*(e['religion']-1)
        schedule[j,off:off+3] += e['counts']
    seeds = np.random.SeedSequence(seed).generate_state(reps)
    out = _ensemble(np.asarray(c['initial'],np.int64),arrays,np.asarray(c['nu'],float),
                    c['b'],c['mu'],dt,schedule,seeds)
    return times,out,seeds

def shares(x):
    x = np.asarray(x,float)
    n = x.sum(axis=-1,keepdims=True)
    if np.any(n <= 0):
        raise ValueError('Population extinct; shares undefined')
    y = x[...,1:].reshape(*x.shape[:-1],-1,3).sum(axis=-1)/n
    return y
