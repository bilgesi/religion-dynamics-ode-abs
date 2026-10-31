#!/usr/bin/env python3
"""Reproduce numerical outputs, figures, and LaTeX numerical tables.

Usage: python run_all.py [--part all|simulations|historical|figures]
No network requests are made. Input data and configurations are included.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,PercentFormatter
from scipy.optimize import root
from src.model import solve,stochastic,shares,rhs,RATE_NAMES
from src.fitting import fit,fitted_predict,metrics,aicc
ROOT=Path(__file__).resolve().parent
RES=ROOT/'results'; RES.mkdir(exist_ok=True)
PAPER=ROOT.parent/'paper'; PAPER.mkdir(exist_ok=True)
FIG=RES/'figures'; FIG.mkdir(exist_ok=True)
SEED=20261022

def dump(obj,path):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def load_config(name):
    return json.loads((ROOT/'configs'/f'scenario_{name}.json').read_text())

def save_scenario(name,c,reps,seed):
    t,x=solve(c)
    ts,z,seeds=stochastic(c,reps,seed)
    assert np.allclose(t,ts)
    ys=shares(z);yo=shares(x)
    means=ys.mean(0);qlo,qhi=np.quantile(ys,[.025,.975],axis=0)
    np.savez_compressed(RES/f'scenario_{name}.npz',time=t,ode=x,ode_shares=yo,
        abs_shares=ys,abs_mean=means,lower=qlo,upper=qhi,seeds=seeds,
        abs_mean_population=z.sum(-1).mean(0))
    records=[]
    for r in range(yo.shape[1]):
        for i,tv in enumerate(t):
            records.append((tv,r+1,yo[i,r],means[i,r],qlo[i,r],qhi[i,r]))
    pd.DataFrame(records,columns=['year','religion','ode_share','abs_mean_share','abs_q025','abs_q975']).to_csv(RES/f'scenario_{name}.csv',index=False)
    peaks=[{'religion':r+1,'mean_peak':float(means[:,r].max()),'mean_peak_time':float(t[np.argmax(means[:,r])]),'final_mean':float(means[-1,r]),'final_q025':float(qlo[-1,r]),'final_q975':float(qhi[-1,r])} for r in range(yo.shape[1])]
    summary={'scenario':name,'reps':reps,'base_seed':seed,'configuration':c,'terminal':peaks,
        'ODE_population_error':float(abs(x.sum(1)-np.sum(c['initial'])*np.exp((c['b']-c['mu'])*t)).max()),
        'MAE_ODE_vs_ABS':float(abs(yo-means).mean()),'mean_terminal_population':float(z[:,-1].sum(-1).mean())}
    if name=='A':
        ix=np.where(means[:,1]>=.01)[0]
        par=np.where(means[:,1]>=means[:,0])[0]
        summary.update(one_percent_time=float(t[ix[0]]) if len(ix) else None,
                       parity_time=float(t[par[0]]) if len(par) else None)
    return summary

def jacobian(fun,x,h=1e-6):
    J=np.empty((len(x),len(x)))
    for i in range(len(x)):
        d=np.zeros_like(x);d[i]=h
        J[:,i]=(fun(x+d)-fun(x-d))/(2*h)
    return J

def simulations():
    summary={}
    for i,name in enumerate('ABC'):
        print(f'Running scenario {name}: 100 independent replicates',flush=True)
        summary[name]=save_scenario(name,load_config(name),100,SEED+1000*i)
    # Local stability: eliminate S so the mass-conservation zero eigenvalue is absent.
    c=load_config('C')
    def f(z): return rhs(0,np.r_[1-z.sum(),z],c)[1:]
    sol=root(f,np.array([.2,.03,.22]*2),tol=1e-11)
    eq=np.r_[1-sol.x.sum(),sol.x]
    if not sol.success or np.max(abs(f(sol.x)))>1e-9 or np.min(eq)<0:
        raise RuntimeError('Equilibrium root failed')
    eig=np.linalg.eigvals(jacobian(f,sol.x))
    stability={'equilibrium_compartment_shares':eq.tolist(),'equilibrium_religion_shares':shares(eq).tolist(),
               'residual':float(abs(f(sol.x)).max()),'eigenvalues_real':eig.real.tolist(),
               'eigenvalues_imag':eig.imag.tolist(),'largest_real_eigenvalue':float(eig.real.max())}
    dump(stability,RES/'coexistence_stability.json')
    # Ablations explicitly retain all other configuration values.
    ab=[]
    for name,mode in [('A','no_schism'),('B','no_seeding'),('B','no_rate_changes'),('C','weak_schism')]:
        d=load_config(name)
        if mode=='no_schism': d['nu']=np.zeros_like(d['nu']).tolist()
        if mode=='no_seeding': d['seeds']=[]
        if mode=='no_rate_changes': d['changes']=[]
        if mode=='weak_schism': d['nu']=[[0,.0002],[.0002,0]]
        tt,xx=solve(d)
        yy=shares(xx)
        for r in range(yy.shape[1]):
            ab.append({'scenario':name,'ablation':mode,'religion':r+1,'terminal_share':float(yy[-1,r]),'peak_share':float(yy[:,r].max())})
        np.savez_compressed(RES/f'ablation_{name}_{mode}.npz',time=tt,shares=yy)
    pd.DataFrame(ab).to_csv(RES/'ablations.csv',index=False)
    # 30 unique configurations: 10 for each scenario, all at T=216 years.
    # A/C retain nonzero schism; B uses prescribed forcing and zero schism.
    rng=np.random.default_rng(SEED+91000); bridge=[]; configs=[]
    for i in range(30):
        name='ABC'[i//10];d=load_config(name);d['T']=216.
        for field in RATE_NAMES:
            factors=rng.uniform(.7,1.3,len(d['rates'][field]))
            d['rates'][field]=(np.asarray(d['rates'][field])*factors).tolist()
            if field=='q':d['rates'][field]=np.clip(d['rates'][field],0,1).tolist()
            # Preserve B's scheduled relative changes under the same jitter.
            for ch in d['changes']:
                if field in ch['rates']:ch['rates'][field]*=float(factors[ch['religion']-1])
        d['nu']=(np.asarray(d['nu'])*rng.uniform(.7,1.3,np.asarray(d['nu']).shape)).tolist()
        d['name']=f'{name}_bridge_{i:02}'
        configs.append(d)
        t,x=solve(d);_,z,seeds=stochastic(d,20,SEED+10000+i)
        yo=shares(x);ya=shares(z).mean(0)
        per=[]
        for r in range(yo.shape[1]):
            mt=metrics(yo[:,r],ya[:,r]);per.append(mt)
            bridge.append(dict(case=i,scenario=name,religion=r+1,replicates=20,
                               seed=SEED+10000+i,mutation=bool(np.any(d['nu'])),**mt))
        if i%5==0:print(f'ODE-stochastic comparison: case {i+1}/30',flush=True)
    bdf=pd.DataFrame(bridge);bdf.to_csv(RES/'bridge_per_religion.csv',index=False)
    cases=bdf.groupby(['case','scenario'],as_index=False)[['MAE','RMSE','R2']].mean()
    cases.to_csv(RES/'bridge_per_case.csv',index=False)
    dump(configs,RES/'bridge_configurations.json')
    summary['bridge']={'n_cases':30,'replicates_per_case':20,
       'MAE_mean':float(cases.MAE.mean()),'MAE_sd':float(cases.MAE.std(ddof=1)),
       'MAE_min':float(cases.MAE.min()),'MAE_max':float(cases.MAE.max()),
       'R2_mean':float(cases.R2.mean()),'R2_sd':float(cases.R2.std(ddof=1)),
       'R2_min':float(cases.R2.min()),'R2_max':float(cases.R2.max())}
    # Stochastic step convergence evaluated on a common yearly observation grid.
    cv=[]
    for j,name in enumerate('ABC'):
        c=load_config(name);c['T']=216.
        grid=np.arange(217.)
        _,xo=solve(c,grid);yo=shares(xo)
        for k,dt in enumerate([.1,.05,.025]):
            print(f'Convergence {name}, dt={dt}',flush=True)
            tt,zz,seeds=stochastic(c,100,SEED+20000+100*j+k,dt)
            at=np.rint(grid/dt).astype(int)
            yy=shares(zz[:,at]);avg=yy.mean(0)
            err=float(abs(avg-yo).mean())
            mcse=float((yy.std(0,ddof=1)/10).mean())
            cv.append(dict(scenario=name,dt=dt,replicates=100,MAE=err,mean_pointwise_MCSE=mcse))
    pd.DataFrame(cv).to_csv(RES/'convergence.csv',index=False)
    # Continuous finite-horizon map; no artificial categorical phase labels.
    base=load_config('A');betas=np.linspace(.03,.6,21);rhos=np.linspace(.001,.15,21)
    maprows=[]
    for rho in rhos:
        for beta in betas:
            d=copy.deepcopy(base);d['rates']['beta'][1]=float(beta);d['rates']['rhoM'][1]=float(rho)
            d['T']=432.;t,x=solve(d,np.array([0.,216.,432.]))
            yy=shares(x)
            maprows.append(dict(beta=beta,rhoM=rho,share216=yy[1,1],share432=yy[2,1]))
    pd.DataFrame(maprows).to_csv(RES/'regime_map.csv',index=False)
    summary['regime_map']={'grid':21,'deterministic':True,'horizons':[216,432],
       'max_horizon_difference':float(max(abs(r['share432']-r['share216']) for r in maprows))}
    dump(summary,RES/'simulation_summary.json')

def historical():
    all_results={};rows=[];predrows=[]
    specifications={'sweden':['recruitment','piecewise','linear'],
                    'turkey':['logistic','exponential','linear'],
                    'newzealand':['recruitment','linear']}
    cuts={'sweden':2014,'turkey':2021,'newzealand':2018}
    for dataset,kinds in specifications.items():
        df=pd.read_csv(ROOT/'data'/f'{dataset}.csv');yr=df.year.to_numpy(float);y=df.share.to_numpy(float)
        all_results[dataset]={}
        for kind in kinds:
            print(f'Historical fit: {dataset}, {kind}',flush=True)
            model=fit(yr,y,kind)
            p=fitted_predict(model,yr)
            full=metrics(y[1:],p[1:]) # exclude fixed initialization everywhere
            full['AICc']=aicc(y[1:],p[1:],len(model['theta']))
            train=yr<=cuts[dataset]
            fm=fit(yr[train],y[train],kind)
            holdp=fitted_predict(fm,yr[~train]);hm=metrics(y[~train],holdp)
            all_results[dataset][kind]={'fit':model,'fit_metrics':full,
                     'training_end':cuts[dataset],'holdout_fit':fm,'holdout_metrics':hm}
            rows.append(dict(dataset=dataset,model=kind,fit_n=full['n'],parameters=len(model['theta']),
                     fit_MAE=full['MAE'],fit_RMSE=full['RMSE'],fit_R2=full['R2'],AICc=full['AICc'],
                     holdout_n=hm['n'],holdout_MAE=hm['MAE'],holdout_RMSE=hm['RMSE'],at_bound=model['at_bound']))
            dense=np.linspace(yr[0],yr[-1],1001)
            pd.DataFrame({'year':dense,'full_fit':fitted_predict(model,dense),
                          'holdout_forecast':fitted_predict(fm,dense)}).to_csv(RES/f'fit_curve_{dataset}_{kind}.csv',index=False)
            for i,v in enumerate(yr):
                predrows.append(dict(dataset=dataset,model=kind,year=int(v),observed=y[i],predicted=p[i],
                                    fitted_after_initial=bool(i>0),holdout=bool(not train[i]),
                                    prediction_from_training_fit=float(fitted_predict(fm,[v])[0])))
    # Denominator sensitivity is explicitly separate from the primary count ratio.
    nz=pd.read_csv(ROOT/'data'/'newzealand.csv');ynz=nz.published_percent_series.to_numpy()/100
    sens=fit(nz.year.to_numpy(),ynz,'recruitment')
    all_results['NZ_published_percent_sensitivity']={'fit':sens,
         'fit_metrics':metrics(ynz[1:],fitted_predict(sens,nz.year.to_numpy())[1:]),
         'warning':'Published percentage denominators/processing differ across censuses; not primary series.'}
    dump(all_results,RES/'historical_fits.json')
    pd.DataFrame(rows).to_csv(RES/'historical_metrics.csv',index=False)
    pd.DataFrame(predrows).to_csv(RES/'historical_predictions.csv',index=False)

def setup_plot():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelweight':'bold',
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def savefig(fig,name):
    fig.savefig(FIG/name,bbox_inches='tight',pad_inches=.07)
    fig.savefig(PAPER/name,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)

def figures():
    setup_plot()
    for name in 'ABC':
        data=np.load(RES/f'scenario_{name}.npz');t=data['time'];y=data['abs_mean'];yo=data['ode_shares']
        lo=data['lower'];hi=data['upper']
        fig,ax=plt.subplots(figsize=(5.0,3.7))
        for r in range(y.shape[1]):
            ax.plot(t,y[:,r],lw=1.6,label=f'Religion {r+1}')
            ax.fill_between(t,lo[:,r],hi[:,r],alpha=.12)
        # Distinct markers rather than extra duplicated legend items for ODE.
        if name!='B':
            ax.plot(t,1-y.sum(1),ls=':',lw=1.2,label='Unaffiliated')
        ax.set(xlabel='Time (years)',ylabel='Population share',xlim=(0,t[-1]),ylim=(0,1))
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=8,ncol=2)
        fig.tight_layout();savefig(fig,f'fig_scenario_{name}.pdf')
    fits=json.loads((RES/'historical_fits.json').read_text())
    figspec={'sweden':[('recruitment','Constant rates'),('piecewise','Continuous rate shift')],
             'turkey':[('logistic','Logistic description')],
             'newzealand':[('recruitment','Constant rates')]}
    for dataset,specs in figspec.items():
        df=pd.read_csv(ROOT/'data'/f'{dataset}.csv');scale=1e5 if dataset=='turkey' else 100.
        fig,ax=plt.subplots(figsize=(5.,3.7))
        ax.scatter(df.year,df.share*scale,s=23,label='Observed',zorder=4)
        for i,(kind,label) in enumerate(specs):
            cv=pd.read_csv(RES/f'fit_curve_{dataset}_{kind}.csv')
            ax.plot(cv.year,cv.full_fit*scale,lw=1.7,ls=['--','-'][min(i,1)],label=label)
        yr=np.linspace(df.year.min(),df.year.max(),501)
        baseline=fits[dataset]['linear']['fit']
        ax.plot(yr,fitted_predict(baseline,yr)*scale,ls=':',lw=1.3,label='Anchored linear trend')
        if dataset=='sweden':ax.axvline(2000,ls=':',lw=.8,alpha=.5)
        ylabel='Peak publishers per 100,000' if dataset=='turkey' else ('No religion / all residents (%)' if dataset=='newzealand' else 'Church membership (%)')
        ax.set(xlabel='Census year' if dataset=='newzealand' else 'Year',ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(5,integer=True));ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=7.5,loc='best')
        fig.tight_layout();savefig(fig,f'fig_hist_{dataset}.pdf')
    d=pd.read_csv(RES/'regime_map.csv')
    z=d.pivot(index='rhoM',columns='beta',values='share216')
    fig,ax=plt.subplots(figsize=(6.,4.1))
    im=ax.pcolormesh(z.columns,z.index,z.to_numpy(),shading='nearest',vmin=0,vmax=1)
    cb=fig.colorbar(im,ax=ax);cb.set_label('Religion 2 share at year 216',weight='bold')
    ax.set(xlabel=r'Conversion rate $\beta_2$ (year$^{-1}$)',ylabel=r'Disaffiliation $\rho^M_2$ (year$^{-1}$)')
    fig.tight_layout();savefig(fig,'heatmap.pdf')
    d=pd.read_csv(RES/'bridge_per_case.csv')
    fig,ax=plt.subplots(figsize=(6.,3.8))
    for scenario,mark in zip('ABC',['o','s','^']):
        v=d[d.scenario==scenario];ax.scatter(v.MAE*100,v.R2,s=37,marker=mark,label=f'Scenario {scenario}')
    ax.set(xlabel='ODE-stochastic mean absolute error (percentage points)',ylabel=r'$R^2$ against ODE trajectory')
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();savefig(fig,'ode_abs_compare.pdf')
    # Optional supplementary figure: actual held-out forecasts, never full-data refits.
    for dataset,kind in [('sweden','piecewise'),('turkey','logistic'),('newzealand','recruitment')]:
        df=pd.read_csv(ROOT/'data'/f'{dataset}.csv');scale=1e5 if dataset=='turkey' else 100.
        ff=fits[dataset][kind];cut=ff['training_end'];cv=pd.read_csv(RES/f'fit_curve_{dataset}_{kind}.csv')
        fig,ax=plt.subplots(figsize=(5.,3.5))
        train=df.year<=cut
        ax.scatter(df.year[train],df.share[train]*scale,s=23,label='Training observations')
        ax.scatter(df.year[~train],df.share[~train]*scale,s=34,marker='s',label='Held-out observations')
        ax.plot(cv.year,cv.holdout_forecast*scale,lw=1.5,label='Training-only fit / forecast')
        ax.axvline(cut,ls=':',lw=1,alpha=.6)
        ax.set(xlabel='Year',ylabel='Per 100,000' if dataset=='turkey' else 'Share (%)')
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
        fig.tight_layout();savefig(fig,f'supp_holdout_{dataset}.pdf')

def manifest():
    import numba
    files={}
    for folder in ['src','configs','data','tests']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in str(path):
                files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(ROOT.glob('*.py')):
        files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    dump({'seed':SEED,'python':sys.version,'platform':platform.platform(),
          'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,
          'matplotlib':matplotlib.__version__,'numba':numba.__version__,
          'input_and_source_sha256':files,'numerical_pipeline':'run_all.py'},RES/'run_manifest.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['all','simulations','historical','figures'],default='all');args=ap.parse_args()
    start=time.time()
    if args.part in ['all','simulations']:simulations()
    if args.part in ['all','historical']:historical()
    if args.part in ['all','figures']:figures()
    manifest();print(f'Finished {args.part}; elapsed seconds={time.time()-start:.1f}',flush=True)
if __name__=='__main__':main()
