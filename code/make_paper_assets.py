#!/usr/bin/env python3
"""Generate every numerical LaTeX macro/table directly from saved results."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent; P=ROOT.parent/'paper';P.mkdir(exist_ok=True)
def num(x,n=4):
    if x is None or (isinstance(x,float) and not np.isfinite(x)):return '--'
    if abs(x)<1e-10:return '0'
    return f'{x:.{n}g}'
def tab(name,body): (P/name).write_text(body+'\n')
def main():
    S=json.loads((ROOT/'results/simulation_summary.json').read_text());F=json.loads((ROOT/'results/historical_fits.json').read_text());E=json.loads((ROOT/'results/coexistence_stability.json').read_text())
    vals={'AOnePercent':f"{S['A']['one_percent_time']:.1f}",'AParity':f"{S['A']['parity_time']:.1f}",
          'AFinalOne':f"{S['A']['terminal'][0]['final_mean']:.3f}",'AFinalTwo':f"{S['A']['terminal'][1]['final_mean']:.3f}",
          'BFinalMinor':f"{sum(d['final_mean'] for d in S['B']['terminal'][1:]):.4f}",
          'BFinalIncumbent':f"{S['B']['terminal'][0]['final_mean']:.3f}",
          'CPopOne':f"{S['C']['terminal'][0]['final_mean']:.3f}",'CPopTwo':f"{S['C']['terminal'][1]['final_mean']:.3f}",
          'CEquilibrium':f"{E['equilibrium_religion_shares'][0]:.4f}",
          'CMaxEigen':f"{E['largest_real_eigenvalue']:.4f}",
          'BridgeMAE':f"{S['bridge']['MAE_mean']:.5f}",'BridgeMAESD':f"{S['bridge']['MAE_sd']:.5f}",
          'BridgeMAEMin':f"{S['bridge']['MAE_min']:.5f}",'BridgeMAEMax':f"{S['bridge']['MAE_max']:.5f}",
          'BridgeR':f"{S['bridge']['R2_mean']:.5f}",'BridgeRSD':f"{S['bridge']['R2_sd']:.5f}",
          'BridgeRMin':f"{S['bridge']['R2_min']:.5f}",'BridgeRMax':f"{S['bridge']['R2_max']:.5f}",
          'HorizonDifference':f"{100*S['regime_map']['max_horizon_difference']:.1f}"}
    for dataset,prefix,kinds in [('sweden','Swe',['recruitment','piecewise','linear']),('turkey','Tur',['logistic','exponential','linear']),('newzealand','NZ',['recruitment','linear'])]:
        for kind in kinds:
            g=F[dataset][kind];key=prefix+kind.capitalize();sc=1e5 if dataset=='turkey' else 100
            vals[key+'R']=f"{g['fit_metrics']['R2']:.3f}"
            vals[key+'MAE']=f"{g['fit_metrics']['MAE']*sc:.3f}"
            vals[key+'Holdout']=f"{g['holdout_metrics']['MAE']*sc:.3f}"
    tab('numbers.tex','% Automatically generated; do not edit numerical values by hand.\n'+'\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in vals.items()))
    C={n:json.loads((ROOT/'configs'/f'scenario_{n}.json').read_text()) for n in 'ABC'}
    labels={'beta':r'$\beta$','q':r'$q$','sigma':r'$\sigma$','kappa':r'$\kappa$','tauB':r'$\tau^B$','tauM':r'$\tau^M$','rhoB':r'$\rho^B$','rhoM':r'$\rho^M$','rhoP':r'$\rho^P$'}
    text=r'''\begin{table}[!htbp]\centering\small
\caption{Executable behavioral parameters. All entries except $q$ have units year$^{-1}$. B's last two columns give each minor group's rates before and after its own scheduled change. Unlisted demographic settings are $b=\mu=0.01$ in all scenarios.}\label{tab:scenario_params}
\setlength{\tabcolsep}{6pt}\begin{tabular}{lrrrrrr}\toprule
Parameter & A: $r=1$ & A: $r=2$ & B: $r=1$ & B: minor, pre & B: minor, post & C: both\\\midrule
'''
    for k,l in labels.items():
        arr=[C['A']['rates'][k][0],C['A']['rates'][k][1],C['B']['rates'][k][0],C['B']['rates'][k][1],C['B']['changes'][0]['rates'].get(k,C['B']['rates'][k][1]),C['C']['rates'][k][0]]
        text+=l+' & '+' & '.join(num(x) for x in arr)+r'\\'+'\n'
    text+=r'\bottomrule\end{tabular}\end{table}'
    tab('table_scenarios.tex',text)
    text=r'''\begin{table}[!htbp]\centering\small
\caption{Initial counts and coupling. Counts for B are specified immediately before its year-zero introduction. All scenarios start with $N_0=10{,}000$, and all omitted mutation entries are zero.}\label{tab:initials}
\begin{tabular}{lrrrrl}\toprule
Scenario & $S_0$ & $(B_1,M_1,P_1)$ & $(B_2,M_2,P_2)$ & $T$ (years) & Mutation (year$^{-1}$)\\\midrule
A & 6000 & $(3000,200,800)$ & $(0,0,0)$ &216& $\nu_{12}=0.002$\\
B & 5000 & $(4200,300,500)$ & $(0,0,0)$ &216& None; $r=2,\ldots,6$ absent\\
C & 6300 & $(2000,120,180)$ & $(1200,80,120)$ &1000& $\nu_{12}=\nu_{21}=0.02$\\
\bottomrule\end{tabular}\end{table}'''
    # Replace every displayed initial-condition row from the executable JSON.
    import re
    for n in 'ABC':
        c=C[n];init=c['initial'];assert sum(init)==10000
        triple=lambda v:'$('+','.join(str(int(x)) for x in v)+')$'
        mut=[]
        for r,row in enumerate(c['nu']):
            for ell,value in enumerate(row):
                if value:mut.append(r'$\nu_{'+str(r+1)+str(ell+1)+'}='+num(value)+'$')
        mtext=', '.join(mut) if mut else 'None'
        if len(c['nu'])==2 and c['nu'][0][1]>0 and c['nu'][0][1]==c['nu'][1][0]:
            mtext=r'$\nu_{12}=\nu_{21}='+num(c['nu'][0][1])+'$'
        row=f"{n} & {init[0]} & {triple(init[1:4])} & {triple(init[4:7])} & {num(c['T'])} & {mtext}"+r'\\'
        text=re.sub(r'^'+n+r' &.*$',lambda match:row,text,flags=re.M)
    tab('table_initials.tex',text)
    text=r'''\begin{table}[!htbp]\centering\small
\caption{Scenario B's externally prescribed events. Each introduction transfers 50 people from $S$ to $B_r$ and 100 from $S$ to $M_r$, preserving $N$. At the change year, only the four rates indicated in Table~\ref{tab:scenario_params} change.}\label{tab:schedules}
\begin{tabular}{lrr}\toprule Minor religion & Introduction year & Rate-change year\\\midrule
'''
    for e,ch in zip(C['B']['seeds'],C['B']['changes']):text+=f"{e['religion']} & {num(e['time'])} & {num(ch['time'])}"+r'\\'+'\n'
    text+=r'\bottomrule\end{tabular}\end{table}';tab('table_schedule.tex',text)
    text=r'''\begin{table}[!htbp]\centering\small
\caption{Historical fitting and held-out errors. Sweden and NZ MAE values are percentage points; Turkey MAE values are publishers per 100,000 residents. $n$ excludes the fixed initial observation and $k$ counts fitted curve parameters. Holdouts are 2015--2024 (Sweden), 2022--2024 (Turkey), and 2023 (NZ). Lower MAE is better.}\label{tab:fitmetrics}
\begin{tabular}{llrrrrr}\toprule Case & Specification & $n$ & $k$ & In-sample $R^2$ & Fit MAE & Holdout MAE\\\midrule
'''
    labels={'recruitment':'Constant rates','piecewise':'Rate shift','linear':'Linear trend','logistic':'Logistic','exponential':'Exponential'}
    df=pd.read_csv(ROOT/'results/historical_metrics.csv')
    for _,r in df.iterrows():
        sc=1e5 if r.dataset=='turkey' else 100
        name={'sweden':'Sweden','turkey':'Turkey','newzealand':'NZ'}[r.dataset]
        text+=f"{name} & {labels[r.model]} & {int(r.fit_n)} & {int(r.parameters)} & {r.fit_R2:.3f} & {r.fit_MAE*sc:.3f} & {r.holdout_MAE*sc:.3f}"+r'\\'+'\n'
    text+=r'\bottomrule\end{tabular}\end{table}';tab('table_metrics.tex',text)
    text=r'''\begin{table}[!htbp]\centering\small
\caption{All fitted parameters of the historical dynamical curves. Values below $10^{-8}$ are shown as zero (boundary solutions). Rate parameters are year$^{-1}$; $K$ and $y_0$ are population fractions. The Sweden rate shift is fixed at 2000 and the simulated state remains continuous there.}\label{tab:historical_params}
\begin{tabular}{llrrrrr}\toprule Case & Specification/period & $\beta$ & $\rho$ & $r$ & $K$ & $y_0$\\\midrule
'''
    for dataset,kinds in [('sweden',['recruitment','piecewise']),('turkey',['logistic']),('newzealand',['recruitment'])]:
        for kind in kinds:
            f=F[dataset][kind]['fit'];theta=f['theta'];y0=f['y0'];name={'sweden':'Sweden','turkey':'Turkey','newzealand':'NZ'}[dataset]
            if kind=='piecewise':rows=[('Pre-2000',theta[:2]+[None,None]),('Post-2000',theta[2:]+[None,None])]
            elif kind=='recruitment':rows=[('Constant',theta+[None,None])]
            else:rows=[('Logistic',[None,None]+theta)]
            for label,arr in rows:
                text+=name+' & '+label+' & '+' & '.join(num(v,5) for v in arr+[y0])+r'\\'+'\n'
    text+=r'\bottomrule\end{tabular}\end{table}';tab('table_fitparams.tex',text)
    text=r'''\begin{table}[!htbp]\centering\small
\caption{NZ input counts and the constructed all-resident share. Published percentages are retained separately, not substituted for the count ratio. The denominator includes people without a classified affiliation response.}\label{tab:nzdata}
\begin{tabular}{rrrrr}\toprule Year & No-religion count & All-resident count & Constructed share (\%) & Published (\%)\\\midrule
'''
    nz=pd.read_csv(ROOT/'data/newzealand.csv')
    for _,r in nz.iterrows():
        text+=f"{int(r.year)} & {int(r.none_count):,} & {int(r.resident_count):,} & {100*r.share:.3f} & {r.published_percent_series:.1f}"+r'\\'+'\n'
    text+=r'\bottomrule\end{tabular}\end{table}';tab('table_nzdata.tex',text)
    text=r'''\begin{table}[!htbp]\centering\small
\caption{Finite-step stochastic checks on a common annual grid over 216 years, using 100 replicates per row. MAE is against the ODE; MCSE is the mean pointwise standard error of the replicate mean. Both are fractions. These independently sampled errors need not decrease monotonically with step size.}\label{tab:convergence}
\begin{tabular}{lrrr}\toprule Scenario & $\Delta t$ (years) & MAE & Mean pointwise MCSE\\\midrule
'''
    for _,r in pd.read_csv(ROOT/'results/convergence.csv').iterrows():
        text+=f"{r.scenario} & {r['dt']:.3f} & {r.MAE:.6f} & {r.mean_pointwise_MCSE:.6f}"+r'\\'+'\n'
    text+=r'\bottomrule\end{tabular}\end{table}';tab('table_convergence.tex',text)
    text=r'''\begin{table}[!htbp]\centering\small
\caption{Descriptive complexity diagnostics and anchored trend coefficients. AICc assumes independent Gaussian errors, counts the error variance as a parameter, and is not reported when the small-sample correction is undefined. Comparisons across different datasets are not meaningful. $a$ is the anchored linear slope (fraction/year), and $g$ is the exponential rate (year$^{-1}$).}\label{tab:aicc}
\begin{tabular}{llrl}\toprule Case & Specification & AICc & Trend coefficient\\\midrule
'''
    for _,r in df.iterrows():
        f=F[r.dataset][r.model]['fit'];coef='--'
        if r.model in ['linear','exponential']:coef=('$a=' if r.model=='linear' else '$g=')+num(f['theta'][0],6)+'$'
        text+=f"{dict(sweden='Sweden',turkey='Turkey',newzealand='NZ')[r.dataset]} & {labels[r.model]} & {num(r.AICc,6)} & {coef}"+r'\\'+'\n'
    text+=r'\bottomrule\end{tabular}\end{table}';tab('table_aicc.tex',text)
    template=P/'manuscript_template.tex'
    if template.exists():
        import re
        manuscript=template.read_text()
        manuscript=re.sub(r'\\input\{([^}]+)\}',lambda match:(P/match.group(1)).read_text(),manuscript)
        (P/'main.tex').write_text(manuscript)
    print('Generated LaTeX numerical assets in',P)
if __name__=='__main__': main()
