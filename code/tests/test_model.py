"""Regression checks for the revised implementation; run with unittest discovery."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from src.model import solve,stochastic,rhs,shares,rate_array,validate
from src.fitting import flow,predict,fit,fitted_predict
ROOT=Path(__file__).resolve().parents[1]
def config(name='A'):
    return json.loads((ROOT/'configs'/f'scenario_{name}.json').read_text())

class ModelTests(unittest.TestCase):
    def test_mass_balance(self):
        c=config();c['b']=.02
        x=np.asarray(c['initial'],float)+1
        self.assertAlmostEqual(rhs(0,x,c).sum(),.01*x.sum(),places=9)
    def test_rhs_against_written_equations(self):
        c=config('C');x=np.arange(1,8,dtype=float)*100
        a=c['rates'];S=x[0];Z=x[1:].reshape(2,3);B,M,P=Z.T
        lam=np.asarray(a['beta'])*M/x.sum();nu=np.asarray(c['nu'])
        z=np.zeros_like(x);z[0]=(c['b']-c['mu'])*S-S*lam.sum()+np.dot(a['rhoB'],B)+np.dot(a['rhoM'],M)+np.dot(a['rhoP'],P)
        for r in range(2):
            L=lam.sum()-lam[r];X=S+B.sum()+M.sum()-B[r]-M[r]
            z[1+3*r]=(c['b']-c['mu'])*B[r]+(1-a['q'][r])*lam[r]*X-B[r]*L-a['sigma'][r]*B[r]+a['kappa'][r]*M[r]-(a['rhoB'][r]+a['tauB'][r])*B[r]
            z[2+3*r]=(c['b']-c['mu'])*M[r]+a['q'][r]*lam[r]*X-M[r]*L+a['sigma'][r]*B[r]-(a['kappa'][r]+a['rhoM'][r]+a['tauM'][r])*M[r]+np.dot(nu[:,r],M)-nu[r].sum()*M[r]
            z[3+3*r]=(c['b']-c['mu'])*P[r]+a['tauB'][r]*B[r]+a['tauM'][r]*M[r]-a['rhoP'][r]*P[r]
        np.testing.assert_allclose(rhs(0,x,c),z,atol=1e-12)
    def test_absence_without_schism(self):
        c=config();c['nu']=[[0,0],[0,0]]
        _,x=solve(c,[0,100,216]);self.assertEqual(float(x[:,4:].sum()),0.)
    def test_no_creation_without_seed_B(self):
        c=config('B');c['seeds']=[]
        _,x=solve(c,[0,216]);self.assertEqual(float(x[:,4:].sum()),0.)
    def test_positive_states_and_conservation(self):
        c=config();t,x=solve(c,[0,1,50,216]);self.assertGreaterEqual(x.min(),-1e-9)
        np.testing.assert_allclose(x.sum(1),10000,rtol=1e-9)
    def test_evolved_state_across_seed(self):
        c=config();c['T']=2.;c['seeds']=[{'time':1.,'religion':2,'counts':[2,3,0]}]
        _,xx=solve(c,[0,1,2]);d=config();d['T']=1.
        _,before=solve(d,[0,1]);expected=before[-1].copy();expected[0]-=5;expected[4:7]+=[2,3,0]
        np.testing.assert_allclose(xx[1],expected,rtol=1e-9,atol=1e-8)
        d=copy.deepcopy(c);d['initial']=expected.tolist();d['seeds']=[];d['T']=1.
        _,after=solve(d,[0,1]);np.testing.assert_allclose(xx[-1],after[-1],rtol=1e-9,atol=1e-8)
    def test_seed_preserves_population(self):
        c=config('B');_,x=solve(c,[0,16,48,112,167,216]);np.testing.assert_allclose(x.sum(1),10000,atol=1e-6)
    def test_seed_happens_once_at_zero(self):
        c=config('B');_,x=solve(c,[0,1]);self.assertEqual(x[0,0],4850);self.assertEqual(x[0,4],50);self.assertEqual(x[0,5],100)
    def test_optional_context(self):
        c=config();c['context']=[{'religion':2,'times':[0,10],'values':[0,1],'eta':.5}]
        self.assertAlmostEqual(rate_array(c,10)[0,1],.45*np.exp(.5))
    def test_stochastic_reproducible(self):
        c=config();c['T']=1.
        t,x,seeds=stochastic(c,3,124);_,z,other=stochastic(c,3,124)
        np.testing.assert_array_equal(x,z);self.assertGreaterEqual(x.min(),0)
    def test_stochastic_closed_population(self):
        c=config();c['T']=2.;c['b']=c['mu']=0
        _,x,_=stochastic(c,3,24);np.testing.assert_array_equal(x.sum(-1),np.full((3,21),10000))
    def test_no_stochastic_spontaneous_emergence(self):
        c=config();c['T']=2.;c['nu']=[[0,0],[0,0]]
        _,x,_=stochastic(c,3,24);self.assertEqual(x[:,:,4:].sum(),0)
    def test_invalid_units(self):
        c=config();c['time_unit']='week'
        with self.assertRaises(ValueError):validate(c)
    def test_exact_reduced_flow(self):
        y0=.3;b=.05;r=.01;t=13.;h=1e-4
        yp=(flow(y0,t+h,b-r,b)-flow(y0,t-h,b-r,b))/(2*h);y=flow(y0,t,b-r,b)
        self.assertAlmostEqual(float(yp),float(b*y*(1-y)-r*y),places=9)
    def test_piecewise_continuity(self):
        p=[.1,.03,.02,.05]
        y=predict([1999.999999,2000,2000.000001],.95,p,'piecewise',1972)
        self.assertLess(abs(y[2]-y[0]),1e-7)
    def test_fit_initialization_fixed(self):
        y=np.array([.5,.55,.60,.65]);years=np.arange(4)
        f=fit(years,y,'recruitment',starts=4);self.assertAlmostEqual(fitted_predict(f,[0])[0],.5)
    def test_share_normalization_per_replicate(self):
        x=np.array([[10,5,5,0],[90,5,5,0]],float)
        np.testing.assert_allclose(shares(x).ravel(),[.5,.1])

if __name__=='__main__':unittest.main()
