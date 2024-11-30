"""Obtained and adapted from https://github.com/bjing2016/mdgen.
All rights reserved to the original authors.
"""

import json
import os

import numpy as np
import pyemma
from tqdm import tqdm

import pyemma, tqdm, os, pickle
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf, acf

PLOT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def get_featurizer(name: str, sidechains: bool = False, cossin: bool =True):
    feat = pyemma.coordinates.featurizer(name + '.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    return feat

def get_featurized_traj(name: str, sidechains: bool = False, cossin: bool = True):
    feat = get_featurizer(name, sidechains=sidechains, cossin=cossin)
    traj = pyemma.coordinates.load(name + '.pdb', features=feat)
    return feat, traj

def get_featurized_atlas_traj(name: str, sidechains: bool = False, cossin: bool =True):
    feat = get_featurizer(name, sidechains=sidechains, cossin=cossin)
    traj = pyemma.coordinates.load(name + '_prod_R1_fit.xtc', features=feat)
    return feat, traj

def get_tica(traj, lag=1000):
    tica = pyemma.coordinates.tica(traj, lag=lag, kinetic_map=True)
    # lag time 100 ps = 0.1 ns
    return tica, tica.transform(traj)

def get_kmeans(traj):
    kmeans = pyemma.coordinates.cluster_kmeans(traj, k=100, max_iter=100, fixed_seed=137)
    return kmeans, kmeans.transform(traj)[:,0]

def get_msm(traj, lag=1000, nstates=10):
    msm = pyemma.msm.estimate_markov_model(traj, lag=lag)
    pcca = msm.pcca(nstates)
    assert len(msm.metastable_assignments) == 100
    cmsm = pyemma.msm.estimate_markov_model(msm.metastable_assignments[traj], lag=lag)
    return msm, pcca, cmsm

def discretize(traj, kmeans, msm):
    return msm.metastable_assignments[kmeans.transform(traj)[:,0]]

def load_tps_ensemble(name, directory):
    metadata = json.load(open(os.path.join(directory, f'{name}_metadata.json'),'rb'))
    all_feats = []
    all_traj = []
    for i, meta_dict in tqdm(enumerate(metadata)):
        feats, traj = get_featurized_traj(f'{directory}/{name}_{i}', sidechains=True)
        all_feats.append(feats)
        all_traj.append(traj)
    return all_feats, all_traj


def sample_tp(trans, start_state, end_state, traj_len, n_samples):
    s_1 = start_state
    s_N = end_state
    N = traj_len

    s_t = np.ones(n_samples, dtype=int) * s_1
    states = [s_t]
    for t in range(1, N - 1):
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        s_t = np.zeros(n_samples, dtype=int)
        for n in range(n_samples):
            s_t[n] = np.random.choice(np.arange(len(trans)), 1, p=probs[n])
        states.append(s_t)
    states.append(np.ones(n_samples, dtype=int) * s_N)
    return np.stack(states, axis=1)


def get_tp_likelihood(tp, trans):
    N = tp.shape[1]
    n_samples = tp.shape[0]
    s_N = tp[0, -1]
    trans_probs = []
    for i in range(N - 1):
        t = i + 1
        s_t = tp[:, i]
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]

        s_tp1 = tp[:, i + 1]
        trans_prob = probs[np.arange(n_samples), s_tp1]
        trans_probs.append(trans_prob)
    probs = np.stack(trans_probs, axis=1)
    probs[np.isnan(probs)] = 0
    return probs


def get_state_probs(tp, num_states=10):
    stationary = np.bincount(tp.reshape(-1), minlength=num_states)
    return stationary / stationary.sum()

def main(name):
    out = {}
    np.random.seed(137)
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))

    ### BACKBONE torsion marginals PLOT ONLY
    if args.plot:
        feats, traj = get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=False, cossin=False)
        if args.truncate: traj = traj[:args.truncate]
        feats, ref = get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=False, cossin=False)
        pyemma.plots.plot_feature_histograms(ref, feature_labels=feats, ax=axs[0,0], color=PLOT_COLORS[0])
        pyemma.plots.plot_feature_histograms(traj, ax=axs[0,0], color=PLOT_COLORS[1])
        axs[0,0].set_title('BB torsions')

    
    ### JENSEN SHANNON DISTANCES ON ALL TORSIONS
    feats, traj = get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=True, cossin=False)
    if args.truncate: traj = traj[:args.truncate]
    feats, ref = get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True, cossin=False)

    out['features'] = feats.describe()

    out['JSD'] = {}
    for i, feat in enumerate(feats.describe()):
        ref_p = np.histogram(ref[:,i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj[:,i], range=(-np.pi, np.pi), bins=100)[0]
        out['JSD'][feat] = jensenshannon(ref_p, traj_p)

    for i in [1,3]:
        ref_p = np.histogram2d(*ref[:,i:i+2].T, range=((-np.pi, np.pi),(-np.pi,np.pi)), bins=50)[0]
        traj_p = np.histogram2d(*traj[:,i:i+2].T, range=((-np.pi, np.pi),(-np.pi,np.pi)), bins=50)[0]
        out['JSD']['|'.join(feats.describe()[i:i+2])] = jensenshannon(ref_p.flatten(), traj_p.flatten())

    ############ Torsion decorrelations
    if args.no_decorr:
        pass
    else:
        out['md_decorrelation'] = {}
        for i, feat in enumerate(feats.describe()):
            
            autocorr = acovf(np.sin(ref[:,i]), demean=False, adjusted=True, nlag=100000) + acovf(np.cos(ref[:,i]), demean=False, adjusted=True, nlag=100000)
            baseline = np.sin(ref[:,i]).mean()**2 + np.cos(ref[:,i]).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if 'PHI' in feat or 'PSI' in feat:
                axs[0,1].plot(lags, (autocorr - baseline) / (1-baseline), color=PLOT_COLORS[i%len(PLOT_COLORS)])
            else:
                axs[0,2].plot(lags, (autocorr - baseline) / (1-baseline), color=PLOT_COLORS[i%len(PLOT_COLORS)])
    
            out['md_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
           
        axs[0,1].set_title('Backbone decorrelation')
        axs[0,2].set_title('Sidechain decorrelation')
        axs[0,1].set_xscale('log')
        axs[0,2].set_xscale('log')
    
        out['our_decorrelation'] = {}
        for i, feat in enumerate(feats.describe()):
            
            autocorr = acovf(np.sin(traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000) + acovf(np.cos(traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000)
            baseline = np.sin(traj[:,i]).mean()**2 + np.cos(traj[:,i]).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if 'PHI' in feat or 'PSI' in feat:
                axs[1,1].plot(lags, (autocorr - baseline) / (1-baseline), color=PLOT_COLORS[i%len(PLOT_COLORS)])
            else:
                axs[1,2].plot(lags, (autocorr - baseline) / (1-baseline), color=PLOT_COLORS[i%len(PLOT_COLORS)])
    
            out['our_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
    
        axs[1,1].set_title('Backbone decorrelation')
        axs[1,2].set_title('Sidechain decorrelation')
        axs[1,1].set_xscale('log')
        axs[1,2].set_xscale('log')

    ####### TICA #############
    feats, traj = get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=True, cossin=True)
    if args.truncate: traj = traj[:args.truncate]
    feats, ref = get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True, cossin=True)

    tica, _ = get_tica(ref)
    ref_tica = tica.transform(ref)
    traj_tica = tica.transform(traj)
    
    tica_0_min = min(ref_tica[:,0].min(), traj_tica[:,0].min())
    tica_0_max = max(ref_tica[:,0].max(), traj_tica[:,0].max())

    tica_1_min = min(ref_tica[:,1].min(), traj_tica[:,1].min())
    tica_1_max = max(ref_tica[:,1].max(), traj_tica[:,1].max())
    
    ref_p = np.histogram(ref_tica[:,0], range=(tica_0_min, tica_0_max), bins=100)[0]
    traj_p = np.histogram(traj_tica[:,0], range=(tica_0_min, tica_0_max), bins=100)[0]
    out['JSD']['TICA-0'] = jensenshannon(ref_p, traj_p)

    ref_p = np.histogram2d(*ref_tica[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]
    traj_p = np.histogram2d(*traj_tica[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]
    out['JSD']['TICA-0,1'] = jensenshannon(ref_p.flatten(), traj_p.flatten())
    
    #### 1,0, 1,1 TICA FES
    if args.plot:
        pyemma.plots.plot_free_energy(*ref_tica[::100, :2].T, ax=axs[2,0], cbar=False)
        pyemma.plots.plot_free_energy(*traj_tica[:, :2].T, ax=axs[2,1], cbar=False)
        axs[2,0].set_title('TICA FES (MD)')
        axs[2,1].set_title('TICA FES (ours)')


    ####### TICA decorrelation ########
    if args.no_decorr:
        pass
    else:
        # x, adjusted=False, demean=True, fft=True, missing='none', nlag=None
        autocorr = acovf(ref_tica[:,0], nlag=100000, adjusted=True, demean=False)
        out['md_decorrelation']['tica'] = autocorr.astype(np.float16)
        if args.plot:
            axs[0,3].plot(autocorr)
            axs[0,3].set_title('MD TICA')
        
    
        autocorr = acovf(traj_tica[:,0], nlag=1 if args.ito else 1000, adjusted=True, demean=False)
        out['our_decorrelation']['tica'] = autocorr.astype(np.float16)
        if args.plot:
            axs[1,3].plot(autocorr)
            axs[1,3].set_title('Traj TICA')

    ###### Markov state model stuff #################
    if not args.no_msm:
        kmeans, ref_kmeans = get_kmeans(tica.transform(ref))
        try:
            msm, pcca, cmsm = get_msm(ref_kmeans, nstates=10)
    
            out['kmeans'] = kmeans
            out['msm'] = msm
            out['pcca'] = pcca
            out['cmsm'] = cmsm
        
            traj_discrete = discretize(tica.transform(traj), kmeans, msm)
            ref_discrete = discretize(tica.transform(ref), kmeans, msm)
            out['traj_metastable_probs'] = (traj_discrete == np.arange(10)[:,None]).mean(1)
            out['ref_metastable_probs'] = (ref_discrete == np.arange(10)[:,None]).mean(1)
            ######### 
        
            msm_transition_matrix = np.eye(10)
            for a, i in enumerate(cmsm.active_set):
                for b, j in enumerate(cmsm.active_set):
                    msm_transition_matrix[i,j] = cmsm.transition_matrix[a,b]
    
            out['msm_transition_matrix'] = msm_transition_matrix
            out['pcca_pi'] = pcca._pi_coarse
        
            msm_pi = np.zeros(10)
            msm_pi[cmsm.active_set] = cmsm.pi
            out['msm_pi'] = msm_pi
            
            if args.no_traj_msm:
                pass
            else:
                
                traj_msm = pyemma.msm.estimate_markov_model(traj_discrete, lag=args.msm_lag)
                out['traj_msm'] = traj_msm
        
                traj_transition_matrix = np.eye(10)
                for a, i in enumerate(traj_msm.active_set):
                    for b, j in enumerate(traj_msm.active_set):
                        traj_transition_matrix[i,j] = traj_msm.transition_matrix[a,b]
                out['traj_transition_matrix'] = traj_transition_matrix
                
            
                traj_pi = np.zeros(10)
                traj_pi[traj_msm.active_set] = traj_msm.pi
                out['traj_pi'] = traj_pi
                
        except Exception as e:
            print('ERROR', e, name, flush=True)
    
    if args.plot:
        fig.savefig(f'{args.pdbdir}/{name}.pdf')
    
    return name, out