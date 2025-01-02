import pickle
import sys

import mdtraj as md
import numpy as np
import wandb

from jamun import utils_md

amino_acid_dict={"A": "ALA",  # Alanine\n",
    "R": "ARG",  # Arginine\n",
    "N": "ASN",  # Asparagine\n",
    "D": "ASP",  # Aspartic acid\n",
    "C": "CYS",  # Cysteine\n",
    "E": "GLU",  # Glutamic acid\n",
    "Q": "GLN",  # Glutamine\n",
    "G": "GLY",  # Glycine\n",
    "H": "HIS",  # Histidine\n",
    "I": "ILE",  # Isoleucine\n",
    "L": "LEU",  # Leucine\n",
    "K": "LYS",  # Lysine\n",
    "M": "MET",  # Methionine\n",
    "F": "PHE",  # Phenylalanine\n",
    "P": "PRO",  # Proline\n",
    "S": "SER",  # Serine\n",
    "T": "THR",  # Threonine\n",
    "W": "TRP",  # Tryptophan\n",
    "Y": "TYR",  # Tyrosine\n",
    "V": "VAL"   # Valine\n"
    }

def compute_JS_divergence_of_ramachandran(hist, ref_hist) -> np.ndarray:
    """Computes the Jensen-Shannon divergence between the Ramachandran histograms from two trajectories."""

    def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Computes the KL divergence between two distributions."""
        both_pq_zero = np.logical_and(p == 0, q == 0)
        p = p[~both_pq_zero]
        q = q[~both_pq_zero]
        p_log_p = p * np.log(np.where(p == 0, 1, p))
        p_log_q = p * np.log(q)
        return np.sum(p_log_p - p_log_q)

    # Compute the histograms.
    # Compute the Jensen-Shannon divergence.
    hist = hist.flatten()
    ref_hist = ref_hist.flatten()
    mix = 0.5 * (hist + ref_hist)
    js_div = 0.5 * (compute_kl_divergence(hist, mix) + compute_kl_divergence(ref_hist, mix))
    return js_div



md_time_10_np=pickle.load(open("time_md_10.pkl","rb"))

import pickle

amino_acid_dict={"A": "ALA",  # Alanine\n",
    "R": "ARG",  # Arginine\n",
    "N": "ASN",  # Asparagine\n",
    "D": "ASP",  # Aspartic acid\n",
    "C": "CYS",  # Cysteine\n",
    "E": "GLU",  # Glutamic acid\n",
    "Q": "GLN",  # Glutamine\n",
    "G": "GLY",  # Glycine\n",
    "H": "HIS",  # Histidine\n",
    "I": "ILE",  # Isoleucine\n",
    "L": "LEU",  # Leucine\n",
    "K": "LYS",  # Lysine\n",
    "M": "MET",  # Methionine\n",
    "F": "PHE",  # Phenylalanine\n",
    "P": "PRO",  # Proline\n",
    "S": "SER",  # Serine\n",
    "T": "THR",  # Threonine\n",
    "W": "TRP",  # Tryptophan\n",
    "Y": "TYR",  # Tyrosine\n",
    "V": "VAL"   # Valine\n"
    }
protein_codes=["AC", "AD", "AH", "AM", "AN", "AP", "AR", "AT", "CK", "CN", "CR", "CS", "CW", "DH", "DK", "DL", "DW", "EK", "EL", "ET", "EV", "EW", "EY", "FA", "FF", "FH", "FS", "GN", "GP", "GQ", "GR", "GT", "HC", "HI", "HK", "HP", "HR", "HT", "IG", "IK", "IM", "IQ", "KC", "KD", "KE", "KG", "KI", "KN", "KQ", "KR", "KS", "LM", "LV", "LW", "LY", "MA", "MC", "ME", "MI", "MK", "MV", "MW", "MY", "NC", "NE", "NF", "NK", "NQ", "NY", "QF", "QG", "QM", "QQ", "QW", "RC", "RF", "RL", "RQ", "RT", "RV", "RY", "SD", "SQ", "ST", "SY", "TA", "TD", "TE", "TF", "TI", "TK", "TT", "TY", "VV", "WF", "WI", "YA", "YD", "YI", "YL"]


def get_traj_from_timewarp(protein):
    traj=md.load("timewarp/%s-traj-state0.pdb"%protein)
    z=np.load("timewarp/%s-traj-arrays.npz"%protein)
    traj.xyz=z["positions"]
    traj.time=np.arange(len(traj))

    phi,psi=md.compute_phi(traj)[1],md.compute_psi(traj)[1]

    return (phi,psi),None

def get_traj_from_tbg(protein):
    traj=md.load("timewarp/%s-traj-state0.pdb"%protein)
    z=np.load("tbg/tbg_full_%s_aligned_corrected.npz"%protein)
    traj.xyz=z["samples_np"]
    traj.time=np.arange(len(traj))

    phi,psi=md.compute_phi(traj)[1],md.compute_psi(traj)[1]

    with open("time_tbg_5000.pkl","rb") as f:
        time=pickle.load(f)

    t=time[protein]/(len(phi[0])*5000)
    return (phi,psi),t

def get_traj_from_md(protein):
    protein_name=amino_acid_dict[protein[0]]+"_"+amino_acid_dict[protein[1]]
    traj=md.load("test/%s.xtc"%protein_name,top="diamine_pdbs/%s.pdb"%protein_name)

    phi,psi=md.compute_phi(traj)[1],md.compute_psi(traj)[1]

    with open("time_md_100.pkl","rb") as f:
        time=pickle.load(f)

    t=time[protein_name]/(len(phi[0])*100)
    return (phi,psi),t

def get_traj_from_wandb(protein,datatype,sigma=None):

    protein_name=amino_acid_dict[protein[0]]+"_"+amino_acid_dict[protein[1]]
    if datatype=="tbg" or datatype=="timewarp" or datatype=="uncapped":
        run_dict_sampling=pickle.load(open("run_dict_sampling_tbg.pkl","rb"))
    elif datatype=="md" or datatype=="capped" or datatype=="ours":
        run_dict_sampling=pickle.load(open("run_dict_sampling.pkl","rb"))

    runs=run_dict_sampling[protein_name]

    if sigma:
        runs_by_sigma=runs[sigma]
    else:
        runs_by_sigma=[]
        for key in runs.keys():
            runs_by_sigma+=runs[key]

    print("runs_by_sigma",runs_by_sigma)
    predicted_trajectories = {}
    for run_by_sig in runs_by_sigma:
        print(run_by_sig["run_id"])
        api = wandb.Api()
        run = api.run("vanib/jamun/runs/%s"%run_by_sig["run_id"])

        artifacts = run.logged_artifacts()


        structures = {}
        for artifact in artifacts:
            if artifact.type != "animated_trajectory_pdb":
                continue

            if "true_traj" not in artifact.name:
                continue

            name, version = artifact.name.split(":")
            name = name.replace("_animated_trajectory_pdb_true_traj", "")
            if name==protein_name:
                artifact_dir = artifact.download()
                structure = md.load(f"{artifact_dir}/animated_trajectory.pdb")
                structures[name] = structure
        predicted_traj=None
        if name not in structures.keys():
            print("no structure", run_by_sig["run_id"])
            continue
        maxversion=0
        for artifact in artifacts:
            if artifact.type != "predicted_samples":
                continue

            name, version = artifact.name.split(":")
            name = name.replace("_predicted_samples", "")

            if name==protein_name:
                if int(version[1:])>maxversion:
                    maxversion=int(version[1:])


        for artifact in artifacts:
            if artifact.type != "predicted_samples":
                        continue

            name, version = artifact.name.split(":")
            name = name.replace("_predicted_samples", "")

            if name==protein_name:
                if int(version[1:])==maxversion:

                    artifact_dir = artifact.download()

                    predicted_samples = np.load(f"{artifact_dir}/predicted_samples.npy")

                    predicted_traj = utils_md.coordinates_to_trajectories(predicted_samples, structures[name])
                    predicted_traj = md.join(predicted_traj, check_topology=True)


        if predicted_traj is None:
            print("no predicted traj", run_by_sig["run_id"])
            continue
        try:
            time=run.summary["sampler/total_sampling_time"]

        except:
            print("no time", run_by_sig["run_id"])
            continue


        phi,psi=md.compute_phi(predicted_traj)[1],md.compute_psi(predicted_traj)[1]

        predicted_trajectories[run_by_sig["run_id"]] = (phi,psi,time)

    return predicted_trajectories



#def get_traj_from_wandb_md(protein,sigma):

bins=np.linspace(-np.pi,np.pi,50)
def JS(protein):
    protein_name=amino_acid_dict[protein[0]]+"_"+amino_acid_dict[protein[1]]
    for chunk in md.iterload("test/%s.xtc"%protein_name,top="diamine_pdbs/%s.pdb"%protein_name,chunk=320000):
        full_md_traj=chunk
        break
    phis,psis=md.compute_phi(full_md_traj)[1],md.compute_psi(full_md_traj)[1]


    ref_hist,edges=np.histogramdd(np.hstack((phis,psis)),bins=[bins,bins,bins,bins])
    ref_hist=ref_hist/np.sum(ref_hist)

    traj_idx=1
    trajs=pickle.load(open("sampler_%s.pkl"%protein,"rb"))
    key=list(trajs.keys())[traj_idx]
    phi,psi,time=trajs[key]
    time_md_10=md_time_10_np[protein_name]
    required_steps=10*time/time_md_10
    alldata=np.hstack((phi,psi))

    alldata=np.vstack([alldata[i::32] for i in range(32)])
    js_ours=np.zeros(len(phi))
    stride=int(len(phi)/1001)
    for i in range(1,1001):
        our_hist,edges=np.histogramdd(alldata[:i*stride],bins=[bins,bins,bins,bins])
        our_hist=our_hist/np.sum(our_hist)
        js_ours[i-1]=compute_JS_divergence_of_ramachandran(our_hist,ref_hist)

    np.save("js_jamun_%s"%protein,js_ours)

    js_md=np.zeros(int(required_steps))
    for i in range(1,int(required_steps)+1):
        md_hist,edges=np.histogramdd(np.hstack((phis[:i],psis[:i])),bins=[bins,bins,bins,bins])
        md_hist=md_hist/np.sum(md_hist)
        js_md[i-1]=compute_JS_divergence_of_ramachandran(md_hist,ref_hist)

    np.save("js_md_%s"%protein,js_md)

    stride=int(len(phis)/1001)
    js_md_converged=np.zeros(1000)
    for i in range(1,1001):
        md_hist,edges=np.histogramdd(np.hstack((phis[:i*stride],psis[:i*stride])),bins=[bins,bins,bins,bins])
        md_hist=md_hist/np.sum(md_hist)
        js_md_converged[i-1]=compute_JS_divergence_of_ramachandran(md_hist,ref_hist)

    np.save("js_md_converged_%s"%protein,js_md_converged)


if __name__=="__main__":
    name=sys.argv[1]

    JS(name)
