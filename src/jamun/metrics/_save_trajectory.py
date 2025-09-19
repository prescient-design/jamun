import os
from typing import Dict, Union

import numpy as np
import wandb
from lightning.pytorch.utilities import rank_zero_only

from jamun import utils
from rdkit import Chem
from jamun.metrics._utils import TrajectoryMetric


class SaveTrajectory(TrajectoryMetric):
    """A metric that saves the predicted and true samples."""

    def __init__(self, save_true_trajectory: bool = False, timeseries_format: str = "dcd", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = os.path.join("sampler", self.dataset.label())
        self.pred_samples_dir = os.path.join(self.output_dir, "predicted_samples")
        self.true_samples_dir = os.path.join(self.output_dir, "true_samples")

        # Create the output directories.
        self.save_true_trajectory = save_true_trajectory
        # Determine which time-series formats to write
        if timeseries_format not in ("dcd", "sdf", "both"):
            raise ValueError(f"Invalid timeseries_format: {timeseries_format}. Choose from 'dcd', 'sdf', 'both'.")
        self._timeseries_exts = ["dcd"] if timeseries_format == "dcd" else (["sdf"] if timeseries_format == "sdf" else ["dcd", "sdf"])
        if self.save_true_trajectory:
            self.true_samples_extensions = ["pdb", *self._timeseries_exts]
            for ext in self.true_samples_extensions:
                os.makedirs(os.path.join(self.true_samples_dir, ext), exist_ok=True)

        self.pred_samples_extensions = ["pdb", *self._timeseries_exts]
        for ext in self.pred_samples_extensions:
            os.makedirs(os.path.join(self.pred_samples_dir, ext), exist_ok=True)

        # Precompute atomic numbers and bonds for SDF writing (no MDTraj dependency)
        self._atomic_numbers = None
        self._bonds = None
        self._reference_mol = None
        try:
            source_graph = getattr(self.dataset, "graph", None)
            if source_graph is None:
                source_graph = getattr(self.dataset, "data", None)
            if source_graph is not None:
                pt = Chem.GetPeriodicTable()
                atom_type_index = source_graph.atom_type_index.tolist()
                atomic_symbols = [utils.ResidueMetadata.ATOM_TYPES[i] for i in atom_type_index]
                self._atomic_numbers = [pt.GetAtomicNumber(sym) for sym in atomic_symbols]
                edge_index = source_graph.edge_index
                bonds = set()
                for i, j in edge_index.T.tolist():
                    if i == j:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    bonds.add((a, b))
                self._bonds = sorted(bonds)
            
            # Try to extract reference molecule for bond order information
            if hasattr(self.dataset, 'rdkit_mol'):
                self._reference_mol = self.dataset.rdkit_mol
            elif hasattr(self.dataset, 'rdkit_mol_withH'):
                self._reference_mol = self.dataset.rdkit_mol_withH
                
        except Exception:
            # Leave None; will error later if SDF requested and we cannot infer
            pass

    def filename_pred(self, trajectory_index: Union[int, str], extension: str) -> str:
        """Returns the filename for the predicted samples."""
        if extension not in self.pred_samples_extensions:
            raise ValueError(f"Invalid extension: {extension}")
        filenames = {
            "npy": os.path.join(self.pred_samples_dir, "npy", f"{trajectory_index}.npy"),
            "pdb": os.path.join(self.pred_samples_dir, "pdb", f"{trajectory_index}.pdb"),
            "dcd": os.path.join(self.pred_samples_dir, "dcd", f"{trajectory_index}.dcd"),
            "sdf": os.path.join(self.pred_samples_dir, "sdf", f"{trajectory_index}.sdf"),
        }
        return filenames[extension]

    def filename_true(self, trajectory_index: Union[int, str], extension: str) -> str:
        """Returns the filename for the true samples."""
        if extension not in self.true_samples_extensions:
            raise ValueError(f"Invalid extension: {extension}")
        filenames = {
            "pdb": os.path.join(self.true_samples_dir, "pdb", f"{trajectory_index}.pdb"),
            "dcd": os.path.join(self.true_samples_dir, "dcd", f"{trajectory_index}.dcd"),
            "sdf": os.path.join(self.true_samples_dir, "sdf", f"{trajectory_index}.sdf"),
        }
        return filenames[extension]

    def on_sample_start(self):
        # Save topology from the true trajectory.
        true_trajectory = self.dataset.trajectory
        utils.save_pdb(true_trajectory[0], os.path.join(self.output_dir, "topology.pdb"))

        if not self.save_true_trajectory:
            return

        utils.save_pdb(true_trajectory, self.filename_true(0, "pdb"))
        if "dcd" in self._timeseries_exts:
            true_trajectory.save_dcd(self.filename_true(0, "dcd"))
        if "sdf" in self._timeseries_exts:
            # Convert true trajectory to coords (num_atoms, num_frames, 3)
            coords = np.transpose(true_trajectory.xyz, (1, 0, 2))
            # Use cached metadata or derive from dataset.{graph|data}
            if self._atomic_numbers is None or self._bonds is None:
                source_graph = getattr(self.dataset, "graph", None)
                if source_graph is None:
                    source_graph = getattr(self.dataset, "data", None)
                if source_graph is None:
                    raise AttributeError("Cannot write SDF: missing atomic_numbers/bonds metadata from dataset.")
                pt = Chem.GetPeriodicTable()
                atom_type_index = source_graph.atom_type_index.tolist()
                atomic_symbols = [utils.ResidueMetadata.ATOM_TYPES[i] for i in atom_type_index]
                self._atomic_numbers = [pt.GetAtomicNumber(sym) for sym in atomic_symbols]
                edge_index = source_graph.edge_index
                bonds = set()
                for i, j in edge_index.T.tolist():
                    if i == j:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    bonds.add((a, b))
                self._bonds = sorted(bonds)
            
            # Try to get reference molecule if not already cached
            if self._reference_mol is None:
                if hasattr(self.dataset, 'rdkit_mol'):
                    self._reference_mol = self.dataset.rdkit_mol
                elif hasattr(self.dataset, 'rdkit_mol_withH'):
                    self._reference_mol = self.dataset.rdkit_mol_withH
                    
            utils.save_sdf_from_coords(coords, self._atomic_numbers, self._bonds, self.filename_true(0, "sdf"), self._reference_mol)

    def on_sample_end(self):
        if rank_zero_only.rank != 0:
            return

        # Save the joined samples at the very end of sampling to wandb.
        label = self.dataset.label()
        label = label.replace("/", "_").replace("=", "-")

        # for ext in self.pred_samples_extensions:
        #     filename = self.filename_pred("joined", ext)
        #     artifact = wandb.Artifact(f"{label}_pred_samples_joined", type="pred_samples_joined")
        #     artifact.add_file(filename, f"pred_samples_joined.{ext}")
        #     wandb.log_artifact(artifact)

    def compute(self) -> Dict[str, float]:
        # Save the predicted samples as numpy files.
        # samples_np = self.sample_tensors(new=True).cpu().detach().numpy()
        # for trajectory_index, sample in enumerate(samples_np):
        #     np.save(self.filename_pred(trajectory_index, "npy"), sample)

        # samples_joined_np = self.joined_sample_tensor().cpu().detach().numpy()
        # np.save(self.filename_pred("joined", "npy"), samples_joined_np)

        # Save the predicted sample trajectory in the selected formats.
        if "dcd" in self._timeseries_exts:
            pred_trajectories = self.sample_trajectories(new=True)
            for trajectory_index, pred_trajectory in enumerate(pred_trajectories, start=self.num_chains_seen):
                pred_trajectory.save_dcd(self.filename_pred(trajectory_index, "dcd"))
        if "sdf" in self._timeseries_exts:
            # Ensure cached metadata present or derive from dataset.{graph|data}
            if self._atomic_numbers is None or self._bonds is None:
                source_graph = getattr(self.dataset, "graph", None)
                if source_graph is None:
                    source_graph = getattr(self.dataset, "data", None)
                if source_graph is None:
                    raise AttributeError("Cannot write SDF: missing atomic_numbers/bonds metadata from dataset.")
                pt = Chem.GetPeriodicTable()
                atom_type_index = source_graph.atom_type_index.tolist()
                atomic_symbols = [utils.ResidueMetadata.ATOM_TYPES[i] for i in atom_type_index]
                self._atomic_numbers = [pt.GetAtomicNumber(sym) for sym in atomic_symbols]
                edge_index = source_graph.edge_index
                bonds = set()
                for i, j in edge_index.T.tolist():
                    if i == j:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    bonds.add((a, b))
                self._bonds = sorted(bonds)
            
            # Try to get reference molecule if not already cached
            if self._reference_mol is None:
                if hasattr(self.dataset, 'rdkit_mol'):
                    self._reference_mol = self.dataset.rdkit_mol
                elif hasattr(self.dataset, 'rdkit_mol_withH'):
                    self._reference_mol = self.dataset.rdkit_mol_withH
                    
            pred_coords = self.sample_tensors(new=True)  # (batch_size, num_atoms, num_frames, 3)
            # Iterate each chain and write SDF from coords directly.
            for trajectory_index, coords in enumerate(pred_coords, start=self.num_chains_seen):
                utils.save_sdf_from_coords(coords, self._atomic_numbers, self._bonds, self.filename_pred(trajectory_index, "sdf"), self._reference_mol)

        if "dcd" in self._timeseries_exts:
            pred_trajectory_joined = self.joined_sample_trajectory()
            pred_trajectory_joined.save_dcd(self.filename_pred("joined", "dcd"))
        if "sdf" in self._timeseries_exts:
            # Ensure cached metadata present or derive from dataset.{graph|data}
            if self._atomic_numbers is None or self._bonds is None:
                source_graph = getattr(self.dataset, "graph", None)
                if source_graph is None:
                    source_graph = getattr(self.dataset, "data", None)
                if source_graph is None:
                    raise AttributeError("Cannot write SDF: missing atomic_numbers/bonds metadata from dataset.")
                pt = Chem.GetPeriodicTable()
                atom_type_index = source_graph.atom_type_index.tolist()
                atomic_symbols = [utils.ResidueMetadata.ATOM_TYPES[i] for i in atom_type_index]
                self._atomic_numbers = [pt.GetAtomicNumber(sym) for sym in atomic_symbols]
                edge_index = source_graph.edge_index
                bonds = set()
                for i, j in edge_index.T.tolist():
                    if i == j:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    bonds.add((a, b))
                self._bonds = sorted(bonds)
            
            # Try to get reference molecule if not already cached
            if self._reference_mol is None:
                if hasattr(self.dataset, 'rdkit_mol'):
                    self._reference_mol = self.dataset.rdkit_mol
                elif hasattr(self.dataset, 'rdkit_mol_withH'):
                    self._reference_mol = self.dataset.rdkit_mol_withH
                    
            joined_coords = self.joined_sample_tensor()  # (num_atoms, num_frames, 3)
            utils.save_sdf_from_coords(joined_coords, self._atomic_numbers, self._bonds, self.filename_pred("joined", "sdf"), self._reference_mol)

        return {}
