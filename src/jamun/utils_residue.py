from typing import List, Union, Optional

import torch
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.transforms import BaseTransform
import torch_geometric


class ResidueMetadata:
    """Metadata for residue-level data."""

    ATOM_TYPES = ["C", "O", "N", "F", "S", "other"]
    ATOM_CODES = ["C", "O", "N", "S", "CA", "CB", "CH3"]
    RESIDUE_CODES = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "ACE",
        "NME",
    ]
    RESIDUE_LENGTHS = {
        "ALA": 5,
        "GLY": 4,
        "ACE": 3,
        "NME": 2,
    }
    RELATIVE_COORDS_PAD_LENGTH = 16

    @staticmethod
    def base_atom_code(residue_code: str) -> str:
        if residue_code in ["ACE", "NME"]:
            return "C"
        return "CA"


class DataWithResidueInformation(torch_geometric.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        del value, args, kwargs
        if key in [
            "pos",
            "atom_type_index",
            "atom_code_index",
            "residue_code_index",
            "residue_sequence_index",
            "x",
            "num_residues",
            "loss_weight",
        ]:
            return 0
        if key in ["edge_index"]:
            return self.num_nodes
        if key in ["residue_index", "residue_sequence_edge_index", "residue_spatial_edge_index"]:
            return self.num_residues
        raise NotImplementedError(f"key {key} not implemented")

    def total_num_residues(self) -> int:
        """Returns the total number of residues."""
        return self.num_residues.sum().item()

    def _base_index_and_residue_mask(self, residue_index: int) -> torch.Tensor:
        """Returns the base index and residue mask for a given residue index."""
        residue_mask = self.residue_index == residue_index
        residue_code_index = self.residue_code_index[residue_mask][0]
        residue_code = ResidueMetadata.RESIDUE_CODES[residue_code_index]
        residue_base_atom_code = ResidueMetadata.base_atom_code(residue_code)
        base_atom_code_index = ResidueMetadata.ATOM_CODES.index(residue_base_atom_code)
        base_index = torch.where((self.atom_code_index[residue_mask] == base_atom_code_index))[0]
        return base_index, residue_mask

    def base_coordinates(self) -> torch.Tensor:
        """Returns the base coordinates for each residue."""
        base_coordinates = []
        for residue_index in range(self.total_num_residues()):
            base_index, residue_mask = self._base_index_and_residue_mask(residue_index)
            base_coord = self.pos[residue_mask][base_index]
            base_coordinates.append(base_coord)
        return torch.cat(base_coordinates, dim=0)

    def relative_coordinates(self) -> torch.Tensor:
        """Returns the relative coordinates for each atom in each residue."""
        relative_coordinates = []
        for residue_index in range(self.total_num_residues()):
            base_index, residue_mask = self._base_index_and_residue_mask(residue_index)
            base_coord = self.pos[residue_mask][base_index]
            relative_coordinates_for_residue = self.pos[residue_mask] - base_coord
            relative_coordinates_for_residue = F.pad(
                relative_coordinates_for_residue,
                (0, 0, 0, ResidueMetadata.RELATIVE_COORDS_PAD_LENGTH - relative_coordinates_for_residue.shape[0]),
            )
            relative_coordinates.append(relative_coordinates_for_residue)
        return torch.stack(relative_coordinates, dim=0)

    def update_coordinates(
        self, base_coordinates: torch.Tensor, relative_coordinates: torch.Tensor
    ) -> torch_geometric.data.Data:
        """Updates the coordinates."""
        new_data = self.clone()
        for residue_index in range(self.total_num_residues()):
            base_index, residue_mask = self._base_index_and_residue_mask(residue_index)
            residue_code_index = new_data.residue_code_index[residue_mask][0]
            residue_code = ResidueMetadata.RESIDUE_CODES[residue_code_index]

            updated_coordinates = torch.zeros(
                (ResidueMetadata.RESIDUE_LENGTHS[residue_code], 3),
                dtype=base_coordinates.dtype,
                device=base_coordinates.device,
            )
            residue_base_coordinates = base_coordinates[residue_index]
            residue_relative_coordinates = relative_coordinates[residue_index]
            assert residue_relative_coordinates.shape == updated_coordinates.shape

            for atom_index in range(updated_coordinates.shape[0]):
                if atom_index == base_index:
                    updated_coordinates[atom_index] = residue_base_coordinates
                else:
                    updated_coordinates[atom_index] = (
                        residue_base_coordinates + residue_relative_coordinates[atom_index]
                    )
            new_data.pos[residue_mask] = updated_coordinates
        return new_data

    def atom_types(self) -> int:
        """Returns the atom types for each atom."""
        return [ResidueMetadata.ATOM_TYPES[index] for index in self.atom_type_index]

    def atom_codes(self) -> int:
        """Returns the atom codes for each atom."""
        return [ResidueMetadata.ATOM_CODES[index] for index in self.atom_code_index]

    def atom_types_per_residue(self) -> List[List[str]]:
        """Returns the atom types for each atom in each residue."""
        atom_types = []
        for residue_index in range(self.total_num_residues()):
            _, residue_mask = self._base_index_and_residue_mask(residue_index)
            residue_atom_type_index = self.atom_type_index[residue_mask]
            residue_atom_types = [ResidueMetadata.ATOM_TYPES[index] for index in residue_atom_type_index]
            atom_types.append(residue_atom_types)
        return atom_types

    def atom_codes_per_residue(self) -> List[List[str]]:
        """Returns the atom codes for each atom in each residue."""
        atom_codes = []
        for residue_index in range(self.total_num_residues()):
            residue_mask = self.residue_sequence_index == residue_index
            residue_atom_code_index = self.atom_code_index[residue_mask]
            residue_atom_codes = [ResidueMetadata.ATOM_CODES[index] for index in residue_atom_code_index]
            atom_codes.append(residue_atom_codes)
        return atom_codes

    def residue_code_indices(self) -> torch.Tensor:
        """Returns the residue code indices for each residue."""
        residue_code_indices = []
        for residue_index in range(self.total_num_residues()):
            _, residue_mask = self._base_index_and_residue_mask(residue_index)
            residue_code_index = self.residue_code_index[residue_mask][0]
            residue_code_indices.append(residue_code_index)
        residue_code_indices = torch.tensor(residue_code_indices, device=self.pos.device, dtype=torch.long)
        return residue_code_indices

    def compute_residue_sequence_edges(self) -> torch.Tensor:
        """Computes edges between residues according to sequence adjacency."""
        sequence_edge_index = []
        for graph_index, num_residues_per_graph in enumerate(self.num_residues):
            offset = self.num_residues[:graph_index].sum().item()
            for i in range(offset, offset + num_residues_per_graph - 1):
                sequence_edge_index.append([i, i + 1])
                sequence_edge_index.append([i + 1, i])
        sequence_edge_index = torch.tensor(sequence_edge_index, device=self.pos.device)
        sequence_edge_index = sequence_edge_index.t().contiguous()
        assert sequence_edge_index.shape[-2] == 2, sequence_edge_index.shape
        return sequence_edge_index

    def compute_residue_spatial_edges(self, radial_cutoff: float) -> torch.Tensor:
        """Computes edges between residues according to spatial adjacency between base coordinates."""
        spatial_edge_index = []
        base_coordinates = self.base_coordinates()
        for graph_index, num_residues_per_graph in enumerate(self.num_residues):
            offset = self.num_residues[:graph_index].sum().item()
            for i in range(offset, offset + num_residues_per_graph):
                for j in range(i + 1, offset + num_residues_per_graph):
                    distance = torch.norm(base_coordinates[i] - base_coordinates[j])
                    if distance <= radial_cutoff:
                        spatial_edge_index.append([i, j])
                        spatial_edge_index.append([j, i])
        spatial_edge_index = torch.tensor(spatial_edge_index, device=self.pos.device)
        spatial_edge_index = spatial_edge_index.t().contiguous()
        assert spatial_edge_index.shape[-2] == 2, spatial_edge_index.shape
        return spatial_edge_index


class AddResidueInformation(BaseTransform):
    """Adds residue-level data; see ResidueMetadata for parsing indices."""

    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Convert atom-level data to residue-level data."""
        data = DataWithResidueInformation(
            atom_type_index=data.x[:, 0],
            residue_code_index=data.x[:, 1],
            residue_sequence_index=data.x[:, 2],
            atom_code_index=data.x[:, 3],
            **data,
        )
        data.residue_index = data.residue_sequence_index
        data.num_residues = data.residue_sequence_index.max().item() + 1
        return data
