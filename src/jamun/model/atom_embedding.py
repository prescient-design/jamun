import torch
import torch.nn as nn
import torch_geometric
import e3nn


from jamun import utils


class CoarseGrainedBeadEmbedding(nn.Module):
    """Embed coarse-grained beads."""

    def __init__(self, bead_embedding_dim: int, num_beads: int = 10):
        super().__init__()
        self.embedding = nn.Embedding(num_beads, bead_embedding_dim)
        self.irreps_out = e3nn.o3.Irreps(f"{bead_embedding_dim}x0e")

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return self.embedding(data.x)


class SimpleAtomEmbedding(nn.Module):
    """Embed atoms without residue information."""

    def __init__(self, embedding_dim: int, max_value: int = 20):
        super().__init__()
        self.embedding = nn.Embedding(max_value, embedding_dim)
        self.irreps_out = e3nn.o3.Irreps(f"{embedding_dim}x0e")

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return self.embedding(data.atom_type_index)


class AtomEmbeddingWithResidueInformation(nn.Module):
    """Embed atoms with residue information."""

    def __init__(
        self,
        atom_type_embedding_dim: int,
        atom_code_embedding_dim: int,
        residue_code_embedding_dim: int,
        residue_index_embedding_dim: int,
        use_residue_sequence_index: bool,
        num_atom_types: int = 20,
        max_sequence_length: int = 10,
        num_atom_codes: int = 10,
        num_residue_types: int = 25,
    ):
        super().__init__()
        self.atom_type_embedding = torch.nn.Embedding(num_atom_types, atom_type_embedding_dim)
        self.atom_code_embedding = torch.nn.Embedding(num_atom_codes, atom_code_embedding_dim)
        self.residue_code_embedding = torch.nn.Embedding(num_residue_types, residue_code_embedding_dim)
        self.residue_index_embedding = torch.nn.Embedding(max_sequence_length, residue_index_embedding_dim)
        self.use_residue_sequence_index = use_residue_sequence_index
        self.irreps_out = e3nn.o3.Irreps(
            f"{atom_type_embedding_dim}x0e + {atom_type_embedding_dim}x0o + {residue_code_embedding_dim}x0e + {residue_index_embedding_dim}x0e"
        )

    def forward(self, data: utils.DataWithResidueInformation) -> torch.Tensor:
        features = []
        atom_type_embedded = self.atom_type_embedding(data.atom_type_index)
        features.append(atom_type_embedded)

        atom_code_embedded = self.atom_code_embedding(data.atom_code_index)
        features.append(atom_code_embedded)

        residue_code_embedded = self.residue_code_embedding(data.residue_code_index)
        features.append(residue_code_embedded)

        residue_sequence_index = data.residue_sequence_index
        if not self.use_residue_sequence_index:
            residue_sequence_index = torch.zeros_like(residue_sequence_index)
        residue_sequence_index_embedded = self.residue_index_embedding(residue_sequence_index)
        features.append(residue_sequence_index_embedded)

        features = torch.cat(features, dim=-1)
        return features
