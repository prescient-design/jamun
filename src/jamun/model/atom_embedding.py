import e3nn
import torch
import torch.nn as nn
import torch_geometric

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
        num_atom_types: int,
        max_sequence_length: int,
        num_atom_codes: int,
        num_residue_types: int,
        use_residue_chirality: bool = False,
        residue_chirality_embedding_dim: int = 0,
        num_chirality_types: int = 2,
    ):
        super().__init__()
        self.atom_type_embedding = torch.nn.Embedding(num_atom_types, atom_type_embedding_dim)
        self.atom_code_embedding = torch.nn.Embedding(num_atom_codes, atom_code_embedding_dim)
        self.residue_code_embedding = torch.nn.Embedding(num_residue_types, residue_code_embedding_dim)
        self.residue_index_embedding = torch.nn.Embedding(max_sequence_length, residue_index_embedding_dim)
        self.use_residue_sequence_index = use_residue_sequence_index
        self.use_residue_chirality = use_residue_chirality
        
        # Add chirality embedding if requested
        if self.use_residue_chirality:
            if residue_chirality_embedding_dim <= 0:
                raise ValueError("residue_chirality_embedding_dim must be positive when use_residue_chirality=True")
            self.residue_chirality_embedding = torch.nn.Embedding(num_chirality_types, residue_chirality_embedding_dim)
        
        # Build irreps_out string based on whether chirality is used
        irreps_parts = [
            f"{atom_type_embedding_dim}x0e",
            f"{atom_code_embedding_dim}x0e", 
            f"{residue_code_embedding_dim}x0e",
            f"{residue_index_embedding_dim}x0e"
        ]
        if self.use_residue_chirality:
            irreps_parts.append(f"{residue_chirality_embedding_dim}x0e")
        
        self.irreps_out = e3nn.o3.Irreps(" + ".join(irreps_parts))

    def forward(self, data) -> torch.Tensor:
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

        # Add chirality embedding if enabled and available in data
        if self.use_residue_chirality:
            if hasattr(data, 'residue_chirality_index'):
                residue_chirality_embedded = self.residue_chirality_embedding(data.residue_chirality_index)
                features.append(residue_chirality_embedded)
            else:
                raise ValueError("use_residue_chirality=True but data does not have residue_chirality_index attribute")

        features = torch.cat(features, dim=-1)
        return features
