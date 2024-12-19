import torch
import torch_geometric


class DataWithResidueInformation(torch_geometric.data.Data):
    """Graph with residue-level information."""

    pos: torch.Tensor
    atom_type_index: torch.Tensor
    atom_code_index: torch.Tensor
    residue_code_index: torch.Tensor
    residue_sequence_index: torch.Tensor
    residue_index: torch.Tensor # batched version of residue_sequence_index
    num_residues: int
    loss_weight: float

    def __inc__(self, key, value, *args, **kwargs):
        del value, args, kwargs
        if key in [
            "pos",
            "atom_type_index",
            "atom_code_index",
            "residue_code_index",
            "residue_sequence_index",
            "num_residues",
            "loss_weight",
        ]:
            return 0
        if key in ["edge_index"]:
            return self.num_nodes
        if key in ["residue_index"]:
            return self.num_residues
        raise NotImplementedError(f"key {key} not implemented")
