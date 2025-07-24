"""Converts a PyTorch Geometric Batch object to an AtomGraphs object for the Orb models."""

import torch
import torch_geometric.data
from orb_models.forcefield.base import AtomGraphs


def to_atom_graphs(
    topology: torch_geometric.data.Batch,
) -> AtomGraphs:
    """
    Maps a batched PyTorch Geometric DataWithResidueInformation object
    (which is a torch_geometric.data.Batch) to an AtomGraphs object.

    Args:
        x (torch_geometric.data.Batch): The input PyTorch Geometric Batch object,
                                                 expected to be a batch of DataWithResidueInformation.

    Returns:
        AtomGraphs: An AtomGraphs object populated with data from the PyG Batch.
    """
    senders = topology.edge_index[0]
    receivers = topology.edge_index[1]

    batch = topology.get("batch", None)
    ptr = topology.get("ptr", None)
    if batch is None:
        batch = torch.zeros(topology.num_nodes, dtype=torch.long)
        ptr = torch.arange(0, topology.num_nodes + 1, dtype=torch.long)

    edge_graph_idx = batch[senders]
    num_graphs = topology.get("num_graphs", batch.max().item() + 1)
    n_node = ptr[1:] - ptr[:-1]
    n_edge = torch.bincount(edge_graph_idx, minlength=num_graphs)

    node_features: dict[str, torch.Tensor] = {}
    node_features["positions"] = topology.pos

    if topology.get("atom_type_index", None) is not None:
        node_features["atomic_numbers"] = topology.atom_type_index
        node_features["atomic_numbers_embedding"] = topology.atom_type_index

    # Other residue-specific node features
    if topology.get("atom_code_index", None) is not None:
        node_features["atom_code_index"] = topology.atom_code_index
    if topology.get("residue_code_index", None) is not None:
        node_features["residue_code_index"] = topology.residue_code_index
    if topology.get("residue_sequence_index", None) is not None:
        node_features["residue_sequence_index"] = topology.residue_sequence_index
    if topology.get("residue_index", None) is not None:
        node_features["residue_index"] = topology.residue_index

    edge_features: dict[str, torch.Tensor] = {}
    edge_features["vectors"] = topology.pos[senders] - topology.pos[receivers]
    if topology.get("edge_attr", None) is not None:
        edge_features["edge_attr"] = topology.edge_attr

    system_features: dict[str, torch.Tensor] = {}
    if topology.get("num_residues", None) is not None:
        system_features["num_residues"] = topology.num_residues

    if topology.get("loss_weight", None) is not None:
        system_features["loss_weight"] = topology.loss_weight

    return AtomGraphs(
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        node_features=node_features,
        edge_features=edge_features,
        system_features=system_features,
        node_targets=None,
        edge_targets=None,
        system_targets=None,
        system_id=None,
        fix_atoms=None,
        tags=None,
        radius=None,
        max_num_neighbors=None,
        half_supercell=False,
    )
