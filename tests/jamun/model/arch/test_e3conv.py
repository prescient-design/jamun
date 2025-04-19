import functools
from typing import Callable

import e3nn.util.test
import e3tools.nn
import pytest
import torch
import torch_geometric
import torch_geometric.data
from e3tools import radius_graph
from torch import Tensor

import jamun
import jamun.data
import jamun.model
import jamun.model.arch
from jamun.utils import ResidueMetadata

N_ATOM_TYPES = len(ResidueMetadata.ATOM_TYPES)
N_ATOM_CODES = len(ResidueMetadata.ATOM_CODES)
N_RESIDUE_CODES = len(ResidueMetadata.RESIDUE_CODES)

e3nn.set_optimization_defaults(jit_script_fx=False)


@pytest.fixture(scope="function")
def model():
    m = jamun.model.arch.E3Conv(
        irreps_out="1x1e",
        irreps_hidden="120x0e + 32x1e",
        irreps_sh="1x0e + 1x1e",
        n_layers=1,
        edge_attr_dim=8,
        atom_type_embedding_dim=8,
        atom_code_embedding_dim=8,
        residue_code_embedding_dim=32,
        residue_index_embedding_dim=8,
        use_residue_information=False,
        use_residue_sequence_index=False,
        hidden_layer_factory=functools.partial(
            e3tools.nn.ConvBlock,
            conv=e3tools.nn.Conv,
        ),
        output_head_factory=functools.partial(e3tools.nn.EquivariantMLP, irreps_hidden_list=["120x0e + 32x1e"]),
    )
    return m


@pytest.mark.parametrize("device", [pytest.param(torch.device("cpu"), id="cpu")])
def test_e3conv_smoke(model, device):
    model.test_equivariance = True
    model.to(device)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    N = 32

    batch = torch.zeros(N, dtype=torch.long, device=device)
    pos = torch.randn(N, 3, device=device)
    edge_index = radius_graph(pos, 1.0, batch=batch)
    bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    atom_type_index = torch.randint(N_ATOM_TYPES, (N,))
    atom_code_index = torch.randint(N_ATOM_CODES, (N,))

    c_noise = torch.as_tensor([1.0], device=device)
    effective_radial_cutoff = 1.0

    topology = torch_geometric.data.Data(
        edge_index=edge_index, bond_mask=bond_mask, atom_type_index=atom_type_index, atom_code_index=atom_code_index
    )

    out = model(pos, topology=topology, c_noise=c_noise, effective_radial_cutoff=effective_radial_cutoff)

    assert not torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("device", [pytest.param(torch.device("cpu"), id="cpu")])
def test_e3conv_equivariance(model, device):
    model.to(device)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    N = 32

    batch = torch.zeros(N, dtype=torch.long, device=device)
    pos = torch.randn(N, 3, device=device)
    edge_index = radius_graph(pos, 1.0, batch=batch)
    bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    atom_type_index = torch.randint(N_ATOM_TYPES, (N,))
    atom_code_index = torch.randint(N_ATOM_CODES, (N,))

    c_noise = torch.as_tensor([1.0], device=device)
    effective_radial_cutoff = 1.0

    topology = torch_geometric.data.Data(
        edge_index=edge_index, bond_mask=bond_mask, atom_type_index=atom_type_index, atom_code_index=atom_code_index
    )

    e3nn.util.test.assert_equivariant(
        functools.partial(model, topology=topology, c_noise=c_noise, effective_radial_cutoff=effective_radial_cutoff),
        args_in=[pos],
        irreps_in=[model.irreps_out],
        irreps_out=[model.irreps_out],
    )


@pytest.mark.parametrize("device", [pytest.param(torch.device("cpu"), id="cpu")])
def test_e3conv_compile(model, device):
    torch.compiler.reset()
    model.to(device)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    N = 32

    batch = torch.zeros(N, dtype=torch.long, device=device)
    pos = torch.randn(N, 3, device=device)
    edge_index = radius_graph(pos, 1.0, batch=batch)
    bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    atom_type_index = torch.randint(N_ATOM_TYPES, (N,))
    atom_code_index = torch.randint(N_ATOM_CODES, (N,))

    c_noise = torch.as_tensor([1.0], device=device)
    effective_radial_cutoff = 1.0

    topology = torch_geometric.data.Data(
        edge_index=edge_index, bond_mask=bond_mask, atom_type_index=atom_type_index, atom_code_index=atom_code_index
    )

    ref = model(pos, topology=topology, c_noise=c_noise, effective_radial_cutoff=effective_radial_cutoff)

    model_compiled = torch.compile(model, fullgraph=True)

    out = model_compiled(pos, topology=topology, c_noise=c_noise, effective_radial_cutoff=effective_radial_cutoff)

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("device", [pytest.param(torch.device("cpu"), id="cpu")])
def test_e3conv_energy_parameterization(model, device):
    torch.compiler.reset()
    model.to(device)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    N = 32

    batch = torch.zeros(N, dtype=torch.long, device=device)
    pos = torch.randn(N, 3, device=device)
    edge_index = radius_graph(pos, 1.0, batch=batch)
    bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    atom_type_index = torch.randint(N_ATOM_TYPES, (N,))
    atom_code_index = torch.randint(N_ATOM_CODES, (N,))

    c_noise = torch.as_tensor([1.0], device=device)
    effective_radial_cutoff = 1.0

    topology = torch_geometric.data.Data(
        edge_index=edge_index, bond_mask=bond_mask, atom_type_index=atom_type_index, atom_code_index=atom_code_index
    )

    def energy_f(y: Tensor, g: Callable, sigma) -> Tensor:
        return (g(y) - y).pow(2).sum() / (2 * (sigma**2))

    def xhat_f(y: Tensor, g: Callable) -> Tensor:
        # NOTE g must be Tensor to Tensor
        g_y, vjp_func = torch.func.vjp(g, y)
        return g_y - vjp_func(g_y - y, create_graph=True, retain_graph=True)[0]

    sigma = 0.5
    g = functools.partial(model, topology=topology, c_noise=c_noise, effective_radial_cutoff=effective_radial_cutoff)

    s0 = -torch.func.jacrev(energy_f)(pos, g, sigma)
    s1 = (xhat_f(pos, g) - pos) / (sigma**2)

    print(f"{(s0 - s1).abs().max()=}")

    torch.testing.assert_close(s0, s1)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu", marks=pytest.mark.xfail),
        pytest.param(torch.device("cuda:0"), id="cuda"),
    ],
)
def test_e3conv_energy_parameterization_compile(model, device):
    torch.compiler.reset()
    model.to(device)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    N = 32

    batch = torch.zeros(N, dtype=torch.long, device=device)
    pos = torch.randn(N, 3, device=device)
    edge_index = radius_graph(pos, 1.0, batch=batch)
    bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    atom_type_index = torch.randint(N_ATOM_TYPES, (N,))
    atom_code_index = torch.randint(N_ATOM_CODES, (N,))

    c_noise = torch.as_tensor([1.0], device=device)
    effective_radial_cutoff = 1.0

    topology = torch_geometric.data.Data(
        edge_index=edge_index, bond_mask=bond_mask, atom_type_index=atom_type_index, atom_code_index=atom_code_index
    ).to(device)

    def energy_f(y: Tensor, g: Callable, sigma) -> Tensor:
        return (g(y) - y).pow(2).sum() / (2 * (sigma**2))

    def xhat_f(y: Tensor, g: Callable) -> Tensor:
        # NOTE g must be Tensor to Tensor
        g_y, vjp_func = torch.func.vjp(g, y)
        return g_y - vjp_func(g_y - y, create_graph=True, retain_graph=True)[0]

    g = functools.partial(model, topology=topology, c_noise=c_noise, effective_radial_cutoff=effective_radial_cutoff)

    xhat_ref = xhat_f(pos, g)

    assert not torch.equal(xhat_ref, torch.zeros_like(xhat_ref))

    xhat = torch.compile(xhat_f, fullgraph=True)(pos, g)

    print(f"{(xhat - xhat_ref).abs().max()=}")

    torch.testing.assert_close(xhat, xhat_ref)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu", marks=pytest.mark.xfail),
        pytest.param(torch.device("cuda:0"), id="cuda"),
    ],
)
def test_e3conv_energy_parameterization_double_backprop_compile(model, device):
    torch.compiler.reset()
    model.to(device)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    N = 32

    batch = torch.zeros(N, dtype=torch.long, device=device)
    x = torch.randn(N, 3, device=device)
    edge_index = radius_graph(x, 1.0, batch=batch)
    bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    atom_type_index = torch.randint(N_ATOM_TYPES, (N,))
    atom_code_index = torch.randint(N_ATOM_CODES, (N,))

    c_noise = torch.as_tensor([1.0], device=device)
    effective_radial_cutoff = 1.0

    topology = torch_geometric.data.Data(
        edge_index=edge_index, bond_mask=bond_mask, atom_type_index=atom_type_index, atom_code_index=atom_code_index
    ).to(device)

    def energy_f(y: Tensor, g: Callable, sigma) -> Tensor:
        return (g(y) - y).pow(2).sum() / (2 * (sigma**2))

    def xhat_f(y: Tensor, g: Callable) -> Tensor:
        # NOTE g must be Tensor to Tensor
        g_y, vjp_func = torch.func.vjp(g, y)
        return g_y - vjp_func(g_y - y, create_graph=True, retain_graph=True)[0]

    g = functools.partial(model, topology=topology, c_noise=c_noise, effective_radial_cutoff=effective_radial_cutoff)

    y = x + torch.randn_like(x)

    xhat = xhat_f(y, g)

    loss = (x - xhat).pow(2).sum()
    loss.backward()

    grads_ref = torch.cat([p.grad.view(-1) for p in model.parameters()])
    assert not torch.equal(grads_ref, torch.zeros_like(grads_ref))

    for p in model.parameters():
        p.grad = None

    xhat = torch.compile(xhat_f, fullgraph=True)(y, g)
    loss = (x - xhat).pow(2).sum()
    loss.backward()

    grads = torch.cat([p.grad.view(-1) for p in model.parameters()])

    print(f"{(grads - grads_ref).abs().max()=}")

    torch.testing.assert_close(grads, grads_ref)
