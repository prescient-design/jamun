import functools

import e3nn.util.test
import e3tools.nn
import optree
import pytest
import torch
import torch_geometric
import torch_geometric.data
from e3tools import radius_graph

import jamun
import jamun.data
import jamun.model
import jamun.model.arch
import jamun.model.embedding
from jamun.model.energy import model_predictions_f
from jamun.utils import ResidueMetadata

N_ATOM_TYPES = len(ResidueMetadata.ATOM_TYPES)
N_ATOM_CODES = len(ResidueMetadata.ATOM_CODES)
N_RESIDUE_CODES = len(ResidueMetadata.RESIDUE_CODES)

e3nn.set_optimization_defaults(jit_script_fx=False)


@pytest.fixture(scope="function")
def model():
    e3conv_net = jamun.model.arch.E3Conv(
        irreps_out="1x1e",
        irreps_hidden="120x0e + 32x1e",
        irreps_sh="1x0e + 1x1e",
        n_layers=1,
        hidden_layer_factory=e3tools.nn.SeparableConvBlock,
        output_head_factory=functools.partial(e3tools.nn.EquivariantMLP, irreps_hidden_list=["120x0e + 32x1e"]),
        radial_edge_embedder_factory=functools.partial(
            jamun.model.embedding.RadialEdgeEmbedder,
            radial_edge_attr_dim=32,
            basis="gaussian",
            cutoff=True,
            max_radius=1.0,
        ),
        bond_edge_embedder_factory=functools.partial(jamun.model.embedding.BondEdgeEmbedder, bond_edge_attr_dim=32),
        atom_embedder_factory=functools.partial(
            jamun.model.embedding.ResidueAtomEmbedder,
            atom_type_embedding_dim=8,
            atom_code_embedding_dim=8,
            residue_code_embedding_dim=32,
            residue_index_embedding_dim=8,
            use_residue_sequence_index=False,
            num_atom_types=20,
            max_sequence_length=10,
            num_atom_codes=10,
            num_residue_types=25,
        ),
    )
    return e3conv_net


@pytest.fixture(scope="function")
def data():
    N = 32

    batch = torch.zeros(N, dtype=torch.long)
    pos = torch.randn(N, 3)

    edge_index = radius_graph(pos, 1.0, batch=batch)
    bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long)
    atom_type_index = torch.randint(N_ATOM_TYPES, (N,))
    atom_code_index = torch.randint(N_ATOM_CODES, (N,))
    residue_code_index = torch.randint(N_RESIDUE_CODES, (N,))

    topology = torch_geometric.data.Data(
        edge_index=edge_index,
        bond_mask=bond_mask,
        atom_type_index=atom_type_index,
        atom_code_index=atom_code_index,
        residue_code_index=residue_code_index,
    )

    return pos, batch, topology


@pytest.mark.parametrize("device", [pytest.param(torch.device("cpu"), id="cpu")])
def test_e3conv_smoke(model, device, data):
    model.to(device)

    pos, batch, topology = optree.tree_map(lambda x: x.to(device), data)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    c_noise = torch.as_tensor([1.0], device=device)
    c_in = torch.as_tensor([1.0], device=device)

    out = model(pos, topology=topology, batch=batch, num_graphs=1, c_noise=c_noise, c_in=c_in)

    assert not torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("device", [pytest.param(torch.device("cpu"), id="cpu")])
def test_e3conv_equivariance(model, device, data):
    model.to(device)
    pos, batch, topology = optree.tree_map(lambda x: x.to(device), data)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    c_noise = torch.as_tensor([1.0], device=device)
    c_in = torch.as_tensor([1.0], device=device)

    e3nn.util.test.assert_equivariant(
        functools.partial(model, topology=topology, batch=batch, num_graphs=1, c_noise=c_noise, c_in=c_in),
        args_in=[pos],
        irreps_in=[model.irreps_out],
        irreps_out=[model.irreps_out],
    )


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda:0"),
            id="cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required"),
        ),
    ],
)
def test_e3conv_compile(model, device, data):
    torch.compiler.reset()
    model.to(device)

    pos, batch, topology = optree.tree_map(lambda x: x.to(device), data)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    c_noise = torch.as_tensor([1.0], device=device)
    c_in = torch.as_tensor([1.0], device=device)

    ref = model(pos, topology=topology, batch=batch, num_graphs=1, c_noise=c_noise, c_in=c_in)

    model_compiled = torch.compile(model, fullgraph=True)

    out = model_compiled(pos, topology=topology, batch=batch, num_graphs=1, c_noise=c_noise, c_in=c_in)

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("device", [pytest.param(torch.device("cpu"), id="cpu")])
def test_e3conv_energy_parameterization(model, device, data):
    torch.compiler.reset()
    model.to(device)

    pos, batch, topology = optree.tree_map(lambda x: x.to(device), data)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    c_noise = torch.as_tensor([1.0], device=device)
    c_in = torch.as_tensor([1.0], device=device)

    sigma = 0.5
    g = functools.partial(model, topology=topology, batch=batch, num_graphs=1, c_noise=c_noise, c_in=c_in)

    xhat_f = lambda y: model_predictions_f(y=y, batch=batch, num_graphs=1, sigma=sigma, g=g, energy_only=False)[0]  # noqa: E731
    energy_f = lambda y: model_predictions_f(y=y, batch=batch, num_graphs=1, sigma=sigma, g=g, energy_only=False)[1]  # noqa: E731
    s0 = -torch.func.jacrev(energy_f)(pos).squeeze(0)
    s1 = (xhat_f(pos) - pos) / (sigma**2)

    print(f"{(s0 - s1).abs().max()=}")

    torch.testing.assert_close(s0, s1)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu", marks=pytest.mark.xpass),
        pytest.param(
            torch.device("cuda:0"),
            id="cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required"),
        ),
    ],
)
def test_e3conv_energy_parameterization_compile(model, device, data):
    torch.compiler.reset()
    model.to(device)

    pos, batch, topology = optree.tree_map(lambda x: x.to(device), data)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    c_noise = torch.as_tensor([1.0], device=device)
    c_in = torch.as_tensor([1.0], device=device)

    g = functools.partial(model, topology=topology, batch=batch, num_graphs=1, c_noise=c_noise, c_in=c_in)

    sigma = 0.5
    xhat_ref = model_predictions_f(y=pos, batch=batch, num_graphs=1, sigma=sigma, g=g, energy_only=False)[0]  # noqa: E731

    assert not torch.equal(xhat_ref, torch.zeros_like(xhat_ref))

    xhat = torch.compile(model_predictions_f, fullgraph=True)(
        y=pos, batch=batch, num_graphs=1, sigma=sigma, g=g, energy_only=False
    )[0]

    print(f"{(xhat - xhat_ref).abs().max()=}")

    torch.testing.assert_close(xhat, xhat_ref)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu", marks=pytest.mark.xpass),
        pytest.param(
            torch.device("cuda:0"),
            id="cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required"),
        ),
    ],
)
def test_e3conv_energy_parameterization_double_backprop_compile(model, device, data):
    torch.compiler.reset()
    model.to(device)

    x, batch, topology = optree.tree_map(lambda x: x.to(device), data)

    with torch.no_grad():
        model.output_gain.copy_(torch.as_tensor(1.0, device=device))

    c_noise = torch.as_tensor([1.0], device=device)
    c_in = torch.as_tensor([1.0], device=device)

    g = functools.partial(model, topology=topology, batch=batch, num_graphs=1, c_noise=c_noise, c_in=c_in)

    sigma = 0.5
    y = x + torch.randn_like(x) * sigma

    xhat_f = lambda y: model_predictions_f(y=y, batch=batch, num_graphs=1, sigma=sigma, g=g, energy_only=False)[0]  # noqa: E731
    xhat = torch.compile(xhat_f, fullgraph=True)(y)

    loss = (x - xhat).pow(2).sum()
    loss.backward()

    grads_ref = torch.cat([p.grad.view(-1) for p in model.parameters()])
    assert not torch.equal(grads_ref, torch.zeros_like(grads_ref))

    for p in model.parameters():
        p.grad = None

    xhat = torch.compile(xhat_f, fullgraph=True)(y)
    loss = (x - xhat).pow(2).sum()
    loss.backward()

    grads = torch.cat([p.grad.view(-1) for p in model.parameters()])

    print(f"{(grads - grads_ref).abs().max()=}")

    torch.testing.assert_close(grads, grads_ref)
