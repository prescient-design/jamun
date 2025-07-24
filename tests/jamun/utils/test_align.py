import pytest
import torch

from jamun.utils.align import align_A_to_B_batched_f


def test_perfect_alignment():
    y = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    batch = torch.tensor([0, 0])
    num_graphs = 1
    y_aligned = align_A_to_B_batched_f(y, x, batch, num_graphs)
    assert torch.allclose(y_aligned, x, atol=1e-6)


def test_simple_translation():
    y = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    x = torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.float32)
    batch = torch.tensor([0, 0])
    num_graphs = 1
    y_aligned = align_A_to_B_batched_f(y, x, batch, num_graphs)
    assert torch.allclose(y_aligned, x, atol=1e-6)


def test_simple_rotation():
    y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    x = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    batch = torch.tensor([0, 0])
    num_graphs = 1
    y_aligned = align_A_to_B_batched_f(y, x, batch, num_graphs)
    assert torch.allclose(y_aligned, x, atol=1e-6)


def test_combined_rotation_translation():
    y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    x = torch.tensor(
        [[1.0, 1.0, 1.0], [0.0, 2.0, 1.0]], dtype=torch.float32
    )  # Rotated by 90 deg around z, then translated by [1,1,1]
    batch = torch.tensor([0, 0])
    num_graphs = 1
    y_aligned = align_A_to_B_batched_f(y, x, batch, num_graphs)
    assert torch.allclose(y_aligned, x, atol=1e-6)


def test_reflection_case():
    y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32
    )  # Reflected across xy plane
    batch = torch.tensor([0, 0, 0])
    num_graphs = 1
    y_aligned = align_A_to_B_batched_f(y, x, batch, num_graphs)
    assert torch.allclose(y_aligned, x, atol=1e-6)


def test_batching_two_separate_transformations():
    y = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],  # Graph 0
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # Graph 1
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],  # Graph 0 (translated +1 in all dims)
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],  # Graph 1 (rotated 90 deg around z)
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 1, 1])
    num_graphs = 2
    y_aligned = align_A_to_B_batched_f(y, x, batch, num_graphs)
    assert torch.allclose(y_aligned, x, atol=1e-6)


def test_different_number_of_points_per_graph_in_batch():
    y = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],  # Graph 0 (3 points)
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # Graph 1 (2 points)
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],  # Graph 0 (translated +10)
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],  # Graph 1 (rotated 90 deg around z)
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 0, 1, 1])
    num_graphs = 2
    y_aligned = align_A_to_B_batched_f(y, x, batch, num_graphs)
    assert torch.allclose(y_aligned, x, atol=1e-6)


def test_stress_test_random_points_and_transformations():
    torch.manual_seed(42)
    num_points = 100
    num_dims = 3
    num_graphs_stress = 5
    batch_stress = torch.randint(0, num_graphs_stress, (num_points,))

    y_stress = torch.randn(num_points, num_dims, dtype=torch.float32)
    x_stress = torch.zeros_like(y_stress)

    for i in range(num_graphs_stress):
        graph_indices = batch_stress == i
        if torch.sum(graph_indices) == 0:
            continue

        q, _ = torch.linalg.qr(torch.randn(num_dims, num_dims, dtype=torch.float32))
        if torch.linalg.det(q) < 0:
            q[:, -1] *= -1

        t_stress = torch.randn(num_dims, dtype=torch.float32) * 5

        x_stress[graph_indices] = (y_stress[graph_indices] @ q.T) + t_stress

    y_stress_aligned = align_A_to_B_batched_f(y_stress, x_stress, batch_stress, num_graphs_stress)
    assert torch.allclose(y_stress_aligned, x_stress, atol=1e-5)


@pytest.mark.parametrize("correction_order", [0, 1, 2])
def test_correction_order_zero_sigma(correction_order: int):
    y = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],  # Graph 0 (3 points)
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # Graph 1 (2 points)
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],  # Graph 0 (translated +10)
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],  # Graph 1 (rotated 90 deg around z)
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 0, 1, 1])
    num_graphs = 2
    y_aligned = align_A_to_B_batched_f(y, x, batch, num_graphs, sigma=0.0, correction_order=correction_order)
    assert torch.allclose(y_aligned, x, atol=1e-6)


@pytest.mark.parametrize("correction_order", [1, 2])
@pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0])
def test_nonzero_correction_order_nonzero_sigma(correction_order: int, sigma: float):
    y = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],  # Graph 0 (3 points)
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # Graph 1 (2 points)
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],  # Graph 0 (translated +10)
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],  # Graph 1 (rotated 90 deg around z)
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 0, 1, 1])
    num_graphs = 2
    y_aligned_zero_sigma = align_A_to_B_batched_f(y, x, batch, num_graphs, sigma=0.0, correction_order=correction_order)
    y_aligned = align_A_to_B_batched_f(y, x, batch, num_graphs, sigma=sigma, correction_order=correction_order)
    assert not torch.allclose(y_aligned_zero_sigma, y_aligned, atol=1e-3)
