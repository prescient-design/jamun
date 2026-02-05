import torch
from torch_geometric.data import Batch, Data


def add_noise_hiddens(x: Batch, N_measurements_hidden: int, N_measurements: int, sigma: float) -> Batch:
    """
    Makes N_measurements_hidden number of noisy copies of the hidden states of x
    and then for every noisy copy, makes N_measurements number of noisy copies of the positions of x.

    Args:
        x (Batch): A torch_geometric Batch object. Must have `pos` and `hidden_state` attributes.
                    `hidden_state` is expected to be a list of tensors.
        N_measurements_hidden (int): Number of noisy copies of hidden states.
        N_measurements (int): Number of noisy copies of positions for each noisy hidden state.
        sigma (float): The standard deviation of the Gaussian noise to add.

    Returns:
        Batch: A new Batch object containing all the noisy copies.
    """
    x_list = x.to_data_list()
    noisy_y_list = []

    for graph in x_list:
        for _ in range(N_measurements_hidden):
            # Create a noisy version of the hidden state
            noisy_hidden_state = []
            if hasattr(graph, "hidden_state") and graph.hidden_state is not None:
                for hs_tensor in graph.hidden_state:
                    noise = torch.randn_like(hs_tensor) * sigma
                    noisy_hidden_state.append(hs_tensor + noise)

            for _ in range(N_measurements):
                noisy_graph = graph.clone()

                # Add noise to positions
                pos_noise = torch.randn_like(graph.pos) * sigma
                noisy_graph.pos = graph.pos + pos_noise

                # Assign the noisy hidden state
                if hasattr(graph, "hidden_state") and graph.hidden_state is not None:
                    noisy_graph.hidden_state = [hs.clone() for hs in noisy_hidden_state]

                noisy_y_list.append(noisy_graph)

    return Batch.from_data_list(noisy_y_list)


# --- Testing Script ---
def run_test():
    print("Running test for add_noise_hiddens...")

    # 1. Create dummy data
    num_nodes = 5
    # single data object
    data1 = Data(pos=torch.randn(num_nodes, 3), hidden_state=[torch.randn(num_nodes, 4), torch.randn(num_nodes, 8)])
    # another data object
    data2 = Data(
        pos=torch.randn(num_nodes + 2, 3), hidden_state=[torch.randn(num_nodes + 2, 4), torch.randn(num_nodes + 2, 8)]
    )

    original_batch = Batch.from_data_list([data1, data2])

    # 2. Set parameters
    N_measurements_hidden = 2
    N_measurements = 3
    sigma = 0.1

    # 3. Call the function
    noisy_batch = add_noise_hiddens(original_batch, N_measurements_hidden, N_measurements, sigma)

    # 4. Assertions
    # Check total number of graphs
    expected_num_graphs = original_batch.num_graphs * N_measurements_hidden * N_measurements
    assert noisy_batch.num_graphs == expected_num_graphs, (
        f"Expected {expected_num_graphs} graphs, but got {noisy_batch.num_graphs}"
    )
    print(f"Correct number of graphs in output batch: {noisy_batch.num_graphs}")

    noisy_graphs = noisy_batch.to_data_list()

    # Check that noise was added
    assert not torch.allclose(noisy_graphs[0].pos, data1.pos)
    assert not torch.allclose(noisy_graphs[0].hidden_state[0], data1.hidden_state[0])
    print("Noise was added to pos and hidden_state.")

    # Check hidden state logic
    # The first N_measurements graphs (for the first original graph) should have the same hidden state
    first_hidden_state_set = noisy_graphs[0].hidden_state
    for i in range(1, N_measurements):
        assert torch.allclose(noisy_graphs[i].hidden_state[0], first_hidden_state_set[0])
        assert torch.allclose(noisy_graphs[i].hidden_state[1], first_hidden_state_set[1])

    # But their positions should be different
    assert not torch.allclose(noisy_graphs[0].pos, noisy_graphs[1].pos)

    # The (N_measurements+1)-th graph should have a different hidden state (from the second hidden measurement)
    next_hidden_state_set = noisy_graphs[N_measurements].hidden_state
    assert not torch.allclose(next_hidden_state_set[0], first_hidden_state_set[0])

    print("Hidden state noise logic seems correct.")

    print("Test passed!")


if __name__ == "__main__":
    run_test()
