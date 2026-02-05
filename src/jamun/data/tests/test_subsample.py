import numpy as np

from jamun.data._subsample import get_subsampled_indices, get_subsampled_trajectory


def test_basic_functionality():
    """Test basic functionality with valid inputs."""
    print("\nTesting basic functionality...")
    N = 100
    subsample_rate = 10
    total_lag_time = 3
    lag_subsample_rate = 10

    print(
        f"Input parameters: N={N}, subsample_rate={subsample_rate}, "
        f"total_lag_time={total_lag_time}, lag_subsample_rate={lag_subsample_rate}"
    )

    breakpoint()  # Debug point 1: Check input parameters before function call

    lagged_indices = get_subsampled_indices(N, subsample_rate, total_lag_time, lag_subsample_rate)

    breakpoint()  # Debug point 2: Check function output

    # Extract subsampled indices (first element of each list)
    subsampled_indices = np.array([indices[0] for indices in lagged_indices])
    print(f"Subsampled indices: {subsampled_indices}")
    print(f"Number of lagged indices lists: {len(lagged_indices)}")

    # Check subsampled indices
    expected_subsampled = np.array([20, 30, 40, 50, 60, 70, 80, 90])
    assert np.array_equal(subsampled_indices, expected_subsampled), (
        f"Expected {expected_subsampled}, got {subsampled_indices}"
    )

    # Check lagged indices
    for i, lagged in enumerate(lagged_indices):
        expected_lagged = np.array(
            [
                subsampled_indices[i],
                subsampled_indices[i] - lag_subsample_rate,
                subsampled_indices[i] - 2 * lag_subsample_rate,
            ]
        )
        assert np.array_equal(lagged, expected_lagged), f"For index {i}, expected {expected_lagged}, got {lagged}"

    print("Basic functionality test passed!")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting edge cases...")
    N = 10
    subsample_rate = 10
    total_lag_time = 1
    lag_subsample_rate = 1

    print(
        f"Input parameters: N={N}, subsample_rate={subsample_rate}, "
        f"total_lag_time={total_lag_time}, lag_subsample_rate={lag_subsample_rate}"
    )

    breakpoint()  # Debug point 5: Check edge case parameters before function call

    lagged_indices = get_subsampled_indices(N, subsample_rate, total_lag_time, lag_subsample_rate)

    breakpoint()  # Debug point 6: Check edge case results

    # Extract subsampled indices (first element of each list)
    subsampled_indices = np.array([indices[0] for indices in lagged_indices])
    print(f"Subsampled indices: {subsampled_indices}")
    print(f"Lagged indices: {lagged_indices}")

    assert len(subsampled_indices) == 1, f"Expected 1 subsampled index, got {len(subsampled_indices)}"
    assert len(lagged_indices) == 1, f"Expected 1 lagged indices list, got {len(lagged_indices)}"
    assert np.array_equal(subsampled_indices, np.array([0])), f"Expected [0], got {subsampled_indices}"
    assert np.array_equal(lagged_indices[0], np.array([0])), f"Expected [0], got {lagged_indices[0]}"

    print("Edge cases test passed!")


def test_lagged_indices_filtering():
    """Test that lagged indices are properly filtered when they would go negative."""
    print("\nTesting lagged indices filtering...")
    N = 20
    subsample_rate = 5
    total_lag_time = 3
    lag_subsample_rate = 3

    print(
        f"Input parameters: N={N}, subsample_rate={subsample_rate}, "
        f"total_lag_time={total_lag_time}, lag_subsample_rate={lag_subsample_rate}"
    )

    breakpoint()  # Debug point 7: Check filtering parameters before function call

    lagged_indices = get_subsampled_indices(N, subsample_rate, total_lag_time, lag_subsample_rate)

    breakpoint()  # Debug point 8: Check filtering results

    # Extract subsampled indices (first element of each list)
    subsampled_indices = np.array([indices[0] for indices in lagged_indices])
    print(f"Subsampled indices: {subsampled_indices}")
    print(f"Number of lagged indices lists: {len(lagged_indices)}")

    expected_subsampled = np.array([5, 10, 15])
    assert np.array_equal(subsampled_indices, expected_subsampled), (
        f"Expected {expected_subsampled}, got {subsampled_indices}"
    )

    assert len(lagged_indices) == len(subsampled_indices), (
        f"Expected {len(subsampled_indices)} lagged indices lists, got {len(lagged_indices)}"
    )

    expected_first_lagged = np.array([5, 2, -1])
    assert not any(np.array_equal(lagged, expected_first_lagged) for lagged in lagged_indices), (
        "Found unexpected lagged indices that should have been filtered out"
    )

    print("Lagged indices filtering test passed!")


def test_large_numbers():
    """Test with larger numbers to ensure scalability."""
    print("\nTesting large numbers...")
    N = 10000
    subsample_rate = 100
    total_lag_time = 5
    lag_subsample_rate = 10

    print(
        f"Input parameters: N={N}, subsample_rate={subsample_rate}, "
        f"total_lag_time={total_lag_time}, lag_subsample_rate={lag_subsample_rate}"
    )

    breakpoint()  # Debug point 9: Check large number parameters before function call

    lagged_indices = get_subsampled_indices(N, subsample_rate, total_lag_time, lag_subsample_rate)

    breakpoint()  # Debug point 10: Check large number results

    # Extract subsampled indices (first element of each list)
    subsampled_indices = np.array([indices[0] for indices in lagged_indices])
    print(f"Number of subsampled indices: {len(subsampled_indices)}")
    print(f"Number of lagged indices lists: {len(lagged_indices)}")

    assert len(subsampled_indices) == N // subsample_rate, (
        f"Expected {N // subsample_rate} subsampled indices, got {len(subsampled_indices)}"
    )

    for i, lagged in enumerate(lagged_indices):
        print(f"\nChecking lagged indices list {i}:")
        print(f"Lagged indices: {lagged}")
        print(f"Type of lagged indices: {type(lagged)}")
        print("Individual values and their types:")
        for j, val in enumerate(lagged):
            print(f"  Index {j}: value={val}, type={type(val)}")

        assert len(lagged) == total_lag_time, f"Expected lagged indices length {total_lag_time}, got {len(lagged)}"

        # Check each value individually
        for j, val in enumerate(lagged):
            assert isinstance(val, int | np.integer), (
                f"Value at index {j} is not an integer: {val} (type: {type(val)})"
            )
            assert val >= 0, f"Found negative value at index {j}: {val}"

    print("Large numbers test passed!")


def test_trajectory_subsampling():
    """Test subsampling of trajectory positions."""
    print("\nTesting trajectory subsampling...")

    # Create a random trajectory with 100 frames, 10 particles, and 3 coordinates
    N = 100
    np.random.seed(42)  # For reproducibility
    positions = np.random.randn(N, 10, 3)

    subsample_rate = 10
    total_lag_time = 3
    lag_subsample_rate = 10

    print(
        f"Input parameters: N={N}, subsample_rate={subsample_rate}, "
        f"total_lag_time={total_lag_time}, lag_subsample_rate={lag_subsample_rate}"
    )

    breakpoint()  # Debug point 11: Check trajectory parameters

    subsampled_positions, lagged_positions = get_subsampled_trajectory(
        positions, subsample_rate, total_lag_time, lag_subsample_rate
    )

    breakpoint()  # Debug point 12: Check trajectory results

    print(f"Original positions shape: {positions.shape}")
    print(f"Subsampled positions shape: {subsampled_positions.shape}")
    print(f"Number of lagged position lists: {len(lagged_positions)}")

    # Check shapes
    expected_num_subsampled = (N - 20) // subsample_rate  # Starting from index 20
    assert subsampled_positions.shape[0] == expected_num_subsampled, (
        f"Expected {expected_num_subsampled} subsampled positions, got {subsampled_positions.shape[0]}"
    )
    assert subsampled_positions.shape[1:] == (10, 3), (
        f"Expected subsampled positions to have shape (N, 10, 3), got {subsampled_positions.shape}"
    )

    # Check values
    for i in range(expected_num_subsampled):
        # Check subsampled position
        expected_sub_pos = positions[20 + i * subsample_rate]
        assert np.array_equal(subsampled_positions[i], expected_sub_pos), (
            f"For index {i}, expected subsampled position {expected_sub_pos}, got {subsampled_positions[i]}"
        )

        # Check lagged positions
        assert len(lagged_positions[i]) == total_lag_time, (
            f"For index {i}, expected {total_lag_time} lagged positions, got {len(lagged_positions[i])}"
        )

        for j, lag_pos in enumerate(lagged_positions[i]):
            expected_lag_pos = positions[20 + i * subsample_rate - j * lag_subsample_rate]
            assert np.array_equal(lag_pos, expected_lag_pos), (
                f"For index {i}, lag {j}, expected position {expected_lag_pos}, got {lag_pos}"
            )

    print("Trajectory subsampling test passed!")


if __name__ == "__main__":
    print("Starting tests...")
    test_basic_functionality()
    # test_edge_cases()
    # test_lagged_indices_filtering()
    # test_large_numbers()
    test_trajectory_subsampling()
    print("\nAll tests completed!")
