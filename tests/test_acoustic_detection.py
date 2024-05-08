import numpy as np
import pytest

from acousticdetection.main import multilateration, MIC_POSITIONS, SPEED_OF_SOUND, visualize_positions


def simulate_tdoas(source_position, mic_positions, speed_of_sound):
    distances = np.sqrt(((mic_positions - source_position) ** 2).sum(axis=1))
    tdoas = distances / speed_of_sound
    tdoas -= tdoas.min()
    return tdoas


@pytest.mark.parametrize(
    "source_position, expected_position",
    [
        (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        (np.array([0.1, 0.9]), np.array([0.1, 0.9])),
        (np.array([0.9, 0.1]), np.array([0.9, 0.1])),
    ],
)
def test_multilateration(source_position, expected_position):
    tdoas = simulate_tdoas(source_position, MIC_POSITIONS, SPEED_OF_SOUND)
    estimated_position = multilateration(tdoas, MIC_POSITIONS, SPEED_OF_SOUND)
    np.testing.assert_allclose(estimated_position, expected_position, atol=0.25)

@pytest.mark.skip("Visualization test")
@pytest.mark.parametrize(
    "source_position, expected_position",
    [
        (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        (np.array([0.1, 0.9]), np.array([0.1, 0.9])),
        (np.array([0.9, 0.1]), np.array([0.9, 0.1])),
    ],
)
def test_visualization(source_position, expected_position):
    tdoas = simulate_tdoas(source_position, MIC_POSITIONS, SPEED_OF_SOUND)
    estimated_position = multilateration(tdoas, MIC_POSITIONS, SPEED_OF_SOUND)
    visualize_positions(MIC_POSITIONS, estimated_position)


if __name__ == "__main__":
    pytest.main()
