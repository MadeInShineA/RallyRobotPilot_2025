import numpy as np
import json
import random
import lzma
import pickle
from typing import List, Tuple, Union


def load_replay_data(
    replay_file: str,
) -> Tuple[
    List[Tuple[int, int, int, int]], List[Tuple[float, float, float]], List[float]
]:
    """
    Load actions, positions, and angles from replay file.
    Assumes replay is a list of SensingSnapshot objects.
    Returns actions as tuples of booleans, positions as tuples, angles as floats.
    """
    with lzma.open(replay_file, "rb") as f:
        data = pickle.load(f)
    actions = []
    positions = []
    angles = []
    for msg in data:
        actions.append(tuple(msg.current_controls))
        positions.append(tuple(msg.car_position))
        angles.append(msg.car_angle)
    return actions, positions, angles


def mutate_actions(
    actions: List[Tuple[int, int, int, int]], mutation_rate: float = 0.3
) -> List[Tuple[int, int, int, int]]:
    """
    Mutate a copy of actions by randomly flipping individual values (0 or 1).
    """
    new_actions = []
    for action in actions:
        mutated_action = list(action)  # Convert to list for mutation
        for i in range(4):
            if random.random() < mutation_rate:
                mutated_action[i] = 1 - mutated_action[i]  # Flip 0 to 1 or 1 to 0
        new_actions.append(tuple(mutated_action))
    return new_actions


def generate_population_from_replay(
    replay_file: str, population_size: int, mutation_rate: float = 0.3
) -> Tuple[
    List[Tuple[int, int, int, int]],
    List[List[Tuple[int, int, int, int]]],
    List[Tuple[float, float, float]],
    List[float],
]:
    """
    Generate initial population based on replay.
    Each individual is an independent mutation of the replay.
    """
    base_actions, positions, angles = load_replay_data(replay_file)
    population = []
    for _ in range(population_size):
        mutated = mutate_actions(base_actions, mutation_rate)
        population.append(mutated)
    return base_actions, population, positions, angles


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python initialize_population.py <replay_file.npz> <population_size>"
        )
        sys.exit(1)

    replay_file = sys.argv[1]
    population_size = int(sys.argv[2])
    mutation_rate = 0.3
    base_actions, population, positions, angles = generate_population_from_replay(
        replay_file, population_size, mutation_rate
    )

    with open("genetic_data/base_actions.json", "w") as f:
        json.dump(base_actions, f)
    with open("genetic_data/initial_population.json", "w") as f:
        json.dump(population, f)
    with open("genetic_data/replay_positions.json", "w") as f:
        json.dump(positions, f)
    with open("genetic_data/replay_angles.json", "w") as f:
        json.dump(angles, f)
    print(
        f"Saved initial population of {population_size} to genetic_data/initial_population.json"
    )
