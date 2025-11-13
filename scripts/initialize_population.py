import numpy as np
import json
import random
import lzma
import pickle
from typing import List, Tuple, Union


def load_replay_data(
    replay_file: str,
) -> Tuple[
    List[Tuple[bool, bool, bool, bool]], List[Tuple[float, float, float]], List[float]
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
    actions: List[Tuple[bool, bool, bool, bool]], mutation_rate: float = 0.3
) -> List[Tuple[bool, bool, bool, bool]]:
    """
    Mutate a copy of actions by randomly replacing actions.
    """
    new_actions = []
    for action in actions:
        if random.random() < mutation_rate:
            new_actions.append(tuple(random.choice([True, False]) for _ in range(4)))
        else:
            new_actions.append(action)
    return new_actions


def generate_population_from_replay(
    replay_file: str, population_size: int, mutation_rate: float = 0.3
) -> Tuple[
    List[List[Tuple[bool, bool, bool, bool]]],
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
    return population, positions, angles


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
    population, positions, angles = generate_population_from_replay(
        replay_file, population_size, mutation_rate
    )
    with open("initial_population.json", "w") as f:
        json.dump(population, f)
    with open("replay_positions.json", "w") as f:
        json.dump(positions, f)
    with open("replay_angles.json", "w") as f:
        json.dump(angles, f)
    print(f"Saved initial population of {population_size} to initial_population.json")
