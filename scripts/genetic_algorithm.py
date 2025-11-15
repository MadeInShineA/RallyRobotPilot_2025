import json
import random
import os
import lzma
import pickle
import subprocess
import tempfile
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    """Genetic algorithm for finding optimal racing paths"""

    def __init__(
        self,
        track_name: str,
        population_size: int = 50,
        generations: int = 100,
        segment: int = 0,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.8,
        replay_file: Optional[str] = None,
    ):
        self.track_name = track_name
        self.population_size = population_size
        self.generations = generations
        self.segment = segment
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.replay_file = replay_file

        # Load checkpoints
        self.checkpoints = []
        checkpoints_path = f"assets/{self.track_name}/checkpoints.json"
        if os.path.exists(checkpoints_path):
            with open(checkpoints_path) as f:
                self.checkpoints = json.load(f)
        else:
            print(f"Warning: checkpoints.json not found at {checkpoints_path}")

        # Load or initialize population
        self.population = self._load_population()
        self.fitness_scores = []
        self.individual_positions = []

        # Track best individual
        self.best_individual = None
        self.best_fitness = float("inf")

    def load_replay_data(
        self, replay_file: str
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
        self, actions: List[Tuple[int, int, int, int]], mutation_rate: float = 0.3
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

    def _load_population(self) -> List[List[Tuple[int, int, int, int]]]:
        """Load existing population or create new one"""
        population_file = f"genetic_data/{self.track_name}_population.json"

        if os.path.exists(population_file):
            with open(population_file, "r") as f:
                return json.load(f)
        else:
            # Create initial population from base actions
            return self._initialize_population()

    def _initialize_population(self) -> List[List[Tuple[int, int, int, int]]]:
        """Create initial population by mutating base actions from segment"""
        # Load segment data
        segment_file = f"genetic_data/records/{self.track_name}/segments/segment_{self.segment}.json"
        if os.path.exists(segment_file):
            with open(segment_file, "r") as f:
                data = json.load(f)
            base_actions = [item["input"] for item in data]
            # Store initial conditions
            initial = data[0]
            self.initial_angle = initial["angle"]
            self.initial_speed = initial["speed"]
            self.initial_position = json.loads(initial["position"])
            self.original_positions = [json.loads(item["position"]) for item in data]
            print(f"Using segment inputs from {segment_file} as base actions")
        else:
            print(f"No segment file found at {segment_file}")
            exit()
        # Extend base actions to 1.5 times length with [0, 0, 0, 0]
        len_base = len(base_actions)
        extended_len = int(len_base * 1.5)
        extended_actions = base_actions + [[0, 0, 0, 0]] * (extended_len - len_base)
        # All individuals are mutations of extended actions
        population = [
            self.mutate_actions(extended_actions, self.mutation_rate)
            for _ in range(self.population_size)
        ]

        return population

    def _mutate_actions(
        self, actions: List[Tuple[int, int, int, int]], mutation_rate: float
    ) -> List[Tuple[int, int, int, int]]:
        """Mutate a sequence of actions"""
        mutated = []
        for action in actions:
            if random.random() < mutation_rate:
                # Randomly flip one control bit
                control_idx = random.randint(0, 3)
                new_action = list(action)
                new_action[control_idx] = 1 - new_action[control_idx]  # Flip 0<->1
                mutated.append(tuple(new_action))
            else:
                mutated.append(action)
        return mutated

    def _crossover(
        self,
        parent1: List[Tuple[int, int, int, int]],
        parent2: List[Tuple[int, int, int, int]],
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """Perform crossover between two parents"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        # Single point crossover
        min_len = min(len(parent1), len(parent2))
        if min_len < 2:
            return parent1, parent2

        crossover_point = random.randint(1, min_len - 1)

        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def _select_parents(
        self,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """Select two parents using tournament selection"""

        def tournament_select():
            # Select up to 3 random individuals (or fewer if population is small)
            tournament_size = min(3, len(self.population))
            candidates = random.sample(
                list(zip(self.population, self.fitness_scores)), tournament_size
            )
            # Return the one with best (lowest) fitness
            return min(candidates, key=lambda x: x[1])[0]

        parent1 = tournament_select()
        parent2 = tournament_select()

        return parent1, parent2

    def _evolve_population(self):
        """Create new population through selection, crossover, and mutation"""
        new_population = []

        # Elitism: keep best individual
        best_idx = self.fitness_scores.index(min(self.fitness_scores))
        new_population.append(self.population[best_idx])

        # Fill rest of population
        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parents()
            child1, child2 = self._crossover(parent1, parent2)

            # Mutate children
            child1 = self._mutate_actions(child1, self.mutation_rate)
            child2 = self._mutate_actions(child2, self.mutation_rate)

            new_population.extend([child1, child2])

        # Trim to exact population size
        self.population = new_population[: self.population_size]

    def run_evolution(self):
        """Run the genetic algorithm with automatic game evaluation"""
        print(
            f"Starting GA for {self.track_name} with population size {self.population_size}"
        )
        print("This GA will automatically launch the game to evaluate fitness!")

        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")

            # Evaluate current population by launching the game
            print("Evaluating population in game...")
            self.fitness_scores = self._evaluate_population_in_game()

            # Track best individual
            min_fitness = min(self.fitness_scores)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                best_idx = self.fitness_scores.index(min_fitness)
                self.best_individual = self.population[best_idx]

            print(".2f")

            # Plot trajectories
            if (
                hasattr(self, "original_positions")
                and self.original_positions
                and self.individual_positions
            ):
                best_idx = self.fitness_scores.index(min(self.fitness_scores))
                print(
                    f"Debug: Plotting gen {generation + 1}, individual_positions len: {len(self.individual_positions)}, best_idx: {best_idx}"
                )
                for i, pos in enumerate(self.individual_positions):
                    print(f"  Individual {i}: positions len {len(pos)}")
                plt.figure()
                # Plot all individual trajectories in light gray
                for i, positions in enumerate(self.individual_positions):
                    if positions:
                        x = [p[0] for p in positions]
                        z = [p[2] for p in positions]
                        plt.plot(x, z, color="gray", alpha=0.3, linewidth=0.5)
                # Plot best trajectory in red dashed
                best_positions = (
                    self.individual_positions[best_idx]
                    if best_idx < len(self.individual_positions)
                    and self.individual_positions[best_idx]
                    else []
                )
                print(f"Debug: best_positions len: {len(best_positions)}")
                if best_positions:
                    x_best = [p[0] for p in best_positions]
                    z_best = [p[2] for p in best_positions]
                    plt.plot(
                        x_best,
                        z_best,
                        color="red",
                        linewidth=2,
                        linestyle="--",
                        label=f"Best Gen {generation + 1}",
                    )
                # Plot original trajectory in blue solid
                x_orig = [p[0] for p in self.original_positions]
                z_orig = [p[2] for p in self.original_positions]
                plt.plot(
                    x_orig,
                    z_orig,
                    color="blue",
                    linewidth=2,
                    label="Original Trajectory",
                )
                plt.xlabel("X")
                plt.ylabel("Z")
                plt.title(f"All Trajectories - Gen {generation + 1}")
                plt.legend()
                graph_dir = f"genetic_data/populations/{self.track_name}/graphs"
                os.makedirs(graph_dir, exist_ok=True)
                graph_path = (
                    f"{graph_dir}/segment_{self.segment}_gen_{generation + 1}.png"
                )
                plt.savefig(graph_path)
                plt.close()
                print(f"Trajectories saved to {graph_path}")

            # Evolve to next generation (except for last generation)
            if generation < self.generations - 1:
                self._evolve_population()

        print("\nEvolution complete!")
        print(".2f")
        self._save_results()

    def calculate_fitness(
        self,
        wall_hits: int,
        segment_completed: bool,
        inputs_to_finish_segment: int,
        distance_to_next: float,
    ) -> float:
        """Calculate fitness score from evaluation results"""
        # Reward closeness to next checkpoint, bonus for completion
        base_fitness = distance_to_next
        if segment_completed:
            base_fitness -= 1000.0  # bonus for completing
        wall_penalty = wall_hits * 10.0

        fitness = base_fitness + wall_penalty
        return fitness

    def _evaluate_population_in_game(self) -> List[float]:
        """Evaluate the entire population by launching the game"""
        # Save current population to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.population, f)
            population_file = f.name

        try:
            # Launch genetic_autopilot_runner with the population file
            print(f"Launching game to evaluate {len(self.population)} individuals...")
            cmd = [
                "python",
                "scripts/genetic_autopilot_runner.py",
                population_file,
                self.track_name,
                "--start_segment",
                str(self.segment),
                "--initial_angle",
                str(self.initial_angle),
                "--initial_speed",
                str(self.initial_speed),
                "--initial_position",
                json.dumps(self.initial_position),
            ]
            if self.replay_file:
                cmd.append(self.replay_file)

            # Run the subprocess
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=os.getcwd()
            )

            if result.returncode != 0:
                print(f"Error running game evaluation: {result.stderr}")
                # Return high fitness scores as penalty
                return [1000.0] * len(self.population)

            # Parse results from stdout
            # The genetic_autopilot_runner should output results
            lines = result.stdout.strip().split("\n")
            fitness_dict = {}
            self.individual_positions = [[] for _ in range(len(self.population))]

            i = 0
            while i < len(lines):
                line = lines[i]
                if line.startswith("INDIVIDUAL") and "RESULTS:" in line:
                    parts = line.split()
                    try:
                        idx = int(parts[1])
                        results_str = " ".join(
                            parts[3:]
                        )  # wall_hits segment_completed inputs_to_finish_segment
                        results_parts = results_str.split()
                        wall_hits = int(results_parts[0])
                        segment_completed = results_parts[1].lower() == "true"
                        inputs_to_finish_segment = int(results_parts[2])

                        # Next line should be positions JSON
                        positions = []
                        if i + 1 < len(lines):
                            try:
                                positions = json.loads(lines[i + 1])
                                i += 1  # Skip the positions line
                            except json.JSONDecodeError:
                                pass

                        # Compute distance to next checkpoint
                        distance = 1000.0
                        if positions:
                            last_pos = positions[-1]
                            next_cp_idx = self.segment + 1
                            if next_cp_idx < len(self.checkpoints):
                                next_cp = self.checkpoints[next_cp_idx]["position"]
                                distance = (
                                    (last_pos[0] - next_cp[0]) ** 2
                                    + (last_pos[2] - next_cp[2]) ** 2
                                ) ** 0.5

                        fitness_score = self.calculate_fitness(
                            wall_hits,
                            segment_completed,
                            inputs_to_finish_segment,
                            distance,
                        )
                        fitness_dict[idx] = fitness_score

                        self.individual_positions[idx] = positions
                        print(
                            f"Individual {idx}: wall_hits={wall_hits}, segment_completed={segment_completed}, inputs_to_finish_segment={inputs_to_finish_segment}, distance={distance:.2f}, fitness={fitness_score:.2f}"
                        )
                    except (ValueError, IndexError):
                        pass
                i += 1

            # Build fitness_scores list, defaulting to high penalty
            fitness_scores = [
                fitness_dict.get(i, 1000.0) for i in range(len(self.population))
            ]

            return fitness_scores

        finally:
            # Clean up temporary file
            try:
                os.unlink(population_file)
            except:
                pass

    def _save_results(self):
        """Save the best individual and final population"""
        os.makedirs("genetic_data", exist_ok=True)

        # Save best individual
        best_file = f"genetic_data/{self.track_name}_best_individual.json"
        with open(best_file, "w") as f:
            json.dump(self.best_individual, f, indent=2)

        # Save final population
        population_file = f"genetic_data/{self.track_name}_population.json"
        with open(population_file, "w") as f:
            json.dump(self.population, f, indent=2)

        print(f"Results saved to {best_file} and {population_file}")


def main():
    import sys

    if len(sys.argv) < 5:
        print(
            "Usage: python genetic_algorithm.py <track_name> <population_size> <generations> <segment>"
        )
        sys.exit(1)

    track_name = sys.argv[1]
    population_size = int(sys.argv[2])
    generations = int(sys.argv[3])
    segment = int(sys.argv[4])
    mutation_rate = 0.1
    crossover_rate = 0.3

    ga = GeneticAlgorithm(
        track_name,
        population_size,
        generations,
        segment=segment,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
    )
    ga.run_evolution()


if __name__ == "__main__":
    main()
