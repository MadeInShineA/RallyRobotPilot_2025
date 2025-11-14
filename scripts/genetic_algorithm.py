import json
import random
import os
import lzma
import pickle
import subprocess
import tempfile
from typing import List, Tuple, Optional


class GeneticAlgorithm:
    """Genetic algorithm for finding optimal racing paths"""

    def __init__(
        self,
        track_name: str,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.8,
        replay_file: Optional[str] = None,
    ):
        self.track_name = track_name
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.replay_file = replay_file

        # Load or initialize population
        self.population = self._load_population()
        self.fitness_scores = []

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
        """Create initial population by mutating base actions"""
        # First, try to load recorded keys from the track
        record_file = f"genetic_data/records/{self.track_name}/complete_record.json"
        if os.path.exists(record_file):
            with open(record_file, "r") as f:
                data = json.load(f)
            base_actions = [item["input"] for item in data]
            print(f"Using recorded inputs from {record_file} as base actions")
        else:
            print(f"No recorded inputs found at {record_file}, using fallback")
            exit()
        population = []
        for _ in range(self.population_size):
            mutated = self.mutate_actions(base_actions, self.mutation_rate)
            population.append(mutated)

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

            # Evolve to next generation (except for last generation)
            if generation < self.generations - 1:
                self._evolve_population()

        print("\nEvolution complete!")
        print(".2f")
        self._save_results()

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
                "--batch",
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
            # The genetic_autopilot_runner should output fitness scores
            lines = result.stdout.strip().split("\n")
            fitness_scores = []

            for line in lines:
                if line.startswith("FITNESS:"):
                    try:
                        score = float(line.split(":")[1].strip())
                        fitness_scores.append(score)
                    except (ValueError, IndexError):
                        continue

            if len(fitness_scores) != len(self.population):
                print(
                    f"Warning: Expected {len(self.population)} fitness scores, got {len(fitness_scores)}"
                )
                # Pad with high scores if needed
                while len(fitness_scores) < len(self.population):
                    fitness_scores.append(1000.0)

            return fitness_scores[: len(self.population)]

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

    if len(sys.argv) < 2:
        print(
            "Usage: python genetic_algorithm.py <track_name> [population_size] [generations] [replay_file]"
        )
        sys.exit(1)

    track_name = sys.argv[1]
    population_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    replay_file = sys.argv[4] if len(sys.argv) > 4 else None
    mutation_rate = 0

    ga = GeneticAlgorithm(
        track_name,
        population_size,
        generations,
        mutation_rate=mutation_rate,
        replay_file=replay_file,
    )
    ga.run_evolution()


if __name__ == "__main__":
    main()
