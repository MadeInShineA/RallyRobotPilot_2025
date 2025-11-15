import json
import random
import os
import lzma
import pickle
import subprocess
import tempfile
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.colors import Normalize
import seaborn as sns


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
        self.best_individual_idx = None
        self.best_individual_stats = None

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

        # Fill population through selection, crossover, and mutation (no elitism)
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
            self.fitness_scores = self._evaluate_population_in_game(generation + 1)

            # Track best individual for this generation
            min_fitness = min(self.fitness_scores)
            best_idx = self.fitness_scores.index(min_fitness)
            generation_best_stats = self.individual_stats[best_idx]

            print(
                f"Generation {generation + 1} best: idx={best_idx}, fitness={min_fitness:.2f}, stats={generation_best_stats}"
            )

            # Save this generation's data
            self._save_generation_result(
                generation + 1, best_idx, generation_best_stats
            )

            # Track overall best individual
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_individual = self.population[best_idx]
                self.best_individual_idx = best_idx
                self.best_individual_stats = generation_best_stats

                # Update overall best individual in summary
                self._update_overall_best(
                    f"generation_{generation + 1}_individual_{best_idx}"
                )

            print(".2f")

            # Create multiple analysis graphs
            if (
                hasattr(self, "original_positions")
                and self.original_positions
                and self.individual_positions
            ):
                best_idx = self.fitness_scores.index(min(self.fitness_scores))
                print(
                    f"Debug: Plotting gen {generation + 1}, individual_positions len: {len(self.individual_positions)}, best_idx: {best_idx}"
                )

                # Set seaborn style for the combined graph
                sns.set_style("whitegrid")
                sns.set_palette("husl")
                sns.set_context("notebook", font_scale=1.0)

                graph_dir = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/generation_{generation + 1}"
                os.makedirs(graph_dir, exist_ok=True)

                # Create combined figure with subplots
                self._plot_combined_analysis(generation + 1, best_idx, graph_dir)

                # Reset seaborn style
                sns.reset_defaults()

            # Evolve to next generation (except for last generation)
            if generation < self.generations - 1:
                self._evolve_population()

        print("\nEvolution complete!")
        print(".2f")

        # Generate summary plot
        self._generate_summary_plot()

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

    def _evaluate_population_in_game(self, generation: int) -> List[float]:
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
                "--generation",
                str(generation),
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
            self.completion_status = [False] * len(self.population)
            self.individual_stats = [{}] * len(self.population)

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
                        self.completion_status[idx] = segment_completed

                        # Store detailed stats for this individual
                        self.individual_stats[idx] = {
                            "wall_hits": wall_hits,
                            "segment_completed": segment_completed,
                            "inputs_to_finish_segment": inputs_to_finish_segment,
                            "distance_to_checkpoint": distance,
                            "fitness_score": fitness_score,
                        }
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

    def _save_generation_result(self, generation_num, best_idx, best_stats):
        """Save a specific generation's results to the summary file"""
        os.makedirs(
            f"genetic_data/populations/{self.track_name}/segment_{self.segment}",
            exist_ok=True,
        )
        summary_file = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/summary.json"

        # Load existing summary or create new one
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summary = json.load(f)
        else:
            summary = {
                "track_name": self.track_name,
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "overall_best_individual": None,
                "generations": [],
            }

        # Check if this generation already exists (avoid duplicates)
        existing_generations = [g["generation"] for g in summary["generations"]]
        if generation_num in existing_generations:
            # Update existing generation
            for i, gen_data in enumerate(summary["generations"]):
                if gen_data["generation"] == generation_num:
                    summary["generations"][i] = {
                        "generation": generation_num,
                        "best_generation_individual_idx": best_idx,
                        "best_generation_individual_scores": best_stats,
                    }
                    break
        else:
            # Add new generation
            generation_data = {
                "generation": generation_num,
                "best_generation_individual_idx": best_idx,
                "best_generation_individual_scores": best_stats,
            }
            summary["generations"].append(generation_data)

        # Save updated summary
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Generation {generation_num} results saved to {summary_file}")

    def _update_overall_best(self, overall_best_str):
        """Update the overall best individual in the summary file"""
        summary_file = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/summary.json"

        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summary = json.load(f)

            summary["overall_best_individual"] = overall_best_str

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"Updated overall best individual to {overall_best_str}")

    def _generate_summary_plot(self):
        """Generate a comprehensive summary plot for all generations"""
        summary_file = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/summary.json"

        if not os.path.exists(summary_file):
            print("No summary file found, cannot generate summary plot")
            return

        with open(summary_file, "r") as f:
            summary = json.load(f)

        if not summary.get("generations"):
            print("No generations data found, cannot generate summary plot")
            return

        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        sns.set_context("notebook", font_scale=1.1)

        # Create figure with subplots using GridSpec for better control
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Trajectory - full width top row
        ax2 = fig.add_subplot(gs[1, 0])  # Fitness evolution
        ax3 = fig.add_subplot(gs[1, 1])  # Input efficiency
        ax4 = None  # No fourth subplot needed

        generations = [g["generation"] for g in summary["generations"]]
        fitness_scores = [
            g["best_generation_individual_scores"]["fitness_score"]
            for g in summary["generations"]
        ]

        # Handle input efficiency data - only include generations where segment was completed
        inputs_data = []
        completed_generations = []
        for g in summary["generations"]:
            if g["best_generation_individual_scores"]["segment_completed"]:
                inputs_data.append(
                    g["best_generation_individual_scores"]["inputs_to_finish_segment"]
                )
                completed_generations.append(g["generation"])

        # Get original actions count (from initial segment data)
        original_actions_count = (
            len(self.original_positions) if hasattr(self, "original_positions") else 0
        )

        # 1. Original vs Best Trajectory Comparison (now ax1 - full width)
        # Plot all best trajectories from each generation in light gray
        overall_best = summary.get("overall_best_individual")
        best_gen_num, best_ind_idx = None, None

        if overall_best:
            try:
                gen_part, ind_part = overall_best.split("_individual_")
                best_gen_num = int(gen_part.split("_")[1])
                best_ind_idx = int(ind_part)
            except (ValueError, IndexError):
                pass

        # Plot trajectories from all generations
        for g in summary["generations"]:
            gen_num = g["generation"]
            ind_idx = g["best_generation_individual_idx"]

            trajectory_file = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/generation_{gen_num}/individual_{ind_idx}.json"

            if os.path.exists(trajectory_file):
                with open(trajectory_file, "r") as f:
                    trajectory_data = json.load(f)

                # Extract positions
                traj_x = [frame.get("position", [0, 0, 0]) for frame in trajectory_data]
                traj_x = [
                    json.loads(pos)[0] if isinstance(pos, str) else pos[0]
                    for pos in traj_x
                ]
                traj_z = [frame.get("position", [0, 0, 0]) for frame in trajectory_data]
                traj_z = [
                    json.loads(pos)[2] if isinstance(pos, str) else pos[2]
                    for pos in traj_z
                ]

                # Check if this is the overall best
                if gen_num == best_gen_num and ind_idx == best_ind_idx:
                    # Highlight the overall best in blue
                    ax1.plot(
                        traj_x,
                        traj_z,
                        "blue",
                        linewidth=4,
                        label="Overall Best Individual",
                        alpha=0.9,
                        zorder=3,
                    )
                else:
                    # Plot other generation bests in light gray
                    ax1.plot(
                        traj_x, traj_z, "lightgray", linewidth=2, alpha=0.5, zorder=1
                    )

        # Plot original trajectory (from initial segment data)
        if hasattr(self, "original_positions") and self.original_positions:
            orig_x = [p[0] for p in self.original_positions]
            orig_z = [p[2] for p in self.original_positions]
            ax1.plot(
                orig_x,
                orig_z,
                "orange",
                linewidth=3,
                label="Original Trajectory",
                alpha=0.8,
                zorder=2,
            )

        # Add legend and labels
        ax1.set_xlabel("X Position", fontsize=12)
        ax1.set_ylabel("Z Position", fontsize=12)
        ax1.set_title(
            "Trajectory Evolution: Original vs All Generation Bests",
            fontsize=14,
            fontweight="bold",
        )

        # Create custom legend
        legend_elements = [
            Line2D([0], [0], color="orange", linewidth=3, label="Original Trajectory"),
            Line2D(
                [0],
                [0],
                color="lightgray",
                linewidth=2,
                alpha=0.5,
                label="Generation Bests",
            ),
            Line2D([0], [0], color="blue", linewidth=4, label="Overall Best"),
        ]
        ax1.legend(handles=legend_elements, fontsize=10)

        ax1.grid(True, alpha=0.3)
        ax1.axis("equal")

        # 2. Fitness Score over Generations (now ax2)
        ax2.plot(
            generations,
            fitness_scores,
            "b-o",
            linewidth=3,
            markersize=8,
            label="Best Fitness",
        )
        ax2.set_xlabel("Generation", fontsize=12)
        ax2.set_ylabel("Fitness Score", fontsize=12)
        ax2.set_title("Fitness Evolution", fontsize=14, fontweight="bold")
        # Set y-axis based on data range with padding
        if fitness_scores:
            y_min = min(fitness_scores)
            y_max = max(fitness_scores)
            y_range = y_max - y_min
            ax2.set_ylim(bottom=y_min - y_range * 0.1, top=y_max + y_range * 0.1)
        # Set x-axis to show all generations
        ax2.set_xticks(generations)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Input to Complete Segment Evolution (ax3)
        if inputs_data:
            ax3.plot(
                completed_generations,
                inputs_data,
                "g-o",
                linewidth=3,
                markersize=8,
                label="Best Individual Inputs",
            )
            ax3.axhline(
                y=original_actions_count,
                color="red",
                linestyle="--",
                linewidth=3,
                label=f"Original Actions ({original_actions_count})",
            )
            ax3.set_xlabel("Generation", fontsize=12)
            ax3.set_ylabel("Inputs to Complete", fontsize=12)
            ax3.set_title(
                "Input Efficiency Evolution\n(Completed Segments Only)",
                fontsize=14,
                fontweight="bold",
            )
            # Set y-axis limits and ensure integer ticks
            all_values = inputs_data + [original_actions_count]
            y_min = min(all_values)
            y_max = max(all_values)
            y_range = y_max - y_min
            ax3.set_ylim(
                bottom=max(0, y_min - y_range * 0.1), top=y_max + y_range * 0.1
            )
            # Set integer ticks only
            y_ticks = list(range(int(ax3.get_ylim()[0]), int(ax3.get_ylim()[1]) + 1))
            ax3.set_yticks(y_ticks)
            # Set x-axis to show completed generations
            ax3.set_xticks(completed_generations)
            ax3.legend()
        else:
            # No completed segments
            ax3.text(
                0.5,
                0.5,
                "No segments\ncompleted yet",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=14,
                color="gray",
            )
            ax3.set_title(
                "Input Efficiency Evolution\n(Completed Segments Only)",
                fontsize=14,
                fontweight="bold",
            )

        ax3.grid(True, alpha=0.3)

        # No fourth subplot needed

        # Overall title
        total_generations = len(summary["generations"])
        fig.suptitle(
            f"Genetic Algorithm Summary - {summary['track_name']} Segment {self.segment} ({total_generations} Generations)\n"
            f"Population: {summary['population_size']} | Mutation: {summary['mutation_rate']} | Crossover: {summary['crossover_rate']}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()

        # Save the plot
        plot_file = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/summary.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()

        # Reset seaborn style
        sns.reset_defaults()

        print(f"Summary plot saved to {plot_file}")

    def _save_results(self):
        """Save final summary (legacy method, now handled per generation)"""
        # This method is now redundant since we save after each generation
        # But keep it for backward compatibility
        pass

    def _plot_combined_analysis(self, generation, best_idx, graph_dir):
        """Create combined figure with trajectory, completion, and frames analysis"""
        fig = plt.figure(figsize=(16, 12))

        # Create subplot grid: 2 rows, 2 columns
        # Top row: trajectory (spans both columns)
        # Bottom row: completion and frames
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Trajectory - full width
        ax2 = fig.add_subplot(gs[1, 0])  # Completion status
        ax3 = fig.add_subplot(gs[1, 1])  # Frames comparison

        # 1. Trajectory subplot (with all individuals)
        # Plot all individual trajectories in light gray
        for i, positions in enumerate(self.individual_positions):
            if positions:
                x = [p[0] for p in positions]
                z = [p[2] for p in positions]
                ax1.plot(x, z, color="lightgray", alpha=0.4, linewidth=1)

        # Plot best trajectory
        best_positions = (
            self.individual_positions[best_idx]
            if best_idx < len(self.individual_positions)
            and self.individual_positions[best_idx]
            else []
        )

        if best_positions:
            x_best = [p[0] for p in best_positions]
            z_best = [p[2] for p in best_positions]
            ax1.plot(
                x_best,
                z_best,
                color="blue",
                linewidth=3,
                label="Best Individual",
                alpha=0.9,
            )

        # Plot original trajectory
        x_orig = [p[0] for p in self.original_positions]
        z_orig = [p[2] for p in self.original_positions]
        ax1.plot(
            x_orig,
            z_orig,
            color="orange",
            linewidth=3,
            label="Original Trajectory",
            alpha=0.9,
        )

        ax1.set_xlabel("X Position", fontsize=11)
        ax1.set_ylabel("Z Position", fontsize=11)
        ax1.set_title(
            f"Trajectory Analysis - Generation {generation}",
            fontsize=13,
            fontweight="bold",
        )
        ax1.legend(fontsize=10, loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.axis("equal")

        # 2. Completion status pie chart
        completed_count = sum(self.completion_status)
        failed_count = len(self.completion_status) - completed_count

        if completed_count > 0 or failed_count > 0:
            # Create pie chart
            sizes = [completed_count, failed_count]
            labels = [f"Completed\n({completed_count})", f"Failed\n({failed_count})"]
            colors = ["green", "red"]
            explode = (0.1, 0)  # explode the completed slice

            pie_result = ax2.pie(
                sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                shadow=False,
                startangle=90,
            )

            # Style the text (pie returns wedges, texts, autotexts when autopct is used)
            wedges, texts = pie_result[0], pie_result[1]
            autotexts = pie_result[2] if len(pie_result) > 2 else []

            for text in texts:
                text.set_fontsize(10)
                text.set_fontweight("bold")
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color("white")
                autotext.set_fontweight("bold")

            ax2.set_title("Completion Status", fontsize=12, fontweight="bold")
        else:
            ax2.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
            ax2.set_title("Completion Status", fontsize=12, fontweight="bold")

        # 3. Frames comparison subplot (only completed individuals)
        initial_frames = len(self.original_positions)

        # Filter to only completed individuals
        completed_indices = [
            i for i, completed in enumerate(self.completion_status) if completed
        ]
        completed_ids = [f"Ind {i}" for i in completed_indices]
        completed_inputs = [
            self.individual_stats[i]["inputs_to_finish_segment"]
            for i in completed_indices
        ]

        if completed_inputs:  # Only plot if there are completed individuals
            bars = ax3.bar(completed_ids, completed_inputs, color="skyblue", alpha=0.7)

            # Add horizontal line for initial actions
            ax3.axhline(
                y=initial_frames,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Initial ({initial_frames})",
            )

            ax3.set_xlabel("Individual", fontsize=11)
            ax3.set_ylabel("Frames Used", fontsize=11)
            ax3.set_title(
                "Frames vs Initial\n(Completed Only)", fontsize=12, fontweight="bold"
            )
            ax3.legend(fontsize=9)

            # Add value labels on bars
            for bar, inputs in zip(bars, completed_inputs):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(completed_inputs) * 0.02,
                    str(inputs),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            ax3.tick_params(axis="x", rotation=45)
            ax3.grid(True, alpha=0.3, axis="y")
        else:
            # No completed individuals
            ax3.text(
                0.5,
                0.5,
                "No individuals\ncompleted segment",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
                color="gray",
            )
            ax3.set_title(
                "Frames vs Initial\n(Completed Only)", fontsize=12, fontweight="bold"
            )
            ax3.grid(True, alpha=0.3, axis="y")

        # Overall title
        fig.suptitle(
            f"Genetic Algorithm Analysis - Segment {self.segment} - {self.track_name}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Save the combined figure
        graph_path = f"{graph_dir}/analysis.png"
        plt.savefig(graph_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Combined analysis graph saved to {graph_path}")


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
    crossover_rate = 0.2

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
