import json
import random
import os
import lzma
import pickle
import subprocess
import tempfile
import math
import shutil
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

        # Load track metadata for obstacles
        self.track_metadata = {}
        metadata_path = f"assets/{self.track_name}/track_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.track_metadata = json.load(f)
        else:
            print(f"Warning: track_metadata.json not found at {metadata_path}")

        # Load obstacle vertices for plotting track boundaries
        self.obstacle_vertices = self._load_obstacle_vertices()

        # Load or initialize population
        self.population = self._load_population()
        self.fitness_scores = []
        self.individual_positions = []

        # Track best individual
        self.best_individual = None
        self.best_fitness = float("inf")
        self.best_individual_idx = None
        self.best_individual_stats = None

    def _load_obstacle_vertices(self):
        """Load and transform obstacle vertices for track boundary plotting"""
        obstacle_data = []
        if not self.track_metadata or "obstacles" not in self.track_metadata:
            return obstacle_data

        origin_pos = np.array(self.track_metadata.get("origin_position", [0, 0, 0]))
        origin_rot_y = self.track_metadata.get("origin_rotation", [0, 0, 0])[1]
        origin_scale = self.track_metadata.get("origin_scale", [1, 1, 1])
        scale = origin_scale[1] if len(origin_scale) > 1 else 1.0

        for obstacle in self.track_metadata["obstacles"]:
            model_path = f"assets/{self.track_name}/{obstacle['model']}"
            if os.path.exists(model_path):
                obs_vertices, obs_faces = self._parse_obj_vertices(model_path)
                transformed_vertices = []
                for v in obs_vertices:
                    scaled_v = np.array(v) * scale
                    rot_rad = math.radians(origin_rot_y)
                    cos_r = math.cos(rot_rad)
                    sin_r = math.sin(rot_rad)
                    rotated_v = np.array(
                        [
                            scaled_v[0] * cos_r - scaled_v[2] * sin_r,
                            scaled_v[1],
                            scaled_v[0] * sin_r + scaled_v[2] * cos_r,
                        ]
                    )
                    transformed_v = rotated_v + origin_pos
                    transformed_vertices.append(transformed_v)

                # Store both transformed vertices and faces
                obstacle_data.append(
                    {"vertices": transformed_vertices, "faces": obs_faces}
                )

        return obstacle_data

    def _parse_obj_vertices(self, obj_path):
        """Parse vertices and faces from OBJ file"""
        vertices = []
        faces = []
        with open(obj_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            v = [float(parts[1]), float(parts[2]), float(parts[3])]
                            vertices.append(v)
                        except ValueError:
                            pass
                elif line.startswith("f "):
                    parts = line.split()
                    if len(parts) >= 4:  # At least a triangle
                        face = []
                        for p in parts[1:]:
                            # Parse vertex index (ignore texture and normal indices)
                            v_idx = (
                                int(p.split("/")[0]) - 1
                            )  # OBJ uses 1-based indexing
                            face.append(v_idx)
                        faces.append(face)
        return vertices, faces

    def load_replay_data(
        self, replay_file: str
    ) -> Tuple[
        List[Tuple[int, int, int, int]],
        List[Tuple[float, float, float]],
        List[float],
        List[List[float]],
    ]:
        """
        Load actions, positions, angles from replay file.
        Assumes replay is a list of SensingSnapshot objects or dicts.
        Returns actions as tuples of booleans, positions as tuples, angles as floats, raycasts as lists.
        """
        with lzma.open(replay_file, "rb") as f:
            data = pickle.load(f)
        actions = []
        positions = []
        angles = []
        for msg in data:
            if hasattr(msg, "current_controls"):
                # SensingSnapshot
                actions.append(tuple(msg.current_controls))
                positions.append(tuple(msg.car_position))
                angles.append(msg.car_angle)
            else:
                # Dict from manual recording
                actions.append(tuple(msg["input"]))
                positions.append(tuple(json.loads(msg["position"])))
                angles.append(msg["angle"])
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
        print(
            f"DEBUG _initialize_population: segment_file {segment_file} exists: {os.path.exists(segment_file)}"
        )
        if os.path.exists(segment_file):
            with open(segment_file) as f:
                data = json.load(f)
            print(f"DEBUG: data loaded, len: {len(data)}")
            base_actions = [item["input"] for item in data]
            # Store initial conditions
            initial = data[0]
            self.initial_angle = initial["angle"]
            self.initial_speed = initial["speed"]
            self.initial_position = json.loads(initial["position"])
            self.original_positions = [json.loads(item["position"]) for item in data]
            self.original_angles = [item["angle"] for item in data]
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

        # Clear previous results for this segment
        segment_dir = (
            f"genetic_data/populations/{self.track_name}/segment_{self.segment}"
        )
        if os.path.exists(segment_dir):
            shutil.rmtree(segment_dir)
            print(f"Cleared previous results in {segment_dir}")

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
        collision_counter: int,
        segment_completed: bool,
        frames_used: int,
        distance_to_next: float,
    ) -> float:
        """Calculate fitness score from evaluation results"""
        # Reward closeness to next checkpoint, bonus for completion
        if segment_completed:
            base_fitness = -1000
        else:
            base_fitness = distance_to_next

        collision_penalty = collision_counter * 10.0

        fitness = base_fitness + collision_penalty + (frames_used * 20)
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
                        )  # collision_counter segment_completed frames_used
                        results_parts = results_str.split()
                        collision_counter = int(results_parts[0])
                        segment_completed = results_parts[1].lower() == "true"
                        frames_used = int(results_parts[2])

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
                            collision_counter,
                            segment_completed,
                            frames_used,
                            distance,
                        )
                        fitness_dict[idx] = fitness_score

                        self.individual_positions[idx] = positions
                        self.completion_status[idx] = segment_completed

                        # Store detailed stats for this individual
                        self.individual_stats[idx] = {
                            "collision_counter": collision_counter,
                            "segment_completed": segment_completed,
                            "frames_used": frames_used,
                            "distance_to_checkpoint": distance,
                            "fitness_score": fitness_score,
                        }
                        print(
                            f"Individual {idx}: collision_counter={collision_counter}, segment_completed={segment_completed}, frames_used={frames_used}, distance={distance:.2f}, fitness={fitness_score:.2f}"
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
                        "all_fitness_scores": self.fitness_scores.copy(),
                        "all_individual_stats": self.individual_stats.copy(),
                    }
                    break
        else:
            # Add new generation
            generation_data = {
                "generation": generation_num,
                "best_generation_individual_idx": best_idx,
                "best_generation_individual_scores": best_stats,
                "all_fitness_scores": self.fitness_scores.copy(),
                "all_individual_stats": self.individual_stats.copy(),
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

    def _load_original_data(self):
        """Load original segment data for plotting"""
        segment_file = f"genetic_data/records/{self.track_name}/segments/segment_{self.segment}.json"
        if os.path.exists(segment_file):
            with open(segment_file) as f:
                data = json.load(f)
            self.original_positions = [json.loads(item["position"]) for item in data]
            self.original_angles = [item["angle"] for item in data]
        else:
            self.original_positions = []
            self.original_angles = []

    def _generate_summary_plot(self):
        """Generate three separated summary plots for all generations"""
        # Ensure original data is loaded

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

        generations = [g["generation"] for g in summary["generations"]]
        fitness_scores = [
            g["best_generation_individual_scores"]["fitness_score"]
            for g in summary["generations"]
        ]

        # Handle frame efficiency data - only include generations where segment was completed
        frames_data = []
        completed_generations = []
        for g in summary["generations"]:
            if g["best_generation_individual_scores"]["segment_completed"]:
                frames_data.append(
                    g["best_generation_individual_scores"]["frames_used"]
                )
                completed_generations.append(g["generation"])

        # Get original actions count (from initial segment data)
        original_actions_count = (
            len(self.original_positions) if hasattr(self, "original_positions") else 0
        )

        # Plot 1: Trajectory Evolution
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        overall_best = summary.get("overall_best_individual")
        best_gen_num, best_ind_idx = None, None

        if overall_best:
            try:
                gen_part, ind_part = overall_best.split("_individual_")
                best_gen_num = int(gen_part.split("_")[1])
                best_ind_idx = int(ind_part)
            except (ValueError, IndexError):
                pass

        # Initialize overall best variables
        best_positions = []
        overall_best_inputs = 0
        if best_gen_num is not None and best_ind_idx is not None:
            best_trajectory_file = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/generation_{best_gen_num}/individual_{best_ind_idx}.json"
            if os.path.exists(best_trajectory_file):
                with open(best_trajectory_file, "r") as f:
                    best_trajectory_data = json.load(f)
                best_positions = [
                    frame.get("position", [0, 0, 0]) for frame in best_trajectory_data
                ]
                best_positions = [
                    json.loads(pos) if isinstance(pos, str) else pos
                    for pos in best_positions
                ]
                if best_positions and best_positions[0] != self.initial_position:
                    best_positions.insert(0, self.initial_position)
                overall_best_inputs = len(best_positions) if best_positions else 0

        # Plot trajectories from all generations
        for g in summary["generations"]:
            gen_num = g["generation"]
            ind_idx = g["best_generation_individual_idx"]

            trajectory_file = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/generation_{gen_num}/individual_{ind_idx}.json"

            if os.path.exists(trajectory_file):
                with open(trajectory_file, "r") as f:
                    trajectory_data = json.load(f)

                # Extract positions, prepend initial if not present
                positions = [
                    frame.get("position", [0, 0, 0]) for frame in trajectory_data
                ]
                positions = [
                    json.loads(pos) if isinstance(pos, str) else pos
                    for pos in positions
                ]
                if positions and positions[0] != self.initial_position:
                    positions.insert(0, self.initial_position)
                traj_x = [-pos[0] for pos in positions]  # Flip horizontally
                traj_z = [-pos[2] for pos in positions]  # Flip vertically

                # Check if this is the overall best
                if gen_num == best_gen_num and ind_idx == best_ind_idx:
                    # Highlight the overall best in blue
                    ax1.plot(
                        traj_x,
                        traj_z,
                        "blue",
                        linewidth=4,
                        label=f"Overall Best Individual ({overall_best_inputs} frames)",
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
            orig_x = [-p[0] for p in self.original_positions]  # Flip horizontally
            orig_z = [-p[2] for p in self.original_positions]  # Flip vertically
            ax1.plot(
                orig_x,
                orig_z,
                "orange",
                linewidth=3,
                label=f"Original Trajectory ({len(self.original_positions)} frames)",
                alpha=0.8,
                zorder=2,
            )
            # Plot track boundaries from obstacles as connected lines

            if self.obstacle_vertices:
                for i, obstacle in enumerate(self.obstacle_vertices):
                    vertices = obstacle["vertices"]
                    faces = obstacle["faces"]

                    # Plot each face as a separate polygon
                    for face in faces:
                        # Get the vertices for this face
                        face_vertices = [vertices[idx] for idx in face]

                        # Extract X and Z coordinates (flip Z vertically)
                        xs = [v[0] for v in face_vertices]
                        zs = [-v[2] for v in face_vertices]

                        # Close the polygon by repeating the first point at the end
                        xs.append(xs[0])
                        zs.append(zs[0])

                        # Plot this face
                        ax1.plot(
                            xs,
                            zs,
                            color="black",
                            linewidth=2,
                            alpha=1.0,
                            zorder=1,
                            label="Track Elements"
                            if i == 0 and face == faces[0]
                            else "",
                        )

                print(
                    f"Plotted {len(self.obstacle_vertices)} obstacles with {sum(len(obstacle['faces']) for obstacle in self.obstacle_vertices)} faces as track boundaries"
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
            Line2D([0], [0], color="orange", linewidth=3, label=f"Original Trajectory ({len(self.original_positions) if hasattr(self, 'original_positions') and self.original_positions else 0} frames)"),
            Line2D(
                [0],
                [0],
                color="lightgray",
                linewidth=2,
                alpha=0.5,
                label="Generation Bests",
            ),
            Line2D([0], [0], color="blue", linewidth=4, label=f"Overall Best ({overall_best_inputs} frames)"),
        ]
        # Add border legends
        if self.obstacle_vertices:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="o",
                    markersize=1,
                    linestyle="None",
                    label="Track Elements",
                )
            )
        ax1.legend(handles=legend_elements, fontsize=10)

        ax1.grid(True, alpha=0.3)
        # Zoom on trajectories
        all_x = []
        all_z = []
        for g in summary["generations"]:
            gen_num = g["generation"]
            ind_idx = g["best_generation_individual_idx"]
            trajectory_file = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/generation_{gen_num}/individual_{ind_idx}.json"
            if os.path.exists(trajectory_file):
                with open(trajectory_file, "r") as f:
                    trajectory_data = json.load(f)
                positions = [
                    frame.get("position", [0, 0, 0]) for frame in trajectory_data
                ]
                positions = [
                    json.loads(pos) if isinstance(pos, str) else pos
                    for pos in positions
                ]
                if positions and positions[0] != self.initial_position:
                    positions.insert(0, self.initial_position)
                traj_x = [-pos[0] for pos in positions]
                traj_z = [-pos[2] for pos in positions]
                all_x.extend(traj_x)
                all_z.extend(traj_z)
        if hasattr(self, "original_positions") and self.original_positions:
            orig_x = [-p[0] for p in self.original_positions]
            orig_z = [-p[2] for p in self.original_positions]
            all_x.extend(orig_x)
            all_z.extend(orig_z)
        if all_x and all_z:
            margin = 15
            ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax1.set_ylim(min(all_z) - margin, max(all_z) + margin)

        # Overall title
        total_generations = len(summary["generations"])
        fig1.suptitle(
            f"Trajectory Evolution - {summary['track_name']} Segment {self.segment} ({total_generations} Generations)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Save the plot
        plot_file1 = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/summary_trajectory.png"
        plt.savefig(plot_file1, dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 2: Fitness Evolution with subplots
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # Best fitness
        ax2a.plot(
            generations,
            fitness_scores,
            "b-o",
            linewidth=3,
            markersize=8,
            label="Best Fitness",
        )
        ax2a.set_ylabel("Best Fitness Score", fontsize=12)
        ax2a.set_title("Best Fitness Evolution", fontsize=14, fontweight="bold")
        ax2a.grid(True, alpha=0.3)
        ax2a.legend()

        # Mean fitness
        mean_fitness_scores = []
        for g in summary["generations"]:
            if "all_fitness_scores" in g:
                mean_fitness_scores.append(np.mean(g["all_fitness_scores"]))
            else:
                mean_fitness_scores.append(g["best_generation_individual_scores"]["fitness_score"])  # fallback

        ax2b.plot(
            generations,
            mean_fitness_scores,
            "r-s",
            linewidth=3,
            markersize=8,
            label="Mean Fitness",
        )
        ax2b.set_xlabel("Generation", fontsize=12)
        ax2b.set_ylabel("Mean Fitness Score", fontsize=12)
        ax2b.set_title("Mean Fitness Evolution", fontsize=14, fontweight="bold")
        ax2b.set_xticks(generations)
        ax2b.grid(True, alpha=0.3)
        ax2b.legend()

        fig2.suptitle(
            f"Fitness Evolution - {summary['track_name']} Segment {self.segment} ({total_generations} Generations)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plot_file2 = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/summary_fitness.png"
        plt.savefig(plot_file2, dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 3: Frame Efficiency Evolution with subplots
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        if frames_data:
            # Best frames
            ax3a.plot(
                completed_generations,
                frames_data,
                "g-o",
                linewidth=3,
                markersize=8,
                label="Best Individual Frames",
            )
            ax3a.axhline(
                y=original_actions_count,
                color="red",
                linestyle="--",
                linewidth=3,
                label=f"Original Frames ({original_actions_count})",
            )
            ax3a.set_ylabel("Best Frames Used", fontsize=12)
            ax3a.set_title(
                "Best Frame Efficiency Evolution\n(Completed Segments Only)",
                fontsize=14,
                fontweight="bold",
            )
            ax3a.grid(True, alpha=0.3)
            ax3a.legend()

            # Mean frames for completed
            mean_frames_data = []
            for g in summary["generations"]:
                if g["best_generation_individual_scores"]["segment_completed"] and "all_individual_stats" in g:
                    completed_frames = [
                        stat["frames_used"]
                        for stat in g["all_individual_stats"]
                        if stat["segment_completed"]
                    ]
                    if completed_frames:
                        mean_frames_data.append(np.mean(completed_frames))
                    else:
                        mean_frames_data.append(g["best_generation_individual_scores"]["frames_used"])
                elif g["best_generation_individual_scores"]["segment_completed"]:
                    mean_frames_data.append(g["best_generation_individual_scores"]["frames_used"])

            if mean_frames_data:
                ax3b.plot(
                    completed_generations,
                    mean_frames_data,
                    "m-^",
                    linewidth=3,
                    markersize=8,
                    label="Mean Frames (Completed)",
                )
                ax3b.axhline(
                    y=original_actions_count,
                    color="red",
                    linestyle="--",
                    linewidth=3,
                    label=f"Original Frames ({original_actions_count})",
                )
                ax3b.set_xlabel("Generation", fontsize=12)
                ax3b.set_ylabel("Mean Frames Used", fontsize=12)
                ax3b.set_title(
                    "Mean Frame Efficiency Evolution\n(Completed Segments Only)",
                    fontsize=14,
                    fontweight="bold",
                )
                ax3b.set_xticks(completed_generations)
                ax3b.grid(True, alpha=0.3)
                ax3b.legend()
            else:
                ax3b.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax3b.transAxes,
                    fontsize=14,
                    color="gray",
                )
                ax3b.set_title(
                    "Mean Input Efficiency Evolution\n(Completed Segments Only)",
                    fontsize=14,
                    fontweight="bold",
                )
        else:
            # No completed segments
            for ax in [ax3a, ax3b]:
                ax.text(
                    0.5,
                    0.5,
                    "No segments\ncompleted yet",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                    color="gray",
                )
                ax.set_title(
                    "Frame Efficiency Evolution\n(Completed Segments Only)",
                    fontsize=14,
                    fontweight="bold",
                )

        fig3.suptitle(
            f"Frame Efficiency - {summary['track_name']} Segment {self.segment} ({total_generations} Generations)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plot_file3 = f"genetic_data/populations/{self.track_name}/segment_{self.segment}/summary_efficiency.png"
        plt.savefig(plot_file3, dpi=150, bbox_inches="tight")
        plt.close()

        # Reset seaborn style
        sns.reset_defaults()

        print(f"Summary plots saved to {plot_file1}, {plot_file2}, {plot_file3}")

    def _save_results(self):
        """Save final summary (legacy method, now handled per generation)"""
        # This method is now redundant since we save after each generation
        # But keep it for backward compatibility
        pass

    def _plot_combined_analysis(self, generation, best_idx, graph_dir):
        """Create combined figure with trajectory, completion, and frames analysis"""
        fig = plt.figure(figsize=(16, 12))

        # Create subplot grid: 2 rows, 2 columns
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])  # Trajectory
        ax2 = fig.add_subplot(gs[0, 1])  # Completion status
        ax3 = fig.add_subplot(gs[1, 0])  # Frames comparison
        ax4 = fig.add_subplot(gs[1, 1])  # Collision count

        # 1. Trajectory subplot (with all individuals)
        # Plot all individual trajectories in light gray
        for i, positions in enumerate(self.individual_positions):
            if positions:
                x = [-p[0] for p in positions]  # Flip horizontally
                z = [-p[2] for p in positions]  # Flip vertically
                ax1.plot(x, z, color="lightgray", alpha=0.4, linewidth=1)

        # Plot best trajectory
        best_positions = (
            self.individual_positions[best_idx]
            if best_idx < len(self.individual_positions)
            and self.individual_positions[best_idx]
            else []
        )

        if best_positions:
            x_best = [-p[0] for p in best_positions]  # Flip horizontally
            z_best = [-p[2] for p in best_positions]  # Flip vertically
            ax1.plot(
                x_best,
                z_best,
                color="blue",
                linewidth=3,
                label=f"Best Individual ({len(best_positions)} frames)",
                alpha=0.9,
            )

        # Plot original trajectory
        if hasattr(self, "original_positions") and self.original_positions:
            orig_x = [-p[0] for p in self.original_positions]  # Flip horizontally
            orig_z = [-p[2] for p in self.original_positions]  # Flip vertically
            ax1.plot(
                orig_x,
                orig_z,
                "orange",
                linewidth=3,
                label=f"Original Trajectory ({len(self.original_positions)} frames)",
                alpha=0.8,
                zorder=2,
            )

        ax1.set_xlabel("X Position", fontsize=11)
        ax1.set_ylabel("Z Position", fontsize=11)
        ax1.set_title(
            f"Trajectory Analysis - Generation {generation}",
            fontsize=13,
            fontweight="bold",
        )
        # Add track borders from obstacles
        if self.obstacle_vertices:
            for obstacle in self.obstacle_vertices:
                vertices = obstacle["vertices"]
                faces = obstacle["faces"]

                # Plot each face as a separate polygon
                for face in faces:
                    # Get the vertices for this face
                    face_vertices = [vertices[idx] for idx in face]

                    # Extract X and Z coordinates (flip Z vertically)
                    xs = [v[0] for v in face_vertices]
                    zs = [-v[2] for v in face_vertices]

                    # Close the polygon by repeating the first point at the end
                    xs.append(xs[0])
                    zs.append(zs[0])

                    # Plot this face
                    ax1.plot(xs, zs, color="black", linewidth=2, alpha=1.0, zorder=1)

        ax1.legend(fontsize=10, loc="best")
        ax1.grid(True, alpha=0.3)
        # Zoom on trajectories
        all_x = []
        all_z = []
        # Collect from all individual trajectories
        for positions in self.individual_positions:
            if positions:
                x = [-p[0] for p in positions]
                z = [-p[2] for p in positions]
                all_x.extend(x)
                all_z.extend(z)
        # From best
        if best_positions:
            x_best = [-p[0] for p in best_positions]
            z_best = [-p[2] for p in best_positions]
            all_x.extend(x_best)
            all_z.extend(z_best)
        # From original
        if hasattr(self, "original_positions") and self.original_positions:
            x_orig = [-p[0] for p in self.original_positions]
            z_orig = [-p[2] for p in self.original_positions]
            all_x.extend(x_orig)
            all_z.extend(z_orig)
        if all_x and all_z:
            margin = 15
            ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax1.set_ylim(min(all_z) - margin, max(all_z) + margin)

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
        completed_frames = [
            self.individual_stats[i]["frames_used"]
            for i in completed_indices
        ]

        if completed_frames:  # Only plot if there are completed individuals
            bars = ax3.bar(completed_ids, completed_frames, color="skyblue", alpha=0.7)

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
            for bar, frames in zip(bars, completed_frames):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(completed_frames) * 0.02,
                    str(frames),
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

        # 4. Collision count per individual
        individual_ids = [f"Ind {i}" for i in range(len(self.individual_stats))]
        collision_counts = [
            self.individual_stats[i]["collision_counter"]
            for i in range(len(self.individual_stats))
        ]

        if collision_counts:  # Plot collision counts for all individuals
            bars = ax4.bar(individual_ids, collision_counts, color="red", alpha=0.7)

            ax4.set_xlabel("Individual", fontsize=11)
            ax4.set_ylabel("Collision Count", fontsize=11)
            ax4.set_title(
                "Collision Count per Individual", fontsize=12, fontweight="bold"
            )

            # Set y-axis to show only integers
            if collision_counts:
                y_max = max(collision_counts)
                ax4.set_ylim(bottom=0, top=y_max + 1)
                ax4.set_yticks(range(0, y_max + 2))

            # Add value labels on bars
            for bar, count in zip(bars, collision_counts):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(collision_counts) * 0.02
                    if collision_counts
                    else 0.1,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            ax4.tick_params(axis="x", rotation=45)
            ax4.grid(True, alpha=0.3, axis="y")
        else:
            # No collision data
            ax4.text(
                0.5,
                0.5,
                "No collision\ndata available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
                color="gray",
            )
            ax4.set_title(
                "Collision Count per Individual", fontsize=12, fontweight="bold"
            )
            ax4.grid(True, alpha=0.3, axis="y")

        # Overall title
        fig.suptitle(
            f"Genetic Algorithm Summary - Segment {self.segment} - {self.track_name}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Save the combined figure
        graph_path = f"{graph_dir}/summary.png"
        plt.savefig(graph_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Combined summary graph saved to {graph_path}")


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
