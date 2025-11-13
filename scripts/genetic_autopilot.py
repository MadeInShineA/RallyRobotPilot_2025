import json
import time
import os
from typing import List, Tuple, Optional
from rallyrobopilot.sensing_message import SensingSnapshot
from ursina import held_keys


class GeneticAutopilot:
    """Autopilot that executes genetic algorithm action sequences and collects fitness data"""

    def __init__(
        self, action_sequence: List[Tuple[int, int, int, int]], max_time: float = 300.0
    ):
        self.action_sequence = action_sequence
        self.max_time = max_time
        self.current_action_index = 0
        self.start_time = 0.0
        self.is_running = False
        self.frame_counter = 0

        # Fitness tracking
        self.wall_hits = 0
        self.checkpoints_passed = 0
        self.lap_completed = False
        self.total_time = 0.0

        # Previous sensing data for wall detection
        self.previous_raycasts: Optional[List[float]] = None

    def start_evaluation(self):
        """Start the evaluation run"""
        self.start_time = time.time()
        self.is_running = True
        self.current_action_index = 0
        self.frame_counter = 0
        self.wall_hits = 0
        self.checkpoints_passed = 0
        self.lap_completed = False

        print(
            f"Starting genetic autopilot evaluation with {len(self.action_sequence)} actions"
        )

    def stop_evaluation(self):
        """Stop the evaluation and return results"""
        self.total_time = time.time() - self.start_time
        self.is_running = False

        results = {
            "lap_time": self.total_time,
            "wall_hits": self.wall_hits,
            "checkpoints_passed": self.checkpoints_passed,
            "lap_completed": self.lap_completed,
        }

        print(f"Evaluation complete: {results}")
        return results

    def update(self, sensing_data: SensingSnapshot):
        """Update method called each frame with sensing data"""
        if not self.is_running:
            return

        self.frame_counter += 1

        # Check timeout
        if time.time() - self.start_time > self.max_time:
            self.stop_evaluation()
            return

        # Detect wall hits
        if self._detect_wall_hit(sensing_data):
            self.wall_hits += 1

        # Update checkpoint progress from sensing data
        self.checkpoints_passed = sensing_data.checkpoints_passed

        # Check lap completion (assuming lap completion when checkpoints_passed == total_checkpoints)
        if (
            sensing_data.checkpoints_passed >= sensing_data.total_checkpoints
            and sensing_data.total_checkpoints > 0
        ):
            self.lap_completed = True
            self.stop_evaluation()
            return

        # Execute current action every 10 frames
        if self.frame_counter % 10 == 0:
            if self.current_action_index < len(self.action_sequence):
                action = self.action_sequence[self.current_action_index]
                self._execute_action(action)

                # Move to next action
                self.current_action_index += 1
            else:
                # Action sequence finished
                self.stop_evaluation()

    def _detect_wall_hit(self, sensing_data: SensingSnapshot) -> bool:
        """Detect if car hit a wall based on raycast distances"""
        if not sensing_data.raycast_distances:
            return False

        # Wall hit if any raycast is below threshold
        wall_threshold = 1.0
        current_hits = sum(
            1 for dist in sensing_data.raycast_distances if dist < wall_threshold
        )

        # Compare with previous to detect new hits
        if self.previous_raycasts:
            prev_hits = sum(
                1 for dist in self.previous_raycasts if dist < wall_threshold
            )
            if current_hits > prev_hits:
                self.previous_raycasts = list(sensing_data.raycast_distances)
                return True

        self.previous_raycasts = list(sensing_data.raycast_distances)
        return False

    def _execute_action(self, action: Tuple[int, int, int, int]):
        """Execute a single action by setting held_keys"""
        # action = (forward, back, left, right)
        key_mapping = {
            "w": action[0],  # forward
            "s": action[1],  # back
            "a": action[2],  # left
            "d": action[3],  # right
        }

        # Set held_keys directly
        for key, pressed in key_mapping.items():
            held_keys[key] = bool(pressed)  # type: ignore


class GeneticMsgProcessor:
    """Message processor for genetic autopilot integration with DataCollectionUI"""

    def __init__(
        self,
        action_sequences_path: Optional[str] = None,
        action_sequences: Optional[List[List[Tuple[int, int, int, int]]]] = None,
    ):
        if action_sequences_path:
            self.autopilots = self.load_genetic_autopilots(action_sequences_path)
        elif action_sequences:
            self.autopilots = [GeneticAutopilot(seq) for seq in action_sequences]
        else:
            raise ValueError(
                "Must provide either action_sequences_path or action_sequences"
            )

        self.current_autopilot_index = 0
        self.results = []
        self.checkpoint_handler = None

    def load_genetic_autopilots(
        self, action_sequences_path: str
    ) -> List[GeneticAutopilot]:
        """Load multiple genetic autopilots from a JSON file containing action sequences"""
        if not os.path.exists(action_sequences_path):
            raise FileNotFoundError(
                f"Action sequences file not found: {action_sequences_path}"
            )

        with open(action_sequences_path, "r") as f:
            action_sequences = json.load(f)

        return [GeneticAutopilot(seq) for seq in action_sequences]

    def load_single_autopilot(self, action_sequence_path: str) -> GeneticAutopilot:
        """Load a single genetic autopilot from a JSON file containing action sequences"""
        if not os.path.exists(action_sequence_path):
            raise FileNotFoundError(
                f"Action sequence file not found: {action_sequence_path}"
            )

        with open(action_sequence_path, "r") as f:
            action_sequence = json.load(f)

        return GeneticAutopilot(action_sequence)

    def start_evaluation(self):
        """Start the genetic evaluation for all autopilots"""
        self.current_autopilot_index = 0
        self.results = []
        if self.autopilots:
            self.autopilots[0].start_evaluation()

    def get_results(self):
        """Get evaluation results for all autopilots"""
        return self.results

    def process_message(self, message, data_collector):
        """Process sensing messages for genetic autopilot"""
        if not self.autopilots:
            return

        if self.current_autopilot_index >= len(self.autopilots):
            # All evaluations complete - signal to exit
            print("ALL_EVALUATIONS_COMPLETE")
            return

        current_autopilot = self.autopilots[self.current_autopilot_index]

        # Create SensingSnapshot from message
        snapshot = SensingSnapshot()
        snapshot.current_controls = (
            tuple(message.current_controls)
            if hasattr(message, "current_controls")
            else (0, 0, 0, 0)
        )
        snapshot.car_position = (
            tuple(message.car_position)
            if hasattr(message, "car_position")
            else (0, 0, 0)
        )
        snapshot.car_speed = message.car_speed if hasattr(message, "car_speed") else 0
        snapshot.car_angle = message.car_angle if hasattr(message, "car_angle") else 0
        snapshot.raycast_distances = (
            list(message.raycast_distances)
            if hasattr(message, "raycast_distances")
            else []
        )
        snapshot.checkpoints_passed = (
            message.checkpoints_passed if hasattr(message, "checkpoints_passed") else 0
        )
        snapshot.total_checkpoints = (
            message.total_checkpoints if hasattr(message, "total_checkpoints") else 0
        )

        # Update current autopilot
        current_autopilot.update(snapshot)

        # Check if current autopilot finished
        if not current_autopilot.is_running:
            # Save results and move to next autopilot
            results = current_autopilot.stop_evaluation()
            self.results.append(results)

            self.current_autopilot_index += 1
            if self.current_autopilot_index < len(self.autopilots):
                # Start next autopilot
                self.autopilots[self.current_autopilot_index].start_evaluation()
            else:
                # All evaluations complete
                print("ALL_EVALUATIONS_COMPLETE")

        # Send current controls to data collector for recording
        if hasattr(message, "current_controls"):
            controls = message.current_controls
            commands = [
                ("forward", bool(controls[0])),
                ("back", bool(controls[1])),
                ("left", bool(controls[2])),
                ("right", bool(controls[3])),
            ]

            for command, active in commands:
                data_collector.onCarControlled(command, active)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python genetic_autopilot.py <action_sequences.json>")
        sys.exit(1)

    action_file = sys.argv[1]
    processor = GeneticMsgProcessor(action_sequences_path=action_file)
    print(f"Loaded {len(processor.autopilots)} genetic autopilots")

