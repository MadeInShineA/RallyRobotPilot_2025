import json
import time
import os
from typing import List, Tuple, Optional
from rallyrobopilot.sensing_message import SensingSnapshot
import ursina
from ursina import held_keys


class GeneticAutopilot:
    """Autopilot that executes genetic algorithm action sequences and collects fitness data"""

    def __init__(
        self,
        action_sequence: List[Tuple[int, int, int, int]],
        segment: int,
        car=None,
    ):
        self.action_sequence = action_sequence
        self.segment = segment
        self.car = car
        self.current_action_index = 0
        self.start_time = 0.0
        self.is_running = False
        self.frame_counter = 0

        # Fitness tracking
        self.collision_counter = 0
        self.segment_completed = False
        self.positions = []

        print(
            f"Starting genetic autopilot evaluation with {len(self.action_sequence)} actions"
        )

    def start_evaluation(self):
        """Start the evaluation run"""
        self.start_time = time.time()
        self.is_running = True
        self.current_action_index = 0
        self.frame_counter = 0
        self.collision_counter = 0
        self.segment_completed = False
        self.positions = []

        # Start recording
        if self.car:
            self.car.start_record()

    def stop_evaluation(self):
        """Stop the evaluation and return results"""
        self.total_time = time.time() - self.start_time
        self.is_running = False

        self.inputs_to_finish_segment = self.current_action_index

        # Stop recording
        if self.car:
            self.car.stop_record()

        results = {
            "collision_counter": self.collision_counter,
            "segment_completed": self.segment_completed,
            "inputs_to_finish_segment": self.inputs_to_finish_segment,
            "positions": self.positions,
            "recorded_frames": self.car.recorded_frames if self.car else [],
        }

        return results

    def update(self, sensing_data: SensingSnapshot):
        """Update method called each frame with sensing data"""
        if not self.is_running:
            return

        self.frame_counter += 1

        # Record position
        self.positions.append(tuple(sensing_data.car_position))

        # Get collision count from car
        self.collision_counter = sensing_data.collision_counter

        # Check segment completion (assuming segment completion when checkpoints_passed == total_checkpoints)
        if (
            sensing_data.checkpoints_passed > self.segment + 1
            and sensing_data.total_checkpoints > 0
        ):
            self.segment_completed = True
            held_keys["w"] = False
            held_keys["s"] = False
            held_keys["a"] = False
            held_keys["d"] = False
            self.stop_evaluation()
            return

        # Execute current action every frames
        if self.current_action_index < len(self.action_sequence):
            action = self.action_sequence[self.current_action_index]
            print(f"Action: {action}")
            self._execute_action(action)

            # Move to next action
            self.current_action_index += 1
        else:
            # Action sequence finished - release all keys
            held_keys["w"] = False
            held_keys["s"] = False
            held_keys["a"] = False
            held_keys["d"] = False
            self.stop_evaluation()

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
        segment: int = 0,
        generation: int = 0,
        initial_position: Optional[List[float]] = None,
        initial_angle: Optional[float] = None,
        initial_speed: Optional[float] = None,
        track_name: str = "",
    ):
        self.current_autopilot_index = 0
        self.results = []
        self.checkpoint_handler = None
        self.segment = segment
        self.generation = generation
        self.initial_position = initial_position
        self.initial_angle = initial_angle
        self.initial_speed = initial_speed
        self.track_name = track_name
        self.car = None
        if action_sequences_path:
            self.autopilots = self.load_genetic_autopilots(action_sequences_path)
        elif action_sequences:
            self.autopilots = [
                GeneticAutopilot(seq, self.segment, car=self.car)
                for seq in action_sequences
            ]
        else:
            raise ValueError(
                "Must provide either action_sequences_path or action_sequences"
            )

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

        return [
            GeneticAutopilot(seq, self.segment, car=self.car)
            for seq in action_sequences
        ]

    def load_single_autopilot(self, action_sequence_path: str) -> GeneticAutopilot:
        """Load a single genetic autopilot from a JSON file containing action sequences"""
        if not os.path.exists(action_sequence_path):
            raise FileNotFoundError(
                f"Action sequence file not found: {action_sequence_path}"
            )

        with open(action_sequence_path, "r") as f:
            action_sequence = json.load(f)

        return GeneticAutopilot(action_sequence, self.segment, car=self.car)

    def start_evaluation(self):
        """Start the genetic evaluation for all autopilots"""
        self.current_autopilot_index = 0
        self.results = []
        if self.autopilots:
            for a in self.autopilots:
                a.car = self.car
            self.autopilots[0].start_evaluation()
            self._reset_car_for_individual(0)

    def get_results(self):
        """Get evaluation results for all autopilots"""
        return self.results

    def update(self, sensing_data):
        """Update the current autopilot and handle switching"""
        if self.current_autopilot_index >= len(self.autopilots):
            return

        current_autopilot = self.autopilots[self.current_autopilot_index]
        current_autopilot.update(sensing_data)

        if not current_autopilot.is_running:
            results = current_autopilot.stop_evaluation()
            self.results.append(results)
            print(
                f"INDIVIDUAL {self.current_autopilot_index} RESULTS: {results['collision_counter']} {results['segment_completed']} {results['inputs_to_finish_segment']}"
            )
            print(json.dumps(results["positions"]))

            self.current_autopilot_index += 1
            if self.current_autopilot_index < len(self.autopilots):
                print(f"Switching to individual {self.current_autopilot_index}")
                self.autopilots[self.current_autopilot_index].start_evaluation()
                self._reset_car_for_individual(self.current_autopilot_index)
            else:
                print("ALL_EVALUATIONS_COMPLETE")

    def _reset_car_for_individual(self, idx):
        """Reset the car to initial conditions for a new individual"""
        if not self.car:
            return
        # Set genetic attributes
        self.car.is_genetic_car = True
        self.car.genetic_generation = self.generation
        self.car.genetic_individual = idx
        self.car.genetic_segment = self.segment
        print(
            f"Set genetic car attributes: gen={self.generation}, ind={idx}, seg={self.segment}"
        )
        if self.initial_position:
            self.car.position = ursina.Vec3(*self.initial_position)
        if self.initial_angle is not None:
            self.car.rotation_y = self.initial_angle
        if self.initial_speed is not None:
            self.car.speed = self.initial_speed
        # Reset collision counter
        self.car.reset_collision_counter()
        if self.car.checkpoint_handler:
            num_checkpoints = len(self.car.checkpoint_handler.lap_checkpoints)
            passed = set(range(self.segment + 1))
            self.car.checkpoint_handler.passed_checkpoints = passed
            self.car.checkpoint_handler.next_checkpoint_index = (
                self.segment + 1
            ) % num_checkpoints
            if self.car.checkpoint_handler.next_checkpoint_index == 0:
                self.car.checkpoint_handler.lap_completed_checkpoints = True
            # Update visuals
            for entity_data in self.car.checkpoint_handler.checkpoint_entities:
                cp_id = entity_data["data"]["id"]
                if cp_id in passed:
                    entity_data["passed"] = True
                    entity_data["entity"].color = ursina.color.blue
                    entity_data["text"].color = ursina.color.cyan
                else:
                    entity_data["passed"] = False
                    entity_data["entity"].color = ursina.color.green
                    entity_data["text"].color = ursina.color.yellow

    @property
    def is_running(self):
        """Check if the current evaluation is running"""
        if self.current_autopilot_index >= len(self.autopilots):
            return False
        return self.autopilots[self.current_autopilot_index].is_running

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
        snapshot.collision_counter = self.car.collision_counter if self.car else 0

        # Update current autopilot
        current_autopilot.update(snapshot)

        # Check if current autopilot finished
        if not current_autopilot.is_running:
            # Save results and move to next autopilot
            results = current_autopilot.stop_evaluation()
            self.results.append(results)
            # Print results for GA to parse
            print(
                f"INDIVIDUAL {self.current_autopilot_index} RESULTS: {results['collision_counter']} {results['segment_completed']} {results['inputs_to_finish_segment']}"
            )

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
