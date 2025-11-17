import sys
import os
import json
import ursina
from rallyrobopilot import prepare_game_app
import time
from ursina import held_keys

"""
Replay Autopilot Runner

This script runs a replay autopilot that interacts with the main RallyRobotPilot game.
It loads recorded frames from JSON and replays the inputs while recording new frames.

Usage:
    python replay_autopilot_runner.py <frames.json> <track_name>

The frames.json should contain a list of frame objects with 'input' arrays: [{"input": [fwd, back, left, right], ...}, ...]
"""


class ReplayAutopilot:
    """Simple autopilot that executes action sequences"""

    def __init__(self, action_sequence):
        self.action_sequence = action_sequence
        self.current_action_index = 0
        self.is_running = False

    def start(self):
        self.is_running = True
        self.current_action_index = 0

    def update(self, sensing_data):
        if not self.is_running:
            return

        if self.current_action_index < len(self.action_sequence):
            action = self.action_sequence[self.current_action_index]
            self._execute_action(action)
            self.current_action_index += 1
        else:
            # Release all keys
            held_keys["w"] = False
            held_keys["s"] = False
            held_keys["a"] = False
            held_keys["d"] = False
            self.is_running = False

    def _execute_action(self, action):
        held_keys["w"] = bool(action[0])
        held_keys["s"] = bool(action[1])
        held_keys["a"] = bool(action[2])
        held_keys["d"] = bool(action[3])


class ReplayProcessor:
    """Simple processor for replay autopilot"""

    def __init__(self, action_sequence):
        self.autopilot = ReplayAutopilot(action_sequence)
        self.car = None

    def start_evaluation(self):
        self.autopilot.start()
        if self.car:
            self.car.start_record()

    def stop_evaluation(self):
        if self.car:
            self.car.stop_record()

    @property
    def is_running(self):
        return self.autopilot.is_running

    def update(self, sensing_data):
        self.autopilot.update(sensing_data)


def segment_by_checkpoints(snapshots):
    """
    Segment the snapshots into lists based on checkpoint passages.
    Returns a list of segments, where each segment is the data between checkpoints.
    """
    if not snapshots:
        return []

    segments = []
    current_segment = []
    last_checkpoint_count = 0

    for snapshot in snapshots:
        current_checkpoints = getattr(snapshot, "checkpoints_passed", 0)

        # If checkpoint count increased, start a new segment
        if current_checkpoints > last_checkpoint_count:
            if current_segment:
                segments.append(current_segment)
            current_segment = [snapshot]
            last_checkpoint_count = current_checkpoints
        else:
            current_segment.append(snapshot)

    # Add the last segment
    if current_segment:
        segments.append(current_segment)

    return segments





def main():
    if len(sys.argv) < 3:
        print("Usage: python replay_autopilot_runner.py <track_name> <frames.json>")
        print(
            "Example: python replay_autopilot_runner.py SimpleTrack records/record_0/segments/record_0_segment_0.json"
        )
        sys.exit(1)

    track_name = sys.argv[1]
    frames_file = sys.argv[2]

    if not os.path.exists(frames_file):
        print(f"Error: Frames file '{frames_file}' not found")
        sys.exit(1)

    # Load frames and extract action sequence
    with open(frames_file, "r") as f:
        frames = json.load(f)

    action_sequence = []
    initial_position = None
    initial_angle = None
    initial_speed = None

    for frame in frames:
        if "input" in frame:
            action_sequence.append(tuple(frame["input"]))
        # Get initial state from first frame
        if initial_position is None and "position" in frame:
            pos = frame["position"]
            if isinstance(pos, str):
                pos = json.loads(pos)
            initial_position = pos
        if initial_angle is None and "angle" in frame:
            initial_angle = frame["angle"]
        if initial_speed is None and "speed" in frame:
            initial_speed = frame["speed"]

    if not action_sequence:
        print("Error: No actions found in frames file")
        sys.exit(1)

    print(f"Loaded {len(action_sequence)} actions from {frames_file}")

    # Determine track metadata path
    if "/" in track_name:
        track_metadata = track_name
    else:
        track_metadata = f"{track_name}/track_metadata.json"

    # Create the replay processor
    replay_processor = ReplayProcessor(action_sequence)

    # Prepare the game
    app, car = prepare_game_app(track_metadata)

    # Set initial conditions
    if initial_position:
        car.position = ursina.Vec3(*initial_position)
    if initial_angle is not None:
        car.rotation_y = initial_angle
    if initial_speed is not None:
        car.speed = initial_speed

    car.multiray_sensor = None

    # Set up the car with autopilot
    car.autopilot = replay_processor
    replay_processor.car = car

    replay_processor.start_evaluation()

    # Show checkpoints and rays for visualization
    if car.checkpoint_handler:
        car.checkpoint_handler.show_ui()
    if car.multiray_sensor:
        car.multiray_sensor.set_enabled_rays(True)

    # Run the game
    try:
        FPS = 15
        frame_time = 1 / FPS
        while True:
            start_time = time.time()
            app.step()
            elapsed = time.time() - start_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Check if autopilot finished
            if car.autopilot and not car.autopilot.is_running:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Stop recording
    replay_processor.stop_evaluation()


if __name__ == "__main__":
    main()
