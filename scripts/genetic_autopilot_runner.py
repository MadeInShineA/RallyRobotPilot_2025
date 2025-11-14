import sys
import os
import lzma
import pickle
import ursina
from rallyrobopilot import prepare_game_app, RemoteController
from flask import Flask
from genetic_autopilot import GeneticMsgProcessor
import time

"""
Genetic Autopilot Runner

This script runs a genetic algorithm autopilot that interacts with the main RallyRobotPilot game.
It loads action sequences from genetic algorithm results and executes them while collecting
fitness data (wall hits, checkpoints passed, completion time).

Usage:
    python genetic_autopilot_runner.py <action_sequence.json>

The action_sequence.json should contain a list of action tuples: [[fwd, back, left, right], ...]
"""


def load_replay_positions(replay_file):
    """Load car positions from replay file"""
    with lzma.open(replay_file, "rb") as f:
        data = pickle.load(f)
    positions = []
    for msg in data:
        positions.append(tuple(msg.car_position))
    return positions


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python genetic_autopilot_runner.py <action_sequences.json> <track_name> [--batch] [replay_file]"
        )
        print(
            "Example: python genetic_autopilot_runner.py genetic_data/SimpleTrack_population.json SimpleTrack --batch records/record_0.npz"
        )
        sys.exit(1)

    action_sequences_path = sys.argv[1]
    track_name = sys.argv[2]
    batch_mode = "--batch" in sys.argv

    # Parse optional replay_file
    replay_file = None
    args = sys.argv[3:]
    if batch_mode:
        args = args[1:]  # Skip --batch
    if args:
        replay_file = args[0]

    if not os.path.exists(action_sequences_path):
        print(f"Error: Action sequences file '{action_sequences_path}' not found")
        sys.exit(1)

    # Determine track metadata path
    if "/" in track_name:
        track_metadata = track_name
    else:
        track_metadata = f"{track_name}/track_metadata.json"

    # Create the genetic message processor
    try:
        genetic_processor = GeneticMsgProcessor(
            action_sequences_path=action_sequences_path
        )
        print(f"Loaded {len(genetic_processor.autopilots)} genetic autopilots")
    except Exception as e:
        print(f"Error loading genetic autopilots: {e}")
        sys.exit(1)

    # For non-batch, run only the first autopilot
    if not batch_mode:
        genetic_processor.autopilots = genetic_processor.autopilots[:1]

    # Prepare the game
    app, car = prepare_game_app(track_metadata)

    # Set up Flask and remote controller like main.py
    flask_app = Flask(__name__)
    remote_controller = RemoteController(
        car=car, connection_port=7654, flask_app=flask_app
    )

    # Set up the car with autopilot
    car.batch_mode = batch_mode
    car.autopilot = genetic_processor.autopilots[0]
    genetic_processor.autopilots[0].start_evaluation()

    # Show checkpoints and rays for visualization
    if car.checkpoint_handler:
        car.checkpoint_handler.show_ui()
    if car.multiray_sensor:
        car.multiray_sensor.set_enabled_rays(True)

    # Display initial replay trajectory if provided
    if replay_file:
        if os.path.exists(replay_file):
            positions = load_replay_positions(replay_file)
            print(f"Displaying replay trajectory with {len(positions)} points")
            for pos in positions:
                ursina.Entity(
                    model="sphere",
                    scale=0.5,
                    position=pos,
                    color=ursina.color.blue,
                    collider=None,
                )
        else:
            print(f"Warning: Replay file '{replay_file}' not found")

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
    except KeyboardInterrupt:
        print("\nInterrupted by user")


if __name__ == "__main__":
    main()
