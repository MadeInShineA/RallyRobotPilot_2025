import sys
import os
import lzma
import pickle
import json
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
            "Usage: python genetic_autopilot_runner.py <action_sequences.json> <track_name> [--batch] [--start_segment <x>] [--base_record <record>] [replay_file]"
        )
        print(
            "Example: python genetic_autopilot_runner.py genetic_data/SimpleTrack_population.json SimpleTrack --batch --start_segment 4 --base_record 0 records/record_0.npz"
        )
        sys.exit(1)

    action_sequences_path = sys.argv[1]
    track_name = sys.argv[2]

    # Parse optional arguments
    replay_file = None
    initial_angle = None
    initial_speed = None
    initial_position = None
    start_segment = None
    generation = 0
    base_record = None
    port = 7654

    args = sys.argv[3:]

    i = 0
    while i < len(args):
        if args[i] == "--initial_angle":
            initial_angle = float(args[i + 1])
            i += 2
        elif args[i] == "--initial_speed":
            initial_speed = float(args[i + 1])
            i += 2
        elif args[i] == "--initial_position":
            initial_position = json.loads(args[i + 1])
            i += 2
        elif args[i] == "--start_segment":
            start_segment = int(args[i + 1])
            i += 2
        elif args[i] == "--generation":
            generation = int(args[i + 1])
            i += 2
        elif args[i] == "--base_record":
            base_record = args[i + 1]
            i += 2
        elif args[i] == "--port":
            port = int(args[i + 1])
            i += 2
        else:
            if replay_file is None:
                replay_file = args[i]
            i += 1

    if start_segment is None:
        start_segment = 0

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
            action_sequences_path=action_sequences_path,
            segment=start_segment,
            generation=generation,
            initial_position=initial_position,
            initial_angle=initial_angle,
            initial_speed=initial_speed,
            track_name=track_name,
            base_record=base_record,
        )
        print(f"Loaded {len(genetic_processor.autopilots)} genetic autopilots")
    except Exception as e:
        print(f"Error loading genetic autopilots: {e}")
        sys.exit(1)



    # Prepare the game
    app, car = prepare_game_app(track_metadata)

    # Set initial conditions if provided
    if initial_position:
        car.position = ursina.Vec3(*initial_position)
    if initial_angle is not None:
        car.rotation_y = initial_angle
    if initial_speed is not None:
        car.speed = initial_speed

    car.multiray_sensor = None
    # Set up Flask and remote controller like main.py
    flask_app = Flask(__name__)
    remote_controller = RemoteController(
        car=car, connection_port=port, flask_app=flask_app
    )

    # Set up the car with autopilot
    car.autopilot = genetic_processor
    genetic_processor.car = car
    genetic_processor.start_evaluation()

    # Show checkpoints and rays for visualization
    if car.checkpoint_handler:
        car.checkpoint_handler.ui_enabled = True
        car.checkpoint_handler.show_ui()
    if car.multiray_sensor:
        car.multiray_sensor.set_enabled_rays(True)

    # Set initial checkpoint state if start_segment is specified
    if start_segment is not None and car.checkpoint_handler and car.checkpoint_handler.lap_checkpoints:
        num_checkpoints = len(car.checkpoint_handler.lap_checkpoints)
        if 0 <= start_segment < num_checkpoints:
            # For segment x, passed checkpoints 0 to x, next is x+1
            car.checkpoint_handler.passed_checkpoints = set(range(start_segment + 1))
            # Set next checkpoint to (start_segment + 1) % num_checkpoints
            car.checkpoint_handler.next_checkpoint_index = (start_segment + 1) % num_checkpoints
            # If next is 0, all checkpoints are passed
            if car.checkpoint_handler.next_checkpoint_index == 0:
                car.checkpoint_handler.lap_completed_checkpoints = True

            # Update visual entities
            for entity_data in car.checkpoint_handler.checkpoint_entities:
                cp_id = entity_data["data"]["id"]
                if cp_id in car.checkpoint_handler.passed_checkpoints:
                    entity_data["passed"] = True
                    entity_data["entity"].color = ursina.color.blue
                    entity_data["text"].color = ursina.color.cyan
                else:
                    entity_data["passed"] = False
                    entity_data["entity"].color = ursina.color.green
                    entity_data["text"].color = ursina.color.yellow

            print(f"Initialized for segment {start_segment}: passed 0-{start_segment}, next is {car.checkpoint_handler.next_checkpoint_index}")
        else:
            print(f"Warning: start_segment {start_segment} is out of range (0-{num_checkpoints-1})")

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
            # Check if autopilot finished
            if car.autopilot and not car.autopilot.is_running:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")


if __name__ == "__main__":
    main()
