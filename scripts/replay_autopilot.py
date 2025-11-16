import json
import time
import os
from typing import List, Dict, Any
import ursina
from ursina import held_keys


class ReplayAutopilot:
    """Autopilot that replays recorded frames"""

    def __init__(self, frames: List[Dict[str, Any]], car=None, on_finish=None):
        self.frames = frames
        self.car = car
        self.current_frame_index = 0
        self.is_running = False
        self.frame_rate = 15  # FPS, adjust as needed
        self.frame_interval = 1.0 / self.frame_rate
        self.start_time = 0.0
        self.on_finish = on_finish

        print(f"Loaded replay with {len(self.frames)} frames")

    def start_replay(self):
        """Start the replay"""
        self.is_running = True
        self.current_frame_index = 0
        self.start_time = time.time()

    def stop_replay(self):
        """Stop the replay"""
        self.is_running = False
        # Release all keys
        held_keys["w"] = False
        held_keys["s"] = False
        held_keys["a"] = False
        held_keys["d"] = False
        if self.on_finish:
            self.on_finish()

    def update(self):
        """Update method called each frame"""
        if not self.is_running or self.current_frame_index >= len(self.frames):
            if self.is_running:
                self.stop_replay()
                print("Replay finished")
            return

        current_time = time.time()
        elapsed = current_time - self.start_time
        target_frame = int(elapsed / self.frame_interval)

        if target_frame > self.current_frame_index:
            # Advance to the target frame
            self.current_frame_index = min(target_frame, len(self.frames) - 1)
            self._apply_frame(self.frames[self.current_frame_index])

    def _apply_frame(self, frame: Dict[str, Any]):
        """Apply a single frame's state to the car"""
        if not self.car:
            return

        # Set position if available
        if 'position' in frame:
            pos = frame['position']
            if isinstance(pos, str):
                pos = json.loads(pos)
            if isinstance(pos, list):
                self.car.position = ursina.Vec3(*pos)
            elif isinstance(pos, dict):
                self.car.position = ursina.Vec3(pos['x'], pos['y'], pos['z'])

        # Set rotation if available
        if 'angle' in frame:
            self.car.rotation_y = frame['angle']

        # Set speed if available
        if 'speed' in frame:
            self.car.speed = frame['speed']

        # Set controls (held keys) if available
        if 'input' in frame:
            controls = frame['input']
            held_keys['w'] = bool(controls[0])  # forward
            held_keys['s'] = bool(controls[1])  # back
            held_keys['a'] = bool(controls[2])  # left
            held_keys['d'] = bool(controls[3])  # right


class ReplayMsgProcessor:
    """Message processor for replay autopilot integration with DataCollectionUI"""

    def __init__(self, replay_file: str, car=None):
        self.replay_file = replay_file
        self.car = car
        self.replay_autopilot = None
        self.data_collector = None
        self.recording_started = False
        self.load_replay()

    def load_replay(self):
        """Load replay data from JSON file"""
        if not os.path.exists(self.replay_file):
            raise FileNotFoundError(f"Replay file not found: {self.replay_file}")

        with open(self.replay_file, 'r') as f:
            frames = json.load(f)

        # Set initial car position to the first frame
        if self.car and frames:
            first_frame = frames[0]
            if 'position' in first_frame:
                pos = first_frame['position']
                if isinstance(pos, str):
                    pos = json.loads(pos)
                if isinstance(pos, list):
                    self.car.position = ursina.Vec3(*pos)
            if 'angle' in first_frame:
                self.car.rotation_y = first_frame['angle']

        self.replay_autopilot = ReplayAutopilot(frames, car=self.car, on_finish=self.on_replay_finish)
        print(f"Loaded replay from {self.replay_file}")

    def on_replay_finish(self):
        """Called when replay finishes"""
        if self.data_collector:
            self.data_collector.saveRecord()
            print("Replay finished, saving record...")
            # Exit after saving
            import sys
            sys.exit(0)

    def start_replay(self):
        """Start the replay"""
        if self.replay_autopilot:
            self.replay_autopilot.start_replay()

    def process_message(self, message, data_collector):
        """Process sensing messages for replay"""
        self.data_collector = data_collector

        # Start recording on first message
        if not self.recording_started:
            self.recording_started = True
            data_collector.toggleRecord()
            print("Started recording for replay")

        if self.replay_autopilot:
            self.replay_autopilot.update()

            # Send current controls to data collector for recording
            # Since we're setting held_keys, the controls should be reflected in the message
            if hasattr(message, 'current_controls'):
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
    import sys
    from PyQt6 import QtWidgets
    from data_collector import DataCollectionUI

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook

    if len(sys.argv) < 2:
        print("Usage: python replay_autopilot.py <replay_file.json>")
        sys.exit(1)

    replay_file = sys.argv[1]
    processor = ReplayMsgProcessor(replay_file)

    app = QtWidgets.QApplication(sys.argv)

    data_window = DataCollectionUI(processor.process_message)
    data_window.show()

    # Start replay when the window is shown
    processor.start_replay()

    app.exec()