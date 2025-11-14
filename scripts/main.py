import sys
from pathlib import Path
from rallyrobopilot import prepare_game_app, RemoteController
from flask import Flask
from threading import Thread
import time


def get_track_path(track_name, circuit_index=None):
    """Get the full path to the track metadata file."""
    assets_dir = Path(__file__).parent.parent / "assets"
    if circuit_index is None:
        return assets_dir / track_name / "track_metadata.json"
    else:
        return assets_dir / track_name / f"track_circuit{circuit_index}_metadata.json"


def list_available_tracks():
    """List all available tracks."""
    assets_dir = Path(__file__).parent.parent / "assets"
    if not assets_dir.exists():
        return

    print("Available tracks:")
    for track_dir in sorted(assets_dir.iterdir()):
        if track_dir.is_dir():
            track_files = list(track_dir.glob("track*.json"))
            if track_files:
                circuits = []
                has_main = False
                for f in track_files:
                    if f.name == "track_metadata.json":
                        has_main = True
                    elif f.name.startswith("track_circuit") and f.name.endswith(
                        "_metadata.json"
                    ):
                        try:
                            idx = f.name.split("track_circuit")[1].split(
                                "_metadata.json"
                            )[0]
                            circuits.append(int(idx))
                        except:
                            pass
                if has_main:
                    print(f"  {track_dir.name}")
                circuits.sort()
                for idx in circuits:
                    print(f"  {track_dir.name} {idx}")


if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: python main.py <track_name>")
    print("   or: python main.py <track_name> <circuit_index>")
    list_available_tracks()
    sys.exit(1)

track_name = sys.argv[1]
circuit_index = None
if len(sys.argv) == 3:
    try:
        circuit_index = int(sys.argv[2])
    except ValueError:
        print("Error: Circuit index must be an integer")
        sys.exit(1)

track_path = get_track_path(track_name, circuit_index)
if not track_path.exists():
    print(f"Error: Track '{track_path}' not found")
    sys.exit(1)

# Setup Flask
flask_app = Flask(__name__)
flask_thread = Thread(target=flask_app.run, kwargs={"host": "0.0.0.0", "port": 5000})
print("Flask server running on port 5000")
flask_thread.start()

app, car = prepare_game_app(
    str(track_path.relative_to(Path(__file__).parent.parent / "assets"))
)
remote_controller = RemoteController(car=car, connection_port=7654, flask_app=flask_app)

FPS = 15
frame_time = 1 / FPS
while True:
    start_time = time.time()
    app.step()
    elapsed = time.time() - start_time
    sleep_time = frame_time - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
