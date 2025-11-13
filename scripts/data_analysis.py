import lzma
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_record(record_path):
    """Load a record from LZMA compressed pickle file."""
    if not os.path.exists(record_path):
        print(f"Record file {record_path} not found.")
        return None

    with lzma.open(record_path, "rb") as f:
        data = pickle.load(f)
    return data

def analyze_record(data):
    """Analyze the record and return extracted data."""
    if not data:
        return None

    timestamps = []
    speeds = []
    angles = []
    positions_x = []
    positions_y = []
    positions_z = []
    controls_forward = []
    controls_back = []
    controls_left = []
    controls_right = []
    ray_distances = [[] for _ in range(15)]  # Assuming 15 rays

    # Checkpoint data
    checkpoints_passed = []
    total_checkpoints = []
    lap_current = []

    for msg in data:
        timestamps.append(getattr(msg, 'timestamp', 0))

        speeds.append(msg.car_speed)
        angles.append(msg.car_angle)
        positions_x.append(msg.car_position[0])
        positions_y.append(msg.car_position[1])
        positions_z.append(msg.car_position[2])

        controls_forward.append(msg.current_controls[0])
        controls_back.append(msg.current_controls[1])
        controls_left.append(msg.current_controls[2])
        controls_right.append(msg.current_controls[3])

        if msg.raycast_distances:
            for j, dist in enumerate(msg.raycast_distances):
                if j < len(ray_distances):
                    ray_distances[j].append(dist)

        # Checkpoint data
        checkpoints_passed.append(getattr(msg, 'checkpoints_passed', 0))
        total_checkpoints.append(getattr(msg, 'total_checkpoints', 0))
        lap_current.append(getattr(msg, 'lap_current', 0))

    # Handle timestamps - if not available, create synthetic ones assuming 25 FPS
    if not any(t > 0 for t in timestamps):
        # No real timestamps, create synthetic ones
        fps_assumed = 25.0
        delta_t = 1.0 / fps_assumed
        timestamps = [i * delta_t for i in range(len(timestamps))]

    # Calculate time deltas
    if len(timestamps) > 1:
        time_deltas = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        avg_delta = sum(time_deltas) / len(time_deltas)
        fps_actual = 1 / avg_delta if avg_delta > 0 else 0
    else:
        time_deltas = []
        avg_delta = 0
        fps_actual = 0

    # Calculate checkpoint statistics
    checkpoint_changes = []
    if len(checkpoints_passed) > 1:
        for i in range(1, len(checkpoints_passed)):
            if checkpoints_passed[i] > checkpoints_passed[i-1]:
                checkpoint_changes.append((i, checkpoints_passed[i]))

    # Calculate per-checkpoint averages
    checkpoint_segments = []
    if checkpoint_changes:
        # Segment 0: from start to first checkpoint
        start_idx = 0
        for i, (cp_frame, cp_num) in enumerate(checkpoint_changes):
            end_idx = cp_frame
            segment_data = {
                'checkpoint_num': cp_num,
                'start_frame': start_idx,
                'end_frame': end_idx,
                'frame_count': end_idx - start_idx
            }

            # Calculate averages for this segment
            if start_idx < end_idx:
                # Time deltas for this segment
                segment_deltas = time_deltas[start_idx:end_idx-1] if start_idx < len(time_deltas) else []
                if segment_deltas:
                    segment_data['avg_delta'] = sum(segment_deltas) / len(segment_deltas)
                    segment_data['avg_fps'] = 1 / segment_data['avg_delta'] if segment_data['avg_delta'] > 0 else 0
                else:
                    segment_data['avg_delta'] = avg_delta
                    segment_data['avg_fps'] = fps_actual

                # Rolling framerate for this segment
                if len(timestamps) > 10 and end_idx > 10:
                    segment_fps_values = []
                    fps_start = max(10, start_idx + 10)
                    fps_end = min(end_idx, len(timestamps))
                    for j in range(fps_start, fps_end):
                        delta = timestamps[j] - timestamps[j-10]
                        fps = 10 / delta if delta > 0 else 0
                        segment_fps_values.append(fps)
                    if segment_fps_values:
                        segment_data['avg_rolling_fps'] = sum(segment_fps_values) / len(segment_fps_values)
                    else:
                        segment_data['avg_rolling_fps'] = fps_actual
                else:
                    segment_data['avg_rolling_fps'] = fps_actual

            checkpoint_segments.append(segment_data)
            start_idx = end_idx

        # Add final segment if there are frames after the last checkpoint
        if start_idx < len(timestamps):
            segment_data = {
                'checkpoint_num': len(checkpoint_changes) + 1,  # Next checkpoint
                'start_frame': start_idx,
                'end_frame': len(timestamps),
                'frame_count': len(timestamps) - start_idx,
                'avg_delta': avg_delta,
                'avg_fps': fps_actual,
                'avg_rolling_fps': fps_actual
            }
            checkpoint_segments.append(segment_data)

    return {
        'timestamps': timestamps,
        'speeds': speeds,
        'angles': angles,
        'positions_x': positions_x,
        'positions_y': positions_y,
        'positions_z': positions_z,
        'controls_forward': controls_forward,
        'controls_back': controls_back,
        'controls_left': controls_left,
        'controls_right': controls_right,
        'ray_distances': ray_distances,
        'time_deltas': time_deltas,
        'avg_delta': avg_delta,
        'fps_actual': fps_actual,
        'checkpoints_passed': checkpoints_passed,
        'total_checkpoints': total_checkpoints,
        'lap_current': lap_current,
        'checkpoint_changes': checkpoint_changes,
        'checkpoint_segments': checkpoint_segments,
    }

def plot_segment_data(data_dict, segment_info=None):
    """Plot segment-specific graphs when analyzing a checkpoint segment."""
    if not segment_info:
        return

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f'Checkpoint Segment Analysis: {segment_info}')

    # Speed profile within segment
    axes[0, 0].plot(data_dict['timestamps'], data_dict['speeds'], 'b-', linewidth=2)
    axes[0, 0].set_title('Speed Profile in Segment')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Speed')
    axes[0, 0].grid(True, alpha=0.3)

    # Control inputs within segment
    axes[0, 1].plot(data_dict['timestamps'], data_dict['controls_forward'], label='Forward', alpha=0.7)
    axes[0, 1].plot(data_dict['timestamps'], data_dict['controls_back'], label='Back', alpha=0.7)
    axes[0, 1].plot(data_dict['timestamps'], data_dict['controls_left'], label='Left', alpha=0.7)
    axes[0, 1].plot(data_dict['timestamps'], data_dict['controls_right'], label='Right', alpha=0.7)
    axes[0, 1].set_title('Control Inputs in Segment')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Pressed (1/0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Position trajectory within segment
    axes[1, 0].plot(data_dict['positions_x'], data_dict['positions_z'], 'g-', linewidth=2, alpha=0.8)
    axes[1, 0].scatter(data_dict['positions_x'][0], data_dict['positions_z'][0], color='red', s=50, label='Start', zorder=5)
    axes[1, 0].scatter(data_dict['positions_x'][-1], data_dict['positions_z'][-1], color='blue', s=50, label='End', zorder=5)
    axes[1, 0].set_title('Position Trajectory in Segment')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Z Position')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')

    # Raycast distances (first 5 rays) within segment
    for i in range(min(5, len(data_dict['ray_distances']))):
        if i < len(data_dict['ray_distances']) and data_dict['ray_distances'][i]:
            axes[1, 1].plot(data_dict['timestamps'], data_dict['ray_distances'][i], label=f'Ray {i}', alpha=0.7)
    axes[1, 1].set_title('Raycast Sensors in Segment')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Distance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Time deltas within segment
    if data_dict['time_deltas']:
        axes[2, 0].plot(data_dict['time_deltas'], 'r-', alpha=0.8)
        axes[2, 0].axhline(y=data_dict['avg_delta'], color='blue', linestyle='--', label=f'Avg: {data_dict["avg_delta"]:.4f}s')
        axes[2, 0].set_title('Frame Timing in Segment')
        axes[2, 0].set_xlabel('Frame')
        axes[2, 0].set_ylabel('Delta Time (s)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

    # Segment performance summary
    axes[2, 1].axis('off')
    segment_time = data_dict['timestamps'][-1] - data_dict['timestamps'][0] if data_dict['timestamps'] else 0
    avg_speed = sum(data_dict['speeds']) / len(data_dict['speeds']) if data_dict['speeds'] else 0
    total_distance = 0
    if len(data_dict['positions_x']) > 1:
        for i in range(1, len(data_dict['positions_x'])):
            dx = data_dict['positions_x'][i] - data_dict['positions_x'][i-1]
            dz = data_dict['positions_z'][i] - data_dict['positions_z'][i-1]
            total_distance += (dx**2 + dz**2)**0.5

    summary_text = ".2f"".2f"".2f"".2f"".2f"f"""
    Segment Performance Summary:

    Duration: {segment_time:.2f} seconds
    Frames: {len(data_dict['speeds'])}
    Average Speed: {avg_speed:.2f}
    Total Distance: {total_distance:.2f}
    Average FPS: {data_dict['fps_actual']:.2f}
    Average Delta: {data_dict['avg_delta']:.4f}s

    Checkpoints: {max(data_dict['checkpoints_passed']) if data_dict['checkpoints_passed'] else 0}
    """

    axes[2, 1].text(0.1, 0.9, summary_text, transform=axes[2, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_data(data_dict):
    """Plot various graphs from the data."""
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    fig.suptitle('Rally Robot Pilot Data Analysis')

    # Speed over time
    axes[0, 0].plot(data_dict['timestamps'], data_dict['speeds'])
    axes[0, 0].set_title('Car Speed over Time')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Speed')

    # Speed histogram
    axes[0, 1].hist(data_dict['speeds'], bins=20)
    axes[0, 1].set_title('Speed Distribution')
    axes[0, 1].set_xlabel('Speed')
    axes[0, 1].set_ylabel('Frequency')

    # Time deltas
    if data_dict['time_deltas']:
        axes[1, 0].plot(data_dict['time_deltas'])
        axes[1, 0].set_title('Time Deltas between Frames')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Delta Time (s)')
        axes[1, 0].axhline(y=data_dict['avg_delta'], color='r', linestyle='--', label=f'Avg: {data_dict["avg_delta"]:.4f}s')
        axes[1, 0].legend()

    # Framerate over time (rolling average)
    if len(data_dict['timestamps']) > 10:
        fps_over_time = []
        for i in range(10, len(data_dict['timestamps'])):
            delta = data_dict['timestamps'][i] - data_dict['timestamps'][i-10]
            fps = 10 / delta if delta > 0 else 0
            fps_over_time.append(fps)
        axes[1, 1].plot(data_dict['timestamps'][10:], fps_over_time)
        axes[1, 1].set_title('Framerate over Time (Rolling 10-frame avg)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('FPS')
        axes[1, 1].axhline(y=data_dict['fps_actual'], color='r', linestyle='--', label=f'Overall: {data_dict["fps_actual"]:.2f} FPS')
        axes[1, 1].legend()

    # Time delta histogram (frequency bin graph)
    if data_dict['time_deltas']:
        axes[2, 0].hist(data_dict['time_deltas'], bins=20)
        axes[2, 0].set_title('Time Delta Distribution')
        axes[2, 0].set_xlabel('Delta Time (s)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].axvline(x=data_dict['avg_delta'], color='r', linestyle='--', label=f'Avg: {data_dict["avg_delta"]:.4f}s')
        axes[2, 0].legend()

    # Framerate histogram (frequency bin graph)
    if len(data_dict['timestamps']) > 10:
        fps_values = []
        for i in range(10, len(data_dict['timestamps'])):
            delta = data_dict['timestamps'][i] - data_dict['timestamps'][i-10]
            fps = 10 / delta if delta > 0 else 0
            fps_values.append(fps)
        axes[2, 1].hist(fps_values, bins=20)
        axes[2, 1].set_title('Framerate Distribution')
        axes[2, 1].set_xlabel('FPS')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].axvline(x=data_dict['fps_actual'], color='r', linestyle='--', label=f'Overall: {data_dict["fps_actual"]:.2f} FPS')
        axes[2, 1].legend()

    # Checkpoint progress over time
    axes[3, 0].plot(data_dict['timestamps'], data_dict['checkpoints_passed'], 'b-', linewidth=2, label='Checkpoints Passed')
    if data_dict['total_checkpoints'] and any(cp > 0 for cp in data_dict['total_checkpoints']):
        axes[3, 0].plot(data_dict['timestamps'], data_dict['total_checkpoints'], 'r--', alpha=0.7, label='Total Checkpoints')
    axes[3, 0].set_title('Checkpoint Progress over Time')
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_ylabel('Checkpoints')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)

    # Mark checkpoint passages
    for frame_idx, cp_num in data_dict['checkpoint_changes']:
        axes[3, 0].axvline(x=data_dict['timestamps'][frame_idx], color='green', linestyle=':', alpha=0.7, linewidth=1)
        axes[3, 0].text(data_dict['timestamps'][frame_idx], cp_num + 0.5, f'CP{cp_num}', ha='center', va='bottom', fontsize=8)

    # Framerate average per checkpoint segment
    if data_dict['checkpoint_segments']:
        segment_labels = []
        for i, seg in enumerate(data_dict['checkpoint_segments']):
            if i == 0:
                segment_labels.append("Start→CP1")
            elif i < len(data_dict['checkpoint_changes']):
                segment_labels.append(f"CP{i}→CP{i+1}")
            else:
                segment_labels.append(f"CP{i}→End")

        avg_fps_values = [seg['avg_rolling_fps'] for seg in data_dict['checkpoint_segments']]
        axes[3, 1].bar(segment_labels, avg_fps_values, alpha=0.7, color='blue')
        axes[3, 1].set_title('Average Framerate per Checkpoint Segment')
        axes[3, 1].set_xlabel('Track Segment')
        axes[3, 1].set_ylabel('Average FPS')
        axes[3, 1].axhline(y=data_dict['fps_actual'], color='r', linestyle='--', alpha=0.7, label=f'Overall: {data_dict["fps_actual"]:.2f} FPS')
        axes[3, 1].legend()
        axes[3, 1].grid(True, alpha=0.3)
        axes[3, 1].tick_params(axis='x', rotation=45)

    # Time delta (frequency) average per checkpoint segment
    if data_dict['checkpoint_segments']:
        segment_labels = []
        for i, seg in enumerate(data_dict['checkpoint_segments']):
            if i == 0:
                segment_labels.append("Start→CP1")
            elif i < len(data_dict['checkpoint_changes']):
                segment_labels.append(f"CP{i}→CP{i+1}")
            else:
                segment_labels.append(f"CP{i}→End")

        avg_delta_values = [seg['avg_delta'] for seg in data_dict['checkpoint_segments']]
        axes[4, 0].bar(segment_labels, avg_delta_values, alpha=0.7, color='green')
        axes[4, 0].set_title('Average Time Delta per Checkpoint Segment')
        axes[4, 0].set_xlabel('Track Segment')
        axes[4, 0].set_ylabel('Average Delta Time (s)')
        axes[4, 0].axhline(y=data_dict['avg_delta'], color='r', linestyle='--', alpha=0.7, label=f'Overall: {data_dict["avg_delta"]:.4f}s')
        axes[4, 0].legend()
        axes[4, 0].grid(True, alpha=0.3)
        axes[4, 0].tick_params(axis='x', rotation=45)

    # Hide the empty subplot
    axes[4, 1].set_visible(False)

    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_analysis.py <record_file>")
        print("Example: python scripts/data_analysis.py image_records/record_0/record_0.npz")
        print("         python scripts/data_analysis.py image_records/record_0/segments/record_0_segment_0.npz")
        sys.exit(1)

    record_path = sys.argv[1]
    data = load_record(record_path)
    if data is None:
        sys.exit(1)

    print(f"Loaded {len(data)} snapshots from {record_path}")

    analyzed_data = analyze_record(data)
    if analyzed_data:
        print(f"Average time delta: {analyzed_data['avg_delta']:.4f}s")
        print(f"Actual FPS: {analyzed_data['fps_actual']:.2f}")

        # Check if this is a segment file
        is_segment = 'segment' in record_path

        if is_segment:
            # Extract segment info from filename
            import os
            filename = os.path.basename(record_path)
            segment_info = filename.replace('.npz', '').replace('_', ' ').title()
            print(f"\nAnalyzing segment: {segment_info}")

            # Segment-specific statistics
            segment_time = analyzed_data['timestamps'][-1] - analyzed_data['timestamps'][0] if analyzed_data['timestamps'] else 0
            avg_speed = sum(analyzed_data['speeds']) / len(analyzed_data['speeds']) if analyzed_data['speeds'] else 0
            print(f"Segment duration: {segment_time:.2f}s")
            print(f"Average speed: {avg_speed:.2f}")
            print(f"Frames in segment: {len(analyzed_data['speeds'])}")

            plot_segment_data(analyzed_data, segment_info)
        else:
            # Full recording analysis
            # Checkpoint statistics
            max_checkpoints = max(analyzed_data['checkpoints_passed']) if analyzed_data['checkpoints_passed'] else 0
            total_checkpoints = analyzed_data['total_checkpoints'][-1] if analyzed_data['total_checkpoints'] else 0
            max_lap = max(analyzed_data['lap_current']) if analyzed_data['lap_current'] else 0
            checkpoint_changes = len(analyzed_data['checkpoint_changes'])

            print(f"Checkpoints passed: {max_checkpoints}/{total_checkpoints}")
            print(f"Total checkpoint passages: {checkpoint_changes}")
            print(f"Laps completed: {max_lap}")

            if analyzed_data['checkpoint_changes']:
                print("Checkpoint passages at frames:")
                for frame_idx, cp_num in analyzed_data['checkpoint_changes']:
                    time_at_cp = analyzed_data['timestamps'][frame_idx]
                    print(f"  Checkpoint {cp_num} at frame {frame_idx} (t={time_at_cp:.2f}s)")

            plot_data(analyzed_data)
    else:
        print("Failed to analyze data")

if __name__ == "__main__":
    main()