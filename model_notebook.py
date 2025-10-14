import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import lzma
    import pickle
    import os
    import polars as pl
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        confusion_matrix,
    )
    import joblib
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    return (
        DataLoader,
        MinMaxScaler,
        TensorDataset,
        confusion_matrix,
        joblib,
        lzma,
        mo,
        nn,
        np,
        optim,
        os,
        pickle,
        pl,
        plt,
        torch,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""## Data cleaning""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Load the different records""")
    return


@app.cell
def _(lzma, os, pickle, pl):
    record_dir = "./records/"

    all_records = []  # Accumulate all records here

    for filename in os.listdir(record_dir):
        if not filename.endswith(".npz"):
            continue

        input_path = os.path.join(record_dir, filename)
        try:
            with lzma.open(input_path, "rb") as f:
                snapshots = pickle.load(f)
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            continue

        for _idx, s in enumerate(snapshots):
            record = {
                "record": filename,
                "frame_idx": _idx,
                "forward": s.current_controls[0],
                "back": s.current_controls[1],
                "left": s.current_controls[2],
                "right": s.current_controls[3],
                "car_speed": s.car_speed,
                **{f"raycast_{i}": float(d) for i, d in enumerate(s.raycast_distances)},
            }
            all_records.append(record)

    # Create a single Polars DataFrame from all records
    try:
        df = pl.DataFrame(all_records).sort("record")
        print(f"✅ Successfully created combined Polars DataFrame with {len(df)} rows")
    except Exception as e:
        print(f"❌ Error creating combined DataFrame: {e}")

    df.head(10)
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""### Clean the first frames of each record when nothing happens (all inputs are 0)""")
    return


@app.cell
def _(df, pl):
    # Here de filter the cols
    df_first_frames_cleaned = df_filtered = df.filter(
        (
            (pl.col("forward") != 0)
            | (pl.col("back") != 0)
            | (pl.col("left") != 0)
            | (pl.col("right") != 0)
        )
        .cum_max()
        .over("record")
    )

    df_first_frames_cleaned.head()
    return (df_first_frames_cleaned,)


@app.cell
def _(mo):
    mo.md(r"""### Filter out the short records""")
    return


@app.cell
def _(df_first_frames_cleaned, pl):
    records_lengths = df_first_frames_cleaned.group_by("record").agg(pl.len())
    records_lengths
    return


@app.cell
def _(df_first_frames_cleaned, pl):
    min_length = 100

    df_lengths_filtered = df_first_frames_cleaned.filter(
        pl.len().over("record") >= min_length
    )
    return (df_lengths_filtered,)


@app.cell
def _(mo):
    mo.md(r"""## Data exploration""")
    return


@app.cell
def _(axes, df_lengths_filtered, np, pl, plt):
    # Get unique filenames
    records = df_lengths_filtered["record"].unique().to_list()

    n_files = len(records)
    cols = 2
    rows = (n_files + cols - 1) // cols  # ceil division

    _fig, _axes = plt.subplots(
        rows, cols, figsize=(12, 4 * rows), sharex=False, sharey=True
    )
    _axes = _axes.flatten() if n_files > 1 else [axes]

    for idx, fname in enumerate(records):
        _ax = _axes[idx]
        subset = df_lengths_filtered.filter(pl.col("record") == fname).sort("frame_idx")
        frames = subset["frame_idx"].to_numpy()

        # Stack the controls
        controls = np.vstack(
            [
                subset["forward"].to_numpy(),
                subset["back"].to_numpy(),
                subset["left"].to_numpy(),
                subset["right"].to_numpy(),
            ]
        )

        _ax.stackplot(
            frames,
            controls,
            labels=["Forward", "Back", "Left", "Right"],
            colors=["green", "red", "blue", "orange"],
            alpha=0.7,
        )

        _ax.set_title(f"Controls over time: {fname}")
        _ax.set_xlabel("Frame")
        _ax.set_ylabel("Active Controls")
        _ax.legend(loc="upper right")
        _ax.set_ylim(0, None)  # Ensure baseline at 0
        _ax.grid(True, linestyle="--", alpha=0.5)

    # Hide unused subplots
    for _j in range(idx + 1, len(_axes)):
        _fig.delaxes(_axes[_j])

    plt.tight_layout()
    plt.show()
    return (records,)


@app.cell
def _(df_lengths_filtered, np, pl, plt):
    # Compute usage per file: mean of each control (assuming 0/1 or [0,1] signals)
    usage_df = (
        df_lengths_filtered.group_by("record")
        .agg(
            pl.col("forward").mean().alias("forward_usage"),
            pl.col("back").mean().alias("back_usage"),
            pl.col("left").mean().alias("left_usage"),
            pl.col("right").mean().alias("right_usage"),
        )
        .sort("record")
    )

    # Prepare data for heatmap
    control_cols = ["forward_usage", "back_usage", "left_usage", "right_usage"]
    usage_matrix = usage_df.select(control_cols).to_numpy()  # shape: (n_files, 4)
    records_sorted = usage_df["record"].to_list()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(records_sorted))))
    im = ax.imshow(usage_matrix, cmap="Blues", aspect="auto")

    # Set ticks
    ax.set_yticks(np.arange(len(records_sorted)))
    ax.set_yticklabels(records_sorted)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Forward", "Back", "Left", "Right"])

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Fraction of time active")

    # Add text annotations (optional)
    for _i in range(len(records_sorted)):
        for _j in range(4):
            text = ax.text(
                _j,
                _i,
                f"{usage_matrix[_i, _j]:.2f}",
                ha="center",
                va="center",
                color="black" if usage_matrix[_i, _j] < 0.5 else "white",
            )

    ax.set_title("Control Usage per Record")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_lengths_filtered, np, pl, plt):
    # 1. Compute min_ray
    ray_cols = [
        col for col in df_lengths_filtered.columns if col.startswith("raycast_")
    ]
    min_rays_df = df_lengths_filtered.with_columns(min_ray=pl.min_horizontal(ray_cols))

    # 2. Extract data
    x = min_rays_df["min_ray"].to_numpy()
    y = min_rays_df["car_speed"].to_numpy()

    # 3. Remove invalid values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    # 4. Bin the data
    n_bins = 30
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    medians = []
    q25 = []
    q75 = []

    for _i in range(n_bins):
        mask = (x >= bins[_i]) & (x < bins[_i + 1])
        if np.any(mask):
            speeds = y[mask]
            medians.append(np.median(speeds))
            q25.append(np.percentile(speeds, 25))
            q75.append(np.percentile(speeds, 75))
        else:
            medians.append(np.nan)
            q25.append(np.nan)
            q75.append(np.nan)

    # 5. Plot
    plt.figure(figsize=(8, 5))

    # Optional: light scatter for context (reduce opacity further)
    plt.scatter(x, y, alpha=0.15, s=8, color="gray", edgecolors="none", label="Frames")

    # Median line
    plt.plot(bin_centers, medians, color="red", linewidth=2.5, label="Median speed")

    # IQR band (25th–75th percentile)
    plt.fill_between(
        bin_centers, q25, q75, color="red", alpha=0.2, label="25th–75th percentile"
    )

    # Labels & styling
    plt.xlabel("Minimum Raycast Distance", fontsize=12)
    plt.ylabel("Car Speed", fontsize=12)
    plt.title("Car Speed vs. Proximity to Obstacles", fontsize=13, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return (ray_cols,)


@app.cell
def _(df_lengths_filtered, np, pl, plt):
    raycast_plot_df = df_lengths_filtered.sort(["record", "frame_idx"])
    raycast_plot_df = raycast_plot_df.with_columns(
        forward_prev=pl.col("forward").shift(1).over("record"),
        raycasts=pl.concat_list(
            [pl.col(c) for c in raycast_plot_df.columns if c.startswith("raycast_")]
        ),
    )

    # Trigger: forward=1 and forward_prev=0
    triggers = raycast_plot_df.filter(
        (pl.col("forward") == 1) & (pl.col("forward_prev") == 0)
    )

    # Convert raycasts to numpy and plot
    ray_matrix = np.vstack(triggers["raycasts"].to_numpy())
    plt.imshow(ray_matrix, aspect="auto", cmap="viridis")
    plt.xlabel("Ray Index")
    plt.ylabel("Forward-Onset Event")
    plt.title("Raycast Profile at Start of Acceleration")
    return


@app.cell
def _(mo):
    mo.md(r"""## Data separation (test, train) and preparation (scalling)""")
    return


@app.cell
def _(
    MinMaxScaler,
    df_lengths_filtered,
    joblib,
    os,
    pl,
    ray_cols,
    records,
    train_test_split,
):
    # --- 1. Split by RECORD first
    train_recs, test_recs = train_test_split(records, test_size=0.2, random_state=42)

    df_train = df_lengths_filtered.filter(pl.col("record").is_in(train_recs))
    df_test = df_lengths_filtered.filter(pl.col("record").is_in(test_recs))

    # --- 2. Prepare feature and label columns
    feature_cols = ray_cols + ["car_speed"]

    # --- 3. Create X (inputs) as Polars DataFrames
    X_train = df_train.select(feature_cols)
    X_test = df_test.select(feature_cols)

    # --- 4. Scale inputs and convert back to Polars DataFrames
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)  # returns np.ndarray
    X_test_scaled = scaler_X.transform(X_test)

    # Convert back to Polars with original column names
    X_train_scaled = pl.DataFrame(X_train_scaled, schema=feature_cols)
    X_test_scaled = pl.DataFrame(X_test_scaled, schema=feature_cols)

    # --- 5. Encode labels as Polars DataFrames (throttle, steer)
    y_train = df_train.select(
        [
            (pl.col("forward") - pl.col("back")).alias("throttle (forward - back)"),
            (pl.col("right") - pl.col("left")).alias("steer (right - left"),
        ]
    )

    y_test = df_test.select(
        [
            (pl.col("forward") - pl.col("back")).alias("throttle (forward - back)"),
            (pl.col("right") - pl.col("left")).alias("steer (right - left)"),
        ]
    )

    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler_X, "model/scaler_X.joblib")
    return X_test_scaled, X_train, X_train_scaled, y_test, y_train


@app.cell
def _(X_train_scaled):
    X_train_scaled
    return


@app.cell
def _(y_train):
    y_train
    return


@app.cell
def _(mo):
    mo.md(r"""## Building prediction nn""")
    return


@app.cell
def _(
    DataLoader,
    TensorDataset,
    X_test_scaled,
    X_train,
    X_train_scaled,
    nn,
    np,
    optim,
    torch,
    y_test,
    y_train,
):
    # --- Convert to tensors ---
    X_train_torch = torch.tensor(X_train_scaled.to_numpy(), dtype=torch.float32)
    y_train_torch = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    X_test_torch = torch.tensor(X_test_scaled.to_numpy(), dtype=torch.float32)
    y_test_torch = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    # --- Convert labels to classes ( -1 -> 0, 0 -> 1, 1 -> 2 ) ---
    y_train_classes = (y_train_torch + 1).long()
    y_test_classes = (y_test_torch + 1).long()

    # --- Create datasets & loaders ---
    train_dataset = TensorDataset(X_train_torch, y_train_classes)
    test_dataset = TensorDataset(X_test_torch, y_test_classes)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # --- Define model ---
    class DrivingPolicy(nn.Module):
        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 6),  # 2 outputs * 3 classes
            )

        def forward(self, x):
            out = self.net(x)
            return out.view(-1, 2, 3)  # (batch, 2, 3)

    # --- Initialize ---
    n_inputs = X_train.shape[1]  # e.g., 10 raycasts + 1 speed = 11
    model = DrivingPolicy(n_inputs)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Training loop ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)  # (batch, 2, 3)
            loss = criterion(outputs.view(-1, 3), batch_y.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)  # (batch, 2, 3)
                val_loss += criterion(outputs.view(-1, 3), batch_y.view(-1)).item()
                all_preds.append(outputs.cpu())
                all_targets.append(batch_y.cpu())

        y_pred_val = torch.cat(all_preds)  # (N, 2, 3)
        y_true_val = torch.cat(all_targets).numpy()  # (N, 2)

        y_pred_classes = torch.argmax(y_pred_val, dim=2).numpy()  # (N, 2)

        accuracy_throttle = np.mean(y_true_val[:, 0] == y_pred_classes[:, 0])
        accuracy_steer = np.mean(y_true_val[:, 1] == y_pred_classes[:, 1])

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss / len(train_loader):.4f} | "
                f"Val Loss: {val_loss / len(test_loader):.4f}"
            )
            print(
                f"Accuracy Throttle: {accuracy_throttle:.4f}, Accuracy Steer: {accuracy_steer:.4f}"
            )

    # --- Save model ---
    torch.save(model.state_dict(), "model/driving_policy.pth")
    print("✅ Model saved to 'model/driving_policy.pth'")
    return X_test_torch, device, model


@app.cell
def _():
    return


@app.cell
def _(X_test_torch, confusion_matrix, device, model, np, plt, torch, y_test):
    # Get predictions
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test_torch.to(device)).cpu()  # (N, 2, 3)

    y_pred = torch.argmax(y_pred_logits, dim=2).numpy()  # (N, 2)
    y_true = y_test.to_numpy()
    y_true_classes = (y_true + 1).astype(int)  # -1 -> 0, 0 -> 1, 1 -> 2

    # Plot confusion matrices
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- Throttle ----------
    cm_throttle = confusion_matrix(y_true_classes[:, 0], y_pred[:, 0], labels=[0, 1, 2]).T
    _im0 = _axes[0].imshow(cm_throttle, cmap="Blues", aspect="auto")

    _axes[0].set_xticks([0, 1, 2])
    _axes[0].set_xticklabels(["-1", "0", "1"])
    _axes[0].set_yticks([0, 1, 2])
    _axes[0].set_yticklabels(["-1", "0", "1"])
    _axes[0].set_xlabel("True Throttle")
    _axes[0].set_ylabel("Predicted Throttle")
    _axes[0].set_title("Throttle Confusion Matrix")
    plt.colorbar(_im0, ax=_axes[0], shrink=0.8)

    # Add text annotations
    for _i in range(3):  # predicted labels (rows)
        for _j in range(3):  # true labels (columns)
            _value = cm_throttle[_i, _j]
            _axes[0].text(
                _j, _i,
                str(_value),
                ha="center",
                va="center",
                color="black" if _value < np.max(cm_throttle) / 2 else "white"
            )

    # ---------- Steer ----------
    cm_steer = confusion_matrix(y_true_classes[:, 1], y_pred[:, 1], labels=[0, 1, 2]).T
    _im1 = _axes[1].imshow(cm_steer, cmap="Blues", aspect="auto")

    _axes[1].set_xticks([0, 1, 2])
    _axes[1].set_xticklabels(["-1", "0", "1"])
    _axes[1].set_yticks([0, 1, 2])
    _axes[1].set_yticklabels(["-1", "0", "1"])
    _axes[1].set_xlabel("True Steer")
    _axes[1].set_ylabel("Predicted Steer")
    _axes[1].set_title("Steer Confusion Matrix")
    plt.colorbar(_im1, ax=_axes[1], shrink=0.8)

    # Add text annotations
    for _i in range(3):  # predicted labels (rows)
        for _j in range(3):  # true labels (columns)
            _value = cm_steer[_i, _j]
            _axes[1].text(
                _j, _i,
                str(_value),
                ha="center",
                va="center",
                color="black" if _value < np.max(cm_steer) / 2 else "white"
            )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
