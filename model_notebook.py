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
        f1_score,
        accuracy_score,
        classification_report
    )
    import joblib
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import KFold

    return (
        DataLoader,
        KFold,
        MinMaxScaler,
        TensorDataset,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
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
    return X_train_scaled, y_train


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
    mo.md(r"""## Building prediction neural network""")
    return


@app.cell
def _(
    DataLoader,
    KFold,
    TensorDataset,
    X_train_scaled,
    accuracy_score,
    classification_report,
    f1_score,
    nn,
    np,
    optim,
    torch,
    y_train,
):
    # --- Hyperparameters ---
    n_splits = 5
    epochs = 200
    batch_size = 256
    learning_rate = 1e-2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Conversion ---
    X_np = X_train_scaled.to_numpy()  # convert polars dataframe to numpy
    y_np = y_train.to_numpy()
    y_np = (y_np + 1).astype(np.int64)  # map [-1,0,1] to [0,1,2]

    # --- Model Definition ---
    class DrivingPolicy(nn.Module):
        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 6),  # 2 outputs × 3 classes
            )

        def forward(self, x):
            return self.net(x).view(-1, 2, 3)

    # --- Metric Storage ---
    metrics_per_fold = []

    all_y_true = []
    all_y_pred = []

    # --- K-Fold Loop ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        print(f"\n🔁 Fold {fold+1}/{n_splits}")
        print("-" * 40)

        # Data setup
        X_tr = torch.tensor(X_np[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_np[val_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_np[train_idx], dtype=torch.long)
        y_val = torch.tensor(y_np[val_idx], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        # Model, optimizer, loss
        model = DrivingPolicy(X_tr.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Tracking
        train_losses = []
        train_throttle_accs = []
        train_steer_accs = []
        val_throttle_accs = []
        val_steer_accs = []

        # --- Training Loop ---
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1, 3), batch_y.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))

            # --- Train accuracy ---
            model.eval()
            train_preds = []
            train_targets = []
            with torch.no_grad():
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    preds = torch.argmax(outputs.cpu(), dim=2)
                    train_preds.append(preds)
                    train_targets.append(batch_y)

            y_pred_train = torch.cat(train_preds).numpy()
            y_true_train = torch.cat(train_targets).numpy()

            train_throttle_acc = accuracy_score(y_true_train[:, 0], y_pred_train[:, 0])
            train_steer_acc = accuracy_score(y_true_train[:, 1], y_pred_train[:, 1])

            train_throttle_accs.append(train_throttle_acc)
            train_steer_accs.append(train_steer_acc)

            # --- Validation accuracy ---
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    preds = torch.argmax(outputs.cpu(), dim=2)
                    val_preds.append(preds)
                    val_targets.append(batch_y)

            y_pred_val = torch.cat(val_preds).numpy()
            y_true_val = torch.cat(val_targets).numpy()

            val_throttle_acc = accuracy_score(y_true_val[:, 0], y_pred_val[:, 0])
            val_steer_acc = accuracy_score(y_true_val[:, 1], y_pred_val[:, 1])

            val_throttle_accs.append(val_throttle_acc)
            val_steer_accs.append(val_steer_acc)

            # Append for global confusion matrix after last epoch only (or could do after fold)
            if epoch == epochs:
                all_y_true.append(y_true_val)
                all_y_pred.append(y_pred_val)

            print(f"Epoch {epoch:03} | Loss: {epoch_loss:.4f} | "
                  f"Train Throttle Acc: {train_throttle_acc:.3f} | Train Steer Acc: {train_steer_acc:.3f} | "
                  f"Val Throttle Acc: {val_throttle_acc:.3f} | Val Steer Acc: {val_steer_acc:.3f}")

        # --- Final Metrics for Fold ---
        throttle_f1 = f1_score(y_true_val[:, 0], y_pred_val[:, 0], average="macro")
        steer_f1 = f1_score(y_true_val[:, 1], y_pred_val[:, 1], average="macro")

        print("\nThrottle Classification Report:")
        print(classification_report(y_true_val[:, 0], y_pred_val[:, 0], target_names=["-1", "0", "1"]))
        print("Steer Classification Report:")
        print(classification_report(y_true_val[:, 1], y_pred_val[:, 1], target_names=["-1", "0", "1"]))

        metrics_per_fold.append({
            "train_loss": train_losses,
            "train_throttle_acc": train_throttle_accs,
            "train_steer_acc": train_steer_accs,
            "val_throttle_acc": val_throttle_accs,
            "val_steer_acc": val_steer_accs,
            "final_throttle_acc": val_throttle_accs[-1],
            "final_steer_acc": val_steer_accs[-1],
            "throttle_f1": throttle_f1,
            "steer_f1": steer_f1
        })
    return all_y_pred, all_y_true, epochs, metrics_per_fold


@app.cell
def _(epochs, metrics_per_fold, np, plt):
    # --- 📊 Per-Fold Combined Plots ---
    for _fold, metrics in enumerate(metrics_per_fold):
        epochs_range = np.arange(1, epochs + 1)

        # Compute errors (1 - accuracy)
        train_throttle_error = 1 - np.array(metrics["train_throttle_acc"])
        val_throttle_error = 1 - np.array(metrics["val_throttle_acc"])
        train_steer_error = 1 - np.array(metrics["train_steer_acc"])
        val_steer_error = 1 - np.array(metrics["val_steer_acc"])

        plt.figure(figsize=(18, 5))
        plt.suptitle(f"Fold {_fold + 1} Training Metrics")

        # --- Plot 1: Train Loss ---
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, metrics["train_loss"], label="Train Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss")
        plt.grid(True)

        # --- Plot 2: Throttle Errors ---
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, train_throttle_error, label="Throttle Train Error", color="green", linestyle='-')
        plt.plot(epochs_range, val_throttle_error, label="Throttle Val Error", color="green", linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Error (1 - Accuracy)")
        plt.title("Throttle Train vs Validation Error")
        plt.legend()
        plt.grid(True)

        # --- Plot 3: Steer Errors ---
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, train_steer_error, label="Steer Train Error", color="orange", linestyle='-')
        plt.plot(epochs_range, val_steer_error, label="Steer Val Error", color="orange", linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Error (1 - Accuracy)")
        plt.title("Steer Train vs Validation Error")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

    return


@app.cell
def _(metrics_per_fold, plt):
    # --- Summary Plots Across Folds ---
    throttle_accuracies = [m["final_throttle_acc"] for m in metrics_per_fold]
    steer_accuracies = [m["final_steer_acc"] for m in metrics_per_fold]
    throttle_f1s = [m["throttle_f1"] for m in metrics_per_fold]
    steer_f1s = [m["steer_f1"] for m in metrics_per_fold]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.boxplot([throttle_accuracies, steer_accuracies], tick_labels=["Throttle", "Steer"])
    plt.title("Final Accuracy per Fold")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.boxplot([throttle_f1s, steer_f1s], tick_labels=["Throttle", "Steer"])
    plt.title("Final F1 Score per Fold")
    plt.ylabel("F1 Score")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(all_y_pred, all_y_true, confusion_matrix, np, plt):
    # --- Confusion Matrices (Global) ---
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- Throttle ----------
    cm_throttle = confusion_matrix(y_true[:, 0], y_pred[:, 0], labels=[0, 1, 2]).T
    _im0 = _axes[0].imshow(cm_throttle, cmap="Blues", aspect="auto")
    _axes[0].set_xticks([0, 1, 2])
    _axes[0].set_xticklabels(["-1", "0", "1"])
    _axes[0].set_yticks([0, 1, 2])
    _axes[0].set_yticklabels(["-1", "0", "1"])
    _axes[0].set_xlabel("True Throttle")
    _axes[0].set_ylabel("Predicted Throttle")
    _axes[0].set_title("Throttle Confusion Matrix")
    plt.colorbar(_im0, ax=_axes[0], shrink=0.8)
    for _i in range(3):
        for _j in range(3):
            val = cm_throttle[_i, _j]
            _axes[0].text(_j, _i, str(val),
                          ha="center", va="center",
                          color="black" if val < np.max(cm_throttle)/2 else "white")

    # ---------- Steer ----------
    cm_steer = confusion_matrix(y_true[:, 1], y_pred[:, 1], labels=[0, 1, 2]).T
    _im1 = _axes[1].imshow(cm_steer, cmap="Blues", aspect="auto")
    _axes[1].set_xticks([0, 1, 2])
    _axes[1].set_xticklabels(["-1", "0", "1"])
    _axes[1].set_yticks([0, 1, 2])
    _axes[1].set_yticklabels(["-1", "0", "1"])
    _axes[1].set_xlabel("True Steer")
    _axes[1].set_ylabel("Predicted Steer")
    _axes[1].set_title("Steer Confusion Matrix")
    plt.colorbar(_im1, ax=_axes[1], shrink=0.8)
    for _i in range(3):
        for _j in range(3):
            val = cm_steer[_i, _j]
            _axes[1].text(_j, _i, str(val),
                          ha="center", va="center",
                          color="black" if val < np.max(cm_steer)/2 else "white")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
