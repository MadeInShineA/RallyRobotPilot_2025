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
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        confusion_matrix,
        f1_score,
        accuracy_score,
        classification_report,
    )
    import joblib
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import KFold
    import seaborn as sns

    return (
        DataLoader,
        KFold,
        RobustScaler,
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
        sns,
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
            with lzma.open(input_path, "rb") as _f:
                snapshots = pickle.load(_f)
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
    mo.md(
        r"""### Clean the first frames of each record when nothing happens (all inputs are 0)"""
    )
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
def _(df_first_frames_cleaned, pl):
    ### Also clean the end frames of each record when nothing happens
    df_last_frames_cleaned = df_first_frames_cleaned.filter(
        (
            (pl.col("forward") != 0)
            | (pl.col("back") != 0)
            | (pl.col("left") != 0)
            | (pl.col("right") != 0)
        )
        .reverse()
        .cum_max()
        .reverse()
        .over("record")
    )

    df_last_frames_cleaned.tail()
    return (df_last_frames_cleaned,)


@app.cell
def _(df_first_frames_cleaned, pl):
    df_with_no_input = df_first_frames_cleaned.with_columns(
        no_input=(
            (pl.col("forward") == 0)
            & (pl.col("back") == 0)
            & (pl.col("left") == 0)
            & (pl.col("right") == 0)
        ).cast(pl.Int8)  # 1 = no input, 0 = some input
    )

    no_input_per_record = (
        df_with_no_input.group_by("record")
        .agg(
            pl.col("no_input").mean().alias("no_input_fraction"),
            pl.col("no_input").sum().alias("no_input_count"),
            pl.len().alias("total_frames"),
        )
        .sort("no_input_fraction", descending=True)
    )

    no_input_per_record
    return (no_input_per_record,)


@app.cell
def _(no_input_per_record, plt):
    # Extract data
    records = no_input_per_record["record"].to_list()
    fractions = no_input_per_record["no_input_fraction"].to_numpy()

    # Plot
    _fig, _ax = plt.subplots(figsize=(10, max(6, 0.3 * len(records))))
    _bars = _ax.barh(records, fractions, color="lightcoral")

    # Labels
    _ax.set_xlabel("Fraction of Time with No Input")
    _ax.set_ylabel("Replay (record)")
    _ax.set_title("No-Input Usage per Replay")
    _ax.set_xlim(0, 1)

    # Optional: add percentage labels
    for _i, _frac in enumerate(fractions):
        _ax.text(_frac + 0.01, _i, f"{_frac:.1%}", va="center", fontsize=9)

    _ax.invert_yaxis()  # Longest on top
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_last_frames_cleaned, pl):
    df_with_input_only = df_last_frames_cleaned.filter(
        (pl.col("forward") != 0)
        | (pl.col("back") != 0)
        | (pl.col("left") != 0)
        | (pl.col("right") != 0)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Filter out the short records""")
    return


@app.cell
def _(df_last_frames_cleaned, pl):
    records_lengths = df_last_frames_cleaned.group_by("record").agg(pl.len())
    records_lengths
    return (records_lengths,)


@app.cell
def _(records_lengths):
    records_lengths.select("len").sum()
    return


@app.cell
def _(df_last_frames_cleaned, pl):
    min_length = 100

    df_lengths_filtered = df_last_frames_cleaned.filter(
        pl.len().over("record") >= min_length
    )
    return (df_lengths_filtered,)


@app.cell
def _(df_lengths_filtered):
    cleaned_records = df_lengths_filtered["record"].unique()

    cleaned_records
    return


@app.cell
def _(mo):
    mo.md(r"""## Data exploration""")
    return


@app.cell
def _(df_lengths_filtered, pl, sns):
    def ctrl_stripplots():
        control_names = ["forward", "back", "left", "right"]
        subset = df_lengths_filtered.select(["frame_idx", "record"] + control_names)
        long_df = subset.unpivot(
            on=control_names,
            index=("frame_idx", "record"),
            variable_name="control",
            value_name="active",
        ).filter(pl.col("active") == 1)

        g = sns.FacetGrid(long_df, row="record", aspect=3)
        return g.map_dataframe(
            sns.stripplot,
            x="frame_idx",
            y="control",
            hue="control",
            palette=["green", "red", "blue", "orange"],
        )

    ctrl_stripplots()
    return


@app.cell
def _(df_lengths_filtered, np, pl, plt):
    # Compute usage per file for each control
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

    # Prepare data for heatmap — 4 columns
    control_cols = ["forward_usage", "back_usage", "left_usage", "right_usage"]
    usage_matrix = usage_df.select(control_cols).to_numpy()
    records_sorted = usage_df["record"].to_list()

    # Plot heatmap
    _fig, _ax = plt.subplots(figsize=(10, max(4, 0.5 * len(records_sorted))))
    _im = _ax.imshow(usage_matrix, cmap="Blues", aspect="auto")

    # Set ticks
    _ax.set_yticks(np.arange(len(records_sorted)))
    _ax.set_yticklabels(records_sorted)
    _ax.set_xticks(np.arange(len(control_cols)))
    _ax.set_xticklabels(["Forward", "Back", "Left", "Right"], rotation=45, ha="right")

    # Add colorbar
    plt.colorbar(_im, ax=_ax, label="Fraction of time active")

    # Add text annotations
    for _i in range(len(records_sorted)):
        for _j in range(len(control_cols)):
            _val = usage_matrix[_i, _j]
            text = _ax.text(
                _j,
                _i,
                f"{_val:.2f}",
                ha="center",
                va="center",
                color="black" if _val < 0.5 else "white",
            )

    _ax.set_title("Control Usage per Record")
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

    # Create the plot
    plt.figure(figsize=(10, 6))
    _im = plt.imshow(ray_matrix, aspect="auto", cmap="viridis")

    # Add colorbar
    cbar = plt.colorbar(_im)
    cbar.set_label("Raycast Distance", fontsize=12)

    # Labels & title
    plt.xlabel("Ray Index", fontsize=12)
    plt.ylabel("Forward-Onset Event", fontsize=12)
    plt.title("Raycast Profile at Start of Acceleration", fontsize=13, pad=15)

    # Optional: improve layout
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Data separation (test, train) and preparation (scalling)""")
    return


@app.cell
def _(
    RobustScaler,
    df_lengths_filtered,
    joblib,
    os,
    pl,
    ray_cols,
    train_test_split,
):
    # --- 1. Split by RECORD first
    df_train, df_test = train_test_split(
        df_lengths_filtered, test_size=0.2, random_state=42
    )

    # --- 2. Prepare feature and label columns
    feature_cols = ray_cols + ["car_speed"]

    # --- 3. Create X (inputs) as Polars DataFrames
    X_train = df_train.select(feature_cols)
    X_test = df_test.select(feature_cols)

    # --- 4. Scale inputs and convert back to Polars DataFrames
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)  # returns np.ndarray
    X_test_scaled = scaler_X.transform(X_test)

    # Convert back to Polars with original column names
    X_train_scaled = pl.DataFrame(X_train_scaled, schema=feature_cols)
    X_test_scaled = pl.DataFrame(X_test_scaled, schema=feature_cols)

    # --- 5. Encode labels as Polars DataFrames (throttle, steer)
    y_train = df_train.select("forward", "back", "left", "right")

    y_test = df_test.select("forward", "back", "left", "right")

    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler_X, "model/scaler_X.joblib")
    return X_test_scaled, X_train_scaled, y_test, y_train


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
    mo.md(r"""## Define architectures to test""")
    return


@app.cell
def _():
    architectures = [
        ("3_layer_32_32_32_ReLU", [32, 32, 32], "ReLU"),
    ]
    return (architectures,)


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
    architectures,
    classification_report,
    f1_score,
    nn,
    np,
    optim,
    torch,
    y_train,
):
    # --- Hyperparameters ---
    n_splits = 2
    epochs = 200
    batch_size = 100
    learning_rate = 1e-3
    dropout_rate = 0.25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor([5.0, 1.0, 2.0, 2.0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=weights)

    # --- Data Preparation ---
    X_train_np = X_train_scaled.to_numpy()  # Convert Polars dataframe to numpy
    y_train_np = y_train.to_numpy().astype(np.float32)  # Labels are already 0/1

    # --- Model Definition ---
    class DrivingPolicy(nn.Module):
        def __init__(
            self, input_dim, hidden_layers, dropout_rate=0.3, activation="ReLU"
        ):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if activation == "ReLU":
                    layers.append(nn.ReLU())
                elif activation == "Tanh":
                    layers.append(nn.Tanh())
                elif activation == "Sigmoid":
                    layers.append(nn.Sigmoid())
                # Add dropout after activation
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 4))  # 4 outputs × 2 classes each
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            # Output shape: (batch_size, 4)
            return self.net(x)

    # --- Training and Evaluation Functions ---
    def train_one_epoch(model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(model, dataloader):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
                all_preds.append(preds)
                all_targets.append(batch_y.cpu().numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        return y_true, y_pred

    def compute_accuracy(y_true, y_pred):
        forward_acc = accuracy_score(y_true[:, 0], y_pred[:, 0])
        back_acc = accuracy_score(y_true[:, 1], y_pred[:, 1])
        left_acc = accuracy_score(y_true[:, 2], y_pred[:, 2])
        right_acc = accuracy_score(y_true[:, 3], y_pred[:, 3])
        return forward_acc, back_acc, left_acc, right_acc

    def print_classification_reports(y_true, y_pred):
        print("\nForward Classification Report:")
        print(
            classification_report(y_true[:, 0], y_pred[:, 0], target_names=["0", "1"])
        )
        print("Back Classification Report:")
        print(
            classification_report(y_true[:, 1], y_pred[:, 1], target_names=["0", "1"])
        )
        print("Left Classification Report:")
        print(
            classification_report(y_true[:, 2], y_pred[:, 2], target_names=["0", "1"])
        )
        print("Right Classification Report:")
        print(
            classification_report(y_true[:, 3], y_pred[:, 3], target_names=["0", "1"])
        )

    # --- K-Fold Cross Validation ---
    metrics_per_arch = {arch_name: [] for arch_name, _, _ in architectures}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np), 1):
        print(f"\n🔁 Fold {fold}/{n_splits}")
        print("-" * 40)

        # Prepare data loaders for this fold
        X_tr = torch.tensor(X_train_np[train_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_train_np[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_train_np[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_train_np[val_idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        for arch_name, hidden_layers, activation in architectures:
            print(f"Training {arch_name}...")
            # Initialize model, optimizer, loss function
            model = DrivingPolicy(
                X_tr.shape[1], hidden_layers, dropout_rate, activation
            ).to(device)
            _optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Track metrics per epoch
            train_losses = []
            train_forward_accs = []
            train_back_accs = []
            train_left_accs = []
            train_right_accs = []
            val_forward_accs = []
            val_back_accs = []
            val_left_accs = []
            val_right_accs = []

            for epoch in range(1, epochs + 1):
                loss = train_one_epoch(model, train_loader, criterion, _optimizer)
                train_losses.append(loss)

                y_true_train, y_pred_train = evaluate(model, train_loader)
                train_forward_acc, train_back_acc, train_left_acc, train_right_acc = (
                    compute_accuracy(y_true_train, y_pred_train)
                )
                train_forward_accs.append(train_forward_acc)
                train_back_accs.append(train_back_acc)
                train_left_accs.append(train_left_acc)
                train_right_accs.append(train_right_acc)
                train_forward_f1 = f1_score(
                    y_true_train[:, 0], y_pred_train[:, 0], average="binary"
                )
                train_back_f1 = f1_score(
                    y_true_train[:, 1], y_pred_train[:, 1], average="binary"
                )
                train_left_f1 = f1_score(
                    y_true_train[:, 2], y_pred_train[:, 2], average="binary"
                )
                train_right_f1 = f1_score(
                    y_true_train[:, 3], y_pred_train[:, 3], average="binary"
                )

                y_true_val, y_pred_val = evaluate(model, val_loader)
                val_forward_acc, val_back_acc, val_left_acc, val_right_acc = (
                    compute_accuracy(y_true_val, y_pred_val)
                )
                val_forward_accs.append(val_forward_acc)
                val_back_accs.append(val_back_acc)
                val_left_accs.append(val_left_acc)
                val_right_accs.append(val_right_acc)
                val_forward_f1 = f1_score(
                    y_true_val[:, 0], y_pred_val[:, 0], average="binary"
                )
                val_back_f1 = f1_score(
                    y_true_val[:, 1], y_pred_val[:, 1], average="binary"
                )
                val_left_f1 = f1_score(
                    y_true_val[:, 2], y_pred_val[:, 2], average="binary"
                )
                val_right_f1 = f1_score(
                    y_true_val[:, 3], y_pred_val[:, 3], average="binary"
                )

                print(
                    f"Epoch {epoch:03} | Loss: {loss:.4f} | "
                    f"Train Forward F1: {train_forward_f1:.3f} | Train Back F1: {train_back_f1:.3f} | Train Left F1: {train_left_f1:.3f} | Train Right F1: {train_right_f1:.3f} | "
                    f"Val Forward F1: {val_forward_f1:.3f} | Val Back F1: {val_back_f1:.3f} | Val Left F1: {val_left_f1:.3f} | Val Right F1: {val_right_f1:.3f}"
                )

                if epoch == epochs:
                    # Final metrics and report
                    forward_f1 = f1_score(
                        y_true_val[:, 0], y_pred_val[:, 0], average="binary"
                    )
                    back_f1 = f1_score(
                        y_true_val[:, 1], y_pred_val[:, 1], average="binary"
                    )
                    left_f1 = f1_score(
                        y_true_val[:, 2], y_pred_val[:, 2], average="binary"
                    )
                    right_f1 = f1_score(
                        y_true_val[:, 3], y_pred_val[:, 3], average="binary"
                    )
                    print_classification_reports(y_true_val, y_pred_val)

                    metrics_per_arch[arch_name].append(
                        {
                            "train_loss": train_losses,
                            "train_forward_acc": train_forward_accs,
                            "train_back_acc": train_back_accs,
                            "train_left_acc": train_left_accs,
                            "train_right_acc": train_right_accs,
                            "val_forward_acc": val_forward_accs,
                            "val_back_acc": val_back_accs,
                            "val_left_acc": val_left_accs,
                            "val_right_acc": val_right_accs,
                            "final_forward_acc": val_forward_accs[-1],
                            "final_back_acc": val_back_accs[-1],
                            "final_left_acc": val_left_accs[-1],
                            "final_right_acc": val_right_accs[-1],
                            "forward_f1": forward_f1,
                            "back_f1": back_f1,
                            "left_f1": left_f1,
                            "right_f1": right_f1,
                        }
                    )
    return (
        DrivingPolicy,
        X_train_np,
        batch_size,
        device,
        dropout_rate,
        epochs,
        learning_rate,
        weights,
        metrics_per_arch,
        train_one_epoch,
        y_train_np,
    )


@app.cell
def _(architectures, metrics_per_arch, plt):
    # --- Summary Plots Across Architectures ---
    arch_names = [arch_name for arch_name, _, _ in architectures]
    forward_accuracies = [
        [m["final_forward_acc"] for m in metrics_per_arch[arch_name]]
        for arch_name in arch_names
    ]
    back_accuracies = [
        [m["final_back_acc"] for m in metrics_per_arch[arch_name]]
        for arch_name in arch_names
    ]
    left_accuracies = [
        [m["final_left_acc"] for m in metrics_per_arch[arch_name]]
        for arch_name in arch_names
    ]
    right_accuracies = [
        [m["final_right_acc"] for m in metrics_per_arch[arch_name]]
        for arch_name in arch_names
    ]
    forward_f1s = [
        [m["forward_f1"] for m in metrics_per_arch[arch_name]]
        for arch_name in arch_names
    ]
    back_f1s = [
        [m["back_f1"] for m in metrics_per_arch[arch_name]] for arch_name in arch_names
    ]
    left_f1s = [
        [m["left_f1"] for m in metrics_per_arch[arch_name]] for arch_name in arch_names
    ]
    right_f1s = [
        [m["right_f1"] for m in metrics_per_arch[arch_name]] for arch_name in arch_names
    ]

    # Combine data
    acc_data = forward_accuracies + back_accuracies + left_accuracies + right_accuracies
    f1_data = forward_f1s + back_f1s + left_f1s + right_f1s

    # Create labels
    labels = (
        [f"{name} (Forward)" for name in arch_names]
        + [f"{name} (Back)" for name in arch_names]
        + [f"{name} (Left)" for name in arch_names]
        + [f"{name} (Right)" for name in arch_names]
    )

    # Colors
    colors = (
        ["green"] * len(arch_names)
        + ["red"] * len(arch_names)
        + ["blue"] * len(arch_names)
        + ["orange"] * len(arch_names)
    )

    plt.figure(figsize=(16, 5))

    # --- Accuracy subplot ---
    plt.subplot(1, 2, 1)
    bp1 = plt.boxplot(acc_data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
    plt.title("Final Accuracy per Architecture (Throttle + Steer)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")

    # --- F1 subplot ---
    plt.subplot(1, 2, 2)
    bp2 = plt.boxplot(f1_data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
    plt.title("Final F1 Score per Architecture (Throttle + Steer)")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45, ha="right")

    # Add a shared legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Forward"),
        Patch(facecolor="red", label="Back"),
        Patch(facecolor="blue", label="Left"),
        Patch(facecolor="orange", label="Right"),
    ]
    plt.figlegend(
        handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.96)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # make room for legend
    plt.show()
    return


@app.cell
def _(X_train_scaled, architectures, dropout_rate, metrics_per_arch, np):
    # Select the best architecture based on average validation F1 score
    arch_scores = {}
    for _arch_name, metrics_list in metrics_per_arch.items():
        avg_forward_f1 = np.mean([m["forward_f1"] for m in metrics_list])
        avg_back_f1 = np.mean([m["back_f1"] for m in metrics_list])
        avg_left_f1 = np.mean([m["left_f1"] for m in metrics_list])
        avg_right_f1 = np.mean([m["right_f1"] for m in metrics_list])
        avg_f1 = (avg_forward_f1 + avg_back_f1 + avg_left_f1 + avg_right_f1) / 4
        arch_scores[_arch_name] = avg_f1

    best_arch_name = max(arch_scores, key=lambda k: arch_scores[k])
    best_hidden_layers, best_activation = next(
        (hidden_layers, activation)
        for name, hidden_layers, activation in architectures
        if name == best_arch_name
    )

    print(
        f"Best architecture: {best_arch_name} with average validation weighted F1: {arch_scores[best_arch_name]:.4f}"
    )

    # Save model config
    import json

    config = {
        "input_dim": X_train_scaled.shape[1],
        "hidden_layers": best_hidden_layers,
        "dropout_rate": dropout_rate,
        "activation": best_activation,
    }
    with open("./model/model_config.json", "w") as _f:
        json.dump(config, _f)
    print("Model config saved to ./model/model_config.json")
    return best_activation, best_arch_name, best_hidden_layers


@app.cell
def _(best_arch_name, epochs, metrics_per_arch, np, plt):
    # --- 📊 Per-Fold Plots for Best Architecture ---
    best_metrics_list = metrics_per_arch[best_arch_name]
    for _fold, metrics in enumerate(best_metrics_list):
        epochs_range = np.arange(1, epochs + 1)

        # Compute errors (1 - accuracy)
        train_forward_error = 1 - np.array(metrics["train_forward_acc"])
        val_forward_error = 1 - np.array(metrics["val_forward_acc"])
        train_back_error = 1 - np.array(metrics["train_back_acc"])
        val_back_error = 1 - np.array(metrics["val_back_acc"])
        train_left_error = 1 - np.array(metrics["train_left_acc"])
        val_left_error = 1 - np.array(metrics["val_left_acc"])
        train_right_error = 1 - np.array(metrics["train_right_acc"])
        val_right_error = 1 - np.array(metrics["val_right_acc"])

        plt.figure(figsize=(24, 10))
        plt.suptitle(f"Best Arch {best_arch_name} - Fold {_fold + 1} Training Metrics")

        # --- Plot 1: Train Loss ---
        plt.subplot(2, 4, 1)
        plt.plot(epochs_range, metrics["train_loss"], label="Train Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss")
        plt.grid(True)

        # --- Plot 2: Forward Errors ---
        plt.subplot(2, 4, 2)
        plt.plot(
            epochs_range,
            train_forward_error,
            label="Forward Train Error",
            color="green",
            linestyle="-",
        )
        plt.plot(
            epochs_range,
            val_forward_error,
            label="Forward Val Error",
            color="green",
            linestyle="--",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Error (1 - Accuracy)")
        plt.title("Forward Train vs Validation Error")
        plt.legend()
        plt.grid(True)

        # --- Plot 3: Back Errors ---
        plt.subplot(2, 4, 3)
        plt.plot(
            epochs_range,
            train_back_error,
            label="Back Train Error",
            color="red",
            linestyle="-",
        )
        plt.plot(
            epochs_range,
            val_back_error,
            label="Back Val Error",
            color="red",
            linestyle="--",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Error (1 - Accuracy)")
        plt.title("Back Train vs Validation Error")
        plt.legend()
        plt.grid(True)

        # --- Plot 4: Left Errors ---
        plt.subplot(2, 4, 4)
        plt.plot(
            epochs_range,
            train_left_error,
            label="Left Train Error",
            color="blue",
            linestyle="-",
        )
        plt.plot(
            epochs_range,
            val_left_error,
            label="Left Val Error",
            color="blue",
            linestyle="--",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Error (1 - Accuracy)")
        plt.title("Left Train vs Validation Error")
        plt.legend()
        plt.grid(True)

        # --- Plot 5: Right Errors ---
        plt.subplot(2, 4, 5)
        plt.plot(
            epochs_range,
            train_right_error,
            label="Right Train Error",
            color="orange",
            linestyle="-",
        )
        plt.plot(
            epochs_range,
            val_right_error,
            label="Right Val Error",
            color="orange",
            linestyle="--",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Error (1 - Accuracy)")
        plt.title("Right Train vs Validation Error")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()
    return


@app.cell
def _(
    DataLoader,
    DrivingPolicy,
    TensorDataset,
    X_train_np,
    batch_size,
    best_activation,
    best_hidden_layers,
    device,
    dropout_rate,
    epochs,
    weights,
    learning_rate,
    nn,
    optim,
    torch,
    train_one_epoch,
    y_train_np,
):
    # --- Final Model Training on Full Dataset ---
    print("Training final model on full dataset...")
    X_full = torch.tensor(X_train_np, dtype=torch.float32)
    y_full = torch.tensor(y_train_np, dtype=torch.float32)
    full_train_loader = DataLoader(
        TensorDataset(X_full, y_full), batch_size=batch_size, shuffle=True
    )

    final_model = DrivingPolicy(
        X_full.shape[1], best_hidden_layers, dropout_rate, best_activation
    ).to(device)
    final_criterion = nn.BCEWithLogitsLoss(weight=weights)
    final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

    for _epoch in range(1, epochs + 1):
        _loss = train_one_epoch(
            final_model, full_train_loader, final_criterion, final_optimizer
        )
        if _epoch % 50 == 0 or _epoch == 1 or _epoch == epochs:
            print(f"Epoch {_epoch:03} | Full Train Loss: {_loss:.4f}")

    # --- Save Final Model ---
    torch.save(final_model.state_dict(), "./model/driving_policy.pth")
    print(
        "✅ Final model trained on all data saved to './model/driving_policy_model.pth'"
    )
    return (final_model,)


@app.cell
def _(
    DataLoader,
    TensorDataset,
    X_test_scaled,
    accuracy_score,
    batch_size,
    classification_report,
    device,
    f1_score,
    final_model,
    np,
    torch,
    y_test,
):
    # Assume X_test_scaled and y_test are prepared similarly to training data
    X_test_np = X_test_scaled.to_numpy()
    y_test_np = y_test.to_numpy().astype(np.float32)

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).to(device)
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size
    )

    # Evaluate final model on test set
    final_model.eval()
    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = final_model(batch_X)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
            all_test_preds.append(preds)
            all_test_targets.append(batch_y.cpu().numpy())

    y_pred_test = np.concatenate(all_test_preds)
    y_true_test = np.concatenate(all_test_targets)

    # Compute test accuracy and F1 scores
    test_forward_acc = accuracy_score(y_true_test[:, 0], y_pred_test[:, 0])
    test_back_acc = accuracy_score(y_true_test[:, 1], y_pred_test[:, 1])
    test_left_acc = accuracy_score(y_true_test[:, 2], y_pred_test[:, 2])
    test_right_acc = accuracy_score(y_true_test[:, 3], y_pred_test[:, 3])

    test_forward_f1 = f1_score(y_true_test[:, 0], y_pred_test[:, 0], average="binary")
    test_back_f1 = f1_score(y_true_test[:, 1], y_pred_test[:, 1], average="binary")
    test_left_f1 = f1_score(y_true_test[:, 2], y_pred_test[:, 2], average="binary")
    test_right_f1 = f1_score(y_true_test[:, 3], y_pred_test[:, 3], average="binary")

    print("\nFinal Test Set Performance:")
    print(f"Forward Accuracy: {test_forward_acc:.4f}")
    print(f"Back Accuracy: {test_back_acc:.4f}")
    print(f"Left Accuracy: {test_left_acc:.4f}")
    print(f"Right Accuracy: {test_right_acc:.4f}")
    print(f"Forward F1 Score: {test_forward_f1:.4f}")
    print(f"Back F1 Score: {test_back_f1:.4f}")
    print(f"Left F1 Score: {test_left_f1:.4f}")
    print(f"Right F1 Score: {test_right_f1:.4f}")

    print("\nForward Classification Report:")
    print(
        classification_report(
            y_true_test[:, 0], y_pred_test[:, 0], target_names=["0", "1"]
        )
    )
    print("Back Classification Report:")
    print(
        classification_report(
            y_true_test[:, 1], y_pred_test[:, 1], target_names=["0", "1"]
        )
    )
    print("Left Classification Report:")
    print(
        classification_report(
            y_true_test[:, 2], y_pred_test[:, 2], target_names=["0", "1"]
        )
    )
    print("Right Classification Report:")
    print(
        classification_report(
            y_true_test[:, 3], y_pred_test[:, 3], target_names=["0", "1"]
        )
    )
    return y_pred_test, y_true_test


@app.cell
def _(confusion_matrix, np, plt, y_pred_test, y_true_test):
    # --- Confusion Matrices (Test Set) ---
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 10))

    controls = ["Forward", "Back", "Left", "Right"]
    for idx, control in enumerate(controls):
        cm = confusion_matrix(y_true_test[:, idx], y_pred_test[:, idx], labels=[0, 1]).T
        ax = _axes[idx // 2, idx % 2]
        _im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"])
        ax.set_xlabel(f"True {control}")
        ax.set_ylabel(f"Predicted {control}")
        ax.set_title(f"{control} Confusion Matrix")
        plt.colorbar(_im, ax=ax, shrink=0.8)
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="black" if val < np.max(cm) / 2 else "white",
                )

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
