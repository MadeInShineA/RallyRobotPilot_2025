import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import os
    import lzma
    import pickle
    from PIL import Image
    import numpy as np
    import marimo as mo
    import io
    import copy
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import KFold
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        accuracy_score,
        classification_report,
    )
    import joblib
    import mlflow
    import mlflow.pytorch
    return Image, lzma, mo, np, os, pickle, pl, plt, sns, train_test_split


@app.cell
def _(mo):
    mo.md(r"""### Load the different visual records""")
    return


@app.cell
def _(lzma, os, pickle, pl):
    record_dir = "./visual_records/"

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
                "image": s.image,
                **{f"raycast_{i}": float(d) for i, d in enumerate(s.raycast_distances)},
            }
            all_records.append(record)

    # Create a single Polars DataFrame from all records
    try:
        df = pl.DataFrame(all_records).sort("record")
        print(f"✅ Successfully created combined Polars DataFrame with {len(df)} rows")
    except Exception as e:
        print(f"❌ Error creating combined DataFrame: {e}")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.tail()
    return


@app.cell
def _(mo):
    mo.md(r"""### Clean the first frames of each record when nothing happens (all inputs are 0)""")
    return


@app.cell
def _(df, pl):
    df_first_frames_cleaned = df.filter(
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
    mo.md(r"""### Also clean the end frames of each record when nothing happens""")
    return


@app.cell
def _(df_first_frames_cleaned, pl):
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
def _(mo):
    mo.md(r"""### Filter out the short records""")
    return


@app.cell
def _(df_last_frames_cleaned, pl):
    records_lengths = df_last_frames_cleaned.group_by("record").agg(pl.len())
    records_lengths
    return


@app.cell
def _(df_last_frames_cleaned, pl):
    min_length = 100

    df_cleaned = df_last_frames_cleaned.filter(
        pl.len().over("record") >= min_length
    )
    return (df_cleaned,)


@app.cell
def _(mo):
    mo.md(r"""### Convert the images to PIL""")
    return


@app.cell
def _(Image, np):
    def load_image_as_PIL(array_image: list[list[list[int]]])-> Image:
        arr = np.array(array_image, dtype=np.uint8)
        return Image.fromarray(arr)
    return (load_image_as_PIL,)


@app.cell
def _(df_cleaned, load_image_as_PIL, pl):
    df_cleaned_pil = df_cleaned.with_columns(
        pl.col('image').map_elements(load_image_as_PIL, return_dtype=pl.Object).alias('pil_image')
    )
    return (df_cleaned_pil,)


@app.cell
def _(df_cleaned_pil):
    df_cleaned_pil.head()
    return


@app.cell
def _(mo):
    mo.md(r"""### Preprocess the PIL images""")
    return


@app.cell
def _(Image):
    def preprocess_image(image: Image, size: tuple[int] = (90, 160)) -> Image:
        image = image.convert('L')  # Convert to grayscale
        image = image.resize(size, Image.Resampling.LANCZOS)  # Resize to exact dimensions
        return image
    return (preprocess_image,)


@app.cell
def _(df_cleaned_pil, pl, preprocess_image):
    images_dimensions = (90, 160)

    df_cleaned_pil_preprocessed = df_cleaned_pil.with_columns(
        pl.col('pil_image').map_elements(
            lambda img: preprocess_image(img, size=images_dimensions), 
            return_dtype=pl.Object
        ).alias('pil_image_preprocessed')
    )
    return (df_cleaned_pil_preprocessed,)


@app.cell
def _(df_cleaned_pil_preprocessed):
    df_cleaned_pil_preprocessed.head()
    return


@app.cell
def _(df_cleaned_pil_preprocessed):
    df_cleaned_pil_preprocessed["pil_image"][0]
    return


@app.cell
def _(df_cleaned_pil_preprocessed):
    df_cleaned_pil_preprocessed["pil_image_preprocessed"][0]
    return


@app.cell
def _(mo):
    mo.md(r"""### Data exploration""")
    return


@app.cell
def _(pl, sns):
    def plot_controls(df: pl.DataFrame)-> None:
        control_names = ["forward", "back", "left", "right"]
        subset = df.select(["frame_idx", "record"] + control_names)
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
    return (plot_controls,)


@app.cell
def _(df_cleaned_pil_preprocessed, plot_controls):
    plot_controls(df_cleaned_pil_preprocessed)
    return


@app.cell
def _(np, pl, plt):
    def plot_usage(df: pl.DataFrame)-> None:

        # Compute usage per file for each control
        usage_df = (
            df.group_by("record")
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
    return (plot_usage,)


@app.cell
def _(df_cleaned_pil_preprocessed, plot_usage):
    plot_usage(df_cleaned_pil_preprocessed)
    return


@app.cell
def _(np, pl, plt):
    def plot_car_speed_proximity(df: pl.DataFrame)-> None:
        # 1. Compute min_ray
        ray_cols = [
            col for col in df.columns if col.startswith("raycast_")
        ]
        min_rays_df = df.with_columns(min_ray=pl.min_horizontal(ray_cols))

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
    return (plot_car_speed_proximity,)


@app.cell
def _(df_cleaned_pil_preprocessed, plot_car_speed_proximity):
    plot_car_speed_proximity(df_cleaned_pil_preprocessed)
    return


@app.cell
def _(mo):
    mo.md(r"""### Data augmentation""")
    return


@app.cell
def _(Image, df_cleaned_pil_preprocessed, pl):
    def augment_image(img: Image) -> list[tuple[Image, dict]]:
        augmentations = []

        # Horizontal flip (swap left/right controls)
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        augmentations.append((flipped, {"left": "right", "right": "left"}))

        return augmentations

    # Apply augmentations
    augmented_rows = []
    for row in df_cleaned_pil_preprocessed.iter_rows(named=True):
        original_img = row["pil_image_preprocessed"]
        augmented_rows.append(row)  # Keep original

        for aug_img, control_swaps in augment_image(original_img):
            new_row = row.copy()
            new_row["pil_image_preprocessed"] = aug_img
            for old, new in control_swaps.items():
                new_row[new] = row[old]
                new_row[old] = row[new]
            augmented_rows.append(new_row)

    df_augmented = pl.DataFrame(augmented_rows)
    return (df_augmented,)


@app.cell
def _(df_augmented):
    df_augmented.head()
    return


@app.cell
def _(df_augmented, plot_controls):
    plot_controls(df_augmented)
    return


@app.cell
def _(df_augmented, plot_usage):
    plot_usage(df_augmented)
    return


@app.cell
def _(mo):
    mo.md(r"""Add a""")
    return


@app.cell
def _(df_augmented, np, pl):
    df_augmented_array_image = df_augmented.with_columns(
        pl.col("pil_image_preprocessed").map_elements(
            lambda pil_img: np.array(pil_img) if pil_img is not None else None,
            return_dtype=pl.Object
        ).alias("2d_array_image_preprocessed")
    )
    return (df_augmented_array_image,)


@app.cell
def _(mo):
    mo.md(r"""## Data separation (test, train)""")
    return


@app.cell
def _(df_augmented_array_image, train_test_split):
    df_train, df_test = train_test_split(
        df_augmented_array_image, test_size=0.2, random_state=42
    )

    # --- 2. Prepare feature and label columns
    feature_cols = "2d_array_image_preprocessed"

    # --- 3. Create X (inputs) as Polars DataFrames
    X_train = df_train.select(feature_cols)
    X_test = df_test.select(feature_cols)

    y_train = df_train.select("forward", "back", "left", "right")

    y_test = df_test.select("forward", "back", "left", "right")
    return X_train, y_train


@app.cell
def _(X_train):
    X_train.head()
    return


@app.cell
def _(X_train):
    X_train["2d_array_image_preprocessed"][0].shape
    return


@app.cell
def _(y_train):
    y_train.head()
    return


@app.cell
def _():
    architectures = [
        # Architecture 1: Simple CNN
        (
            "simple_cnn",
            [
                ("conv", {"out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.25}),
                ("conv", {"out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.25}),
                ("linear", {"out_features": 16}),
                ("dropout", {"p": 0.25}),
                ("linear", {"out_features": 8}),
            ],
            "ReLU"
        ),
    ]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
