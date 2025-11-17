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
    import json
    return Image, json, mo, np, os, pl, plt, sns, train_test_split


@app.cell
def _(mo):
    mo.md(r"""### Load the different visual records""")
    return


@app.cell
def _(Image, json, os, pl):

    record_dir = "./visual_records/"

    all_records = []  # Accumulate all records here

    for record_subdir in os.listdir(record_dir):
        record_subdir_path = os.path.join(record_dir, record_subdir)
        if not os.path.isdir(record_subdir_path):
            continue
    
        # Load the complete record JSON from the records subdirectory
        records_dir = os.path.join(record_subdir_path, "records")
        json_path = os.path.join(records_dir, "complete_record.json")
    
        if not os.path.exists(json_path):
            print(f"❌ Missing complete_record.json in {record_subdir}/records/")
            continue
    
        try:
            with open(json_path, "r") as f:
                snapshots = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load {json_path}: {e}")
            continue

        # Get list of image files in the images subdirectory (at the same level as records)
        images_dir = os.path.join(record_subdir_path, "images")
        if not os.path.exists(images_dir):
            print(f"❌ Missing images directory in {record_subdir}")
            continue
    
        for s in snapshots:
            frame_idx = s["idx"]
        
            # Construct the expected image filename based on frame index
            image_filename = f"frame_{frame_idx}.png"
            image_path = os.path.join(images_dir, image_filename)
        
            # Verify the image file exists
            if not os.path.exists(image_path):
                print(f"❌ Missing image file: {image_path}")
                continue
        
            # Load the actual PIL Image
            try:
                pil_image = Image.open(image_path)
            except Exception as e:
                print(f"❌ Failed to load image {image_path}: {e}")
                continue
        
            record = {
                "record": record_subdir,
                "frame_idx": frame_idx,
                "forward": s["input"][0],
                "back": s["input"][1],
                "left": s["input"][2],
                "right": s["input"][3],
                "car_speed": s["speed"],
                "image": pil_image,  # Store actual PIL Image object
                "time": s["time"],
                "angle": s["angle"],
                "position": s["position"],
                "checkpoint": s["checkpoint"],
                "lap": s["lap"],
            }
            all_records.append(record)

    # Create a single Polars DataFrame from all records
    try:
        df = pl.DataFrame(all_records).sort(["record", "frame_idx"])
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
def _(df_cleaned, pl, preprocess_image):
    images_dimensions = (90, 160)

    df_cleaned_pil_preprocessed = df_cleaned.with_columns(
        pl.col('image').map_elements(
            lambda img: preprocess_image(img, size=images_dimensions), 
            return_dtype=pl.Object
        ).alias('image_preprocessed')
    )
    return (df_cleaned_pil_preprocessed,)


@app.cell
def _(df_cleaned_pil_preprocessed):
    df_cleaned_pil_preprocessed.head()
    return


@app.cell
def _(df_cleaned_pil_preprocessed):
    df_cleaned_pil_preprocessed["image"][0]
    return


@app.cell
def _(df_cleaned_pil_preprocessed):
    df_cleaned_pil_preprocessed["image_preprocessed"][0]
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
        original_img = row["image_preprocessed"]
        augmented_rows.append(row)  # Keep original

        for aug_img, control_swaps in augment_image(original_img):
            new_row = row.copy()
            new_row["image_preprocessed"] = aug_img
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
        pl.col("image_preprocessed").map_elements(
            lambda pil_img: np.array(pil_img) if pil_img is not None else None,
            return_dtype=pl.Object
        ).alias("2d_array_image_preprocessed")
    )
    return (df_augmented_array_image,)


@app.cell
def _(df_augmented_array_image):
    df_augmented_array_image
    return


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
