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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import joblib
    import json
    from dotenv import load_dotenv

    import torch
    import torch.nn as nn
    import mlflow
    import mlflow.pytorch
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
    import tempfile
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.gridspec as gridspec

    return (
        DataLoader,
        Dataset,
        Image,
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        gridspec,
        json,
        load_dotenv,
        mlflow,
        mo,
        nn,
        np,
        os,
        pl,
        plt,
        roc_curve,
        sns,
        tempfile,
        torch,
        train_test_split,
    )


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

    df_cleaned = df_last_frames_cleaned.filter(pl.len().over("record") >= min_length)
    return (df_cleaned,)


@app.cell
def _(mo):
    mo.md(r"""### Preprocess the PIL images""")
    return


@app.cell
def _(Image):
    def preprocess_image(image: Image, size: tuple[int] = (160, 224)) -> Image:
        image = image.convert("L")  # Convert to grayscale
        image = image.resize(
            size, Image.Resampling.LANCZOS
        )  # Resize to exact dimensions
        return image
    return (preprocess_image,)


@app.cell
def _(df_cleaned, pl, preprocess_image):
    images_dimensions = (160, 224)
    images_width, images_height = images_dimensions

    df_cleaned_pil_preprocessed = df_cleaned.with_columns(
        pl.col("image")
        .map_elements(
            lambda img: preprocess_image(img, size=images_dimensions),
            return_dtype=pl.Object,
        )
        .alias("image_preprocessed")
    )
    return df_cleaned_pil_preprocessed, images_height, images_width


app._unparsable_cell(
    r"""
        df_cleaned_pil_preprocessed.head()
    """,
    name="_"
)


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
    def plot_controls(df: pl.DataFrame) -> None:
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
def _(pl, plt, sns):
    def plot_usage(df) -> plt.Axes:
        # Ensure we have a 'nothing' column: 1 when all controls are 0
        df = df.with_columns(
            nothing=(
                1
                - (
                    pl.col("forward").cast(pl.Boolean)
                    | pl.col("back").cast(pl.Boolean)
                    | pl.col("left").cast(pl.Boolean)
                    | pl.col("right").cast(pl.Boolean)
                ).cast(pl.Int8)
            )
        )

        # Compute average usage per record for all 5 controls
        usage_df = (
            df.group_by("record")
            .agg(
                pl.col("forward").mean().alias("forward_usage"),
                pl.col("back").mean().alias("back_usage"),
                pl.col("left").mean().alias("left_usage"),
                pl.col("right").mean().alias("right_usage"),
                pl.col("nothing").mean().alias("nothing_usage"),
            )
            .sort("record")
        )

        # Prepare data for heatmap
        control_cols = [
            "forward_usage",
            "back_usage",
            "left_usage",
            "right_usage",
            "nothing_usage",
        ]
        records_sorted = usage_df["record"].to_list()

        # Create a DataFrame for seaborn
        heatmap_df = usage_df.select(control_cols + ["record"]).to_pandas()
        heatmap_df = heatmap_df.set_index("record")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(records_sorted))))

        # Plot heatmap using seaborn
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar_kws={"label": "Fraction of time active"},
            ax=ax,
        )

        ax.set_title("Control Usage per Record")
        ax.set_xticklabels(
            ["Forward", "Back", "Left", "Right", "Nothing"], rotation=45, ha="right"
        )

        plt.tight_layout()
        return ax
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
    for _row in df_cleaned_pil_preprocessed.iter_rows(named=True):
        original_img = _row["image_preprocessed"]  # Get the image from the row
        # Add is_augmented=False for original row
        original_row = _row.copy()
        original_row["is_augmented"] = False
        augmented_rows.append(original_row)  # Keep original

        for aug_img, control_swaps in augment_image(original_img):
            new_row = _row.copy()
            new_row["image_preprocessed"] = aug_img
            new_row["is_augmented"] = True  # Mark augmented images as True
            for old, new in control_swaps.items():
                new_row[new] = _row[old]
                new_row[old] = _row[new]
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
    mo.md(r"""### Convert images to 2d array""")
    return


@app.cell
def _(df_augmented, np, pl):
    df_augmented_array_image = df_augmented.with_columns(
        pl.col("image_preprocessed")
        .map_elements(
            lambda pil_img: np.array(pil_img) if pil_img is not None else None,
            return_dtype=pl.Object,
        )
        .alias("2d_array_image_preprocessed")
    )
    return (df_augmented_array_image,)


@app.cell
def _(df_augmented_array_image):
    df_augmented_array_image.head()
    return


@app.cell
def _(mo):
    mo.md(r"""## Data separation (test, train)""")
    return


@app.cell
def _(df_augmented_array_image, train_test_split):
    cnn_df_train, cnn_df_test = train_test_split(
        df_augmented_array_image, test_size=0.2, random_state=42
    )

    # --- 2. Prepare feature and label columns
    cnn_feature_cols = "2d_array_image_preprocessed"

    # --- 3. Create X (inputs) as Polars DataFrames
    cnn_X_train = cnn_df_train.select(cnn_feature_cols)
    cnn_X_test = cnn_df_test.select(cnn_feature_cols)

    cnn_y_train = cnn_df_train.select("forward", "back", "left", "right")

    cnn_y_test = cnn_df_test.select("forward", "back", "left", "right")
    return cnn_X_test, cnn_X_train, cnn_y_test, cnn_y_train


@app.cell
def _(cnn_X_train):
    cnn_X_train.head()
    return


@app.cell
def _(cnn_X_train):
    cnn_X_train["2d_array_image_preprocessed"][0].shape
    return


@app.cell
def _(cnn_y_train):
    cnn_y_train.head()
    return


@app.cell(hide_code=True)
def _(
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    datetime,
    f1_score,
    gridspec,
    json,
    mlflow,
    nn,
    np,
    os,
    plt,
    roc_curve,
    sns,
    tempfile,
    torch,
):
    class FlexibleCNN(nn.Module):
        def __init__(
            self, arch_config, input_channels=1, input_height=40, input_width=66
        ):
            super(FlexibleCNN, self).__init__()
            name, layers = arch_config
            self.arch_name = name
            self.layers_config = layers
            self.input_channels = input_channels
            # Store input dimensions as model attributes
            self.input_height = input_height
            self.input_width = input_width

            # Identify where features end (before first linear layer)
            feature_configs = []
            classifier_configs = []
            # Find the first linear layer to separate features from classifier
            first_linear_idx = len(layers)
            for i, (layer_type, _) in enumerate(layers):
                if layer_type == "linear":
                    first_linear_idx = i
                    break
            feature_configs = layers[:first_linear_idx]
            classifier_configs = layers[first_linear_idx:]
            # Calculate the flattened size after features
            self.features = self._build_sequential_with_channels(feature_configs)
            self.feature_output_size = self._get_feature_output_size(
                input_height, input_width
            )
            if classifier_configs:
                # Update all linear layers' in_features appropriately
                updated_classifier_configs = []
                prev_out_features = (
                    self.feature_output_size
                )  # For the first linear layer
                for layer_type, params in classifier_configs:
                    if layer_type == "linear":
                        updated_params = params.copy()
                        updated_params["in_features"] = prev_out_features
                        updated_classifier_configs.append((layer_type, updated_params))
                        # Update prev_out_features for the next linear layer
                        prev_out_features = params["out_features"]
                    else:
                        updated_classifier_configs.append((layer_type, params))
                self.classifier = self._build_sequential(updated_classifier_configs)
            else:
                self.classifier = nn.Identity()

        def _get_feature_output_size(self, input_height, input_width):
            """Calculate the flattened output size of features"""
            with torch.no_grad():
                dummy_input = torch.zeros(
                    1, self.input_channels, input_height, input_width
                )
                output = self.features(dummy_input)
                flattened_size = output.view(1, -1).size(1)
            return flattened_size

        def _build_sequential_with_channels(self, layer_configs):
            modules = []
            in_channels = self.input_channels
            for layer_type, params in layer_configs:
                if layer_type == "conv":
                    conv_params = params.copy()
                    conv_params["in_channels"] = in_channels
                    modules.append(nn.Conv2d(**conv_params))
                    if "out_channels" in params:
                        in_channels = params["out_channels"]
                    if "out_channels" in params:
                        modules.append(nn.BatchNorm2d(params["out_channels"]))
                elif layer_type == "maxpool":
                    modules.append(nn.MaxPool2d(**params))
                elif layer_type == "dropout":
                    modules.append(nn.Dropout(**params))
                elif layer_type == "relu":
                    modules.append(nn.ReLU(**params))
                elif layer_type == "sigmoid":
                    modules.append(nn.Sigmoid())
                elif layer_type == "tanh":
                    modules.append(nn.Tanh())
                elif layer_type == "softmax":
                    modules.append(nn.Softmax(**params))
            return nn.Sequential(*modules)

        def _build_sequential(self, layer_configs):
            modules = []
            for layer_type, params in layer_configs:
                if layer_type == "linear":
                    if "in_features" not in params:
                        raise ValueError(
                            f"Linear layer missing required 'in_features': {params}"
                        )
                    modules.append(nn.Linear(**params))
                elif layer_type == "dropout":
                    modules.append(nn.Dropout(**params))
                elif layer_type == "relu":
                    modules.append(nn.ReLU(**params))
                elif layer_type == "sigmoid":
                    modules.append(nn.Sigmoid())
                elif layer_type == "tanh":
                    modules.append(nn.Tanh())
                elif layer_type == "softmax":
                    modules.append(nn.Softmax(**params))
                # Add other activation functions as needed
            return nn.Sequential(*modules)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    def add_nothing_class(y):
        """Calculates the 'nothing' class based on existing labels/predictions."""
        # Check if any of the first 4 classes are active (1) along the class axis (axis=1)
        # If none are active, 'nothing' is 1, otherwise 0.
        # Assume y is a numpy array of shape (N, num_classes) where num_classes is 4 or 5
        if y.shape[1] >= 4:
            nothing_mask = np.all(y[:, :4] == 0, axis=1)
            nothing_col = nothing_mask.astype(int).reshape(-1, 1)
            return np.concatenate([y, nothing_col], axis=1)
        else:
            # If y doesn't have at least 4 original classes, just add a zero column
            zero_col = np.zeros((y.shape[0], 1), dtype=y.dtype)
            return np.concatenate([y, zero_col], axis=1)


    def create_training_plots(
        train_losses,
        val_losses,
        train_f1s,
        val_f1s,
        train_f1_per_class,
        val_f1_per_class,
        epoch,
        class_names,
    ):
        """Create training plots with Model Loss, Weighted F1-Score, and Per-Class F1-Scores"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Training and validation loss
        axes[0].plot(range(1, epoch + 2), train_losses, "b-", label="Training Loss")
        axes[0].plot(range(1, epoch + 2), val_losses, "r-", label="Validation Loss")
        axes[0].set_title("Model Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Weighted F1 score
        axes[1].plot(range(1, epoch + 2), train_f1s, "b-", label="Training F1-Score")
        axes[1].plot(range(1, epoch + 2), val_f1s, "r-", label="Validation F1-Score")
        axes[1].set_title("Weighted F1-Score")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1-Score")
        axes[1].legend()
        axes[1].grid(True)

        # Per-class F1 scores
        epochs_range = range(1, epoch + 2)
        # Use a color map or list that can handle the number of classes
        # Adjust color list if necessary, or use a colormap
        num_classes = len(class_names)
        colors = plt.cm.get_cmap('tab10', num_classes)(range(num_classes))
        # If you have a specific color list, ensure it's long enough
        # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"] # Example for 5 classes

        for i, class_name in enumerate(class_names):
            axes[2].plot(
                epochs_range,
                train_f1_per_class[i],
                color=colors[i],
                linestyle="-",
                alpha=0.7,
                label=f"Train {class_name}",
            )
            axes[2].plot(
                epochs_range,
                val_f1_per_class[i],
                color=colors[i],
                linestyle="--",
                alpha=0.9,
                label=f"Val {class_name}",
            )
        axes[2].set_title("Per-Class F1-Scores")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1-Score")
        axes[2].set_ylim(0, 1)
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[2].grid(True)

        plt.tight_layout()
        return fig


    def create_f1_per_class_plot(
        train_f1_per_class, val_f1_per_class, epoch, class_names
    ):
        """Create plots showing F1 scores per class - Updated for N classes"""
        num_classes = len(class_names)
        # Calculate subplot grid dimensions
        n_cols = 2
        n_rows = (num_classes + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if num_classes == 1 else axes
        else:
            axes = axes.ravel()

        for i, class_name in enumerate(class_names):
            epochs_range = range(1, epoch + 2)
            axes[i].plot(
                epochs_range, train_f1_per_class[i], "b-", label=f"Training F1"
            )
            axes[i].plot(
                epochs_range, val_f1_per_class[i], "r-", label=f"Validation F1"
            )
            axes[i].set_title(f"F1-Score - {class_name}")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel("F1-Score")
            axes[i].set_ylim(0, 1)
            axes[i].legend()
            axes[i].grid(True)
            # Add final values as text
            if len(val_f1_per_class[i]) > 0:
                final_val_f1 = val_f1_per_class[i][-1]
                axes[i].text(
                    0.02,
                    0.98,
                    f"Final Val F1: {final_val_f1:.3f}",
                    transform=axes[i].transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig


    def plot_confusion_matrices(y_true, y_pred, class_names):
        """Create confusion matrices for each class (multi-label) - Updated for N classes"""
        num_classes = len(class_names)
        # Calculate subplot grid dimensions
        n_cols = 2
        n_rows = (num_classes + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols / 2, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if num_classes == 1 else axes
        else:
            axes = axes.ravel()

        for i, class_name in enumerate(class_names):
            # Binary confusion matrix for each class
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=axes[i],
            )
            axes[i].set_title(f"Confusion Matrix - {class_name}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig


    def plot_roc_curves(y_true, y_scores, class_names):
        """Create ROC curves for each class"""
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves - Multi-label Classification")
        ax.legend()
        ax.grid(True)
        return fig


    def plot_prediction_examples(model, test_loader, class_names, num_examples=4):
        """Visualize prediction examples with bar plots under images - Updated for N classes"""
        model.eval()
        images, true_labels, pred_labels, pred_scores = [], [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                scores = torch.sigmoid(outputs)  # Convert logits to probabilities
                # Store examples
                images.extend(data.cpu().numpy()[:2])
                true_labels.extend(target.cpu().numpy()[:2])
                pred_labels.extend(predictions.cpu().numpy()[:2])
                pred_scores.extend(scores.cpu().numpy()[:2])
                if len(images) >= num_examples:
                    break

        # Ensure labels/preds/scores have 'nothing' column if needed
        # This function assumes the class_names list passed in already includes 'nothing'
        # and the model outputs are for the original 4 classes.
        # We add 'nothing' here based on the original 4 classes in the labels/preds/scores.
        true_labels_np = np.array(true_labels)
        pred_labels_np = np.array(pred_labels)
        pred_scores_np = np.array(pred_scores)

        # Only add 'nothing' if class_names indicates it should be there (i.e., length 5)
        if len(class_names) == 5:
            true_labels_np = add_nothing_class(true_labels_np)
            pred_labels_np = add_nothing_class(pred_labels_np)
            # For scores, the 'nothing' probability is 1 if all original are 0, else 0 (or derive from model somehow)
            # For simplicity here, we'll just add a column of zeros to scores, as the model doesn't predict 'nothing' directly.
            # If you want the model to predict 'nothing' probability, you'd need to modify the model.
            # For now, we calculate 'nothing' only for labels/preds for metrics.
            # Let's just pass the original scores to the bar plot for the original classes.
            # We need to adjust the plotting loop to handle the score array correctly.
            scores_for_plot = pred_scores_np # Use original scores for plotting
            if scores_for_plot.shape[1] < len(class_names): # If scores array is missing 'nothing'
                 # Add a column of zeros or calculate 'nothing' score (e.g., 1 - max of others, or 1 if all others < 0.5)
                 # For now, adding zeros is simplest if the model doesn't predict 'nothing'
                 # Let's assume the model scores are for 4 classes, and we add a 'nothing' score column.
                 # A simple 'nothing' score could be: prob_nothing = 1 if all original probs < 0.5 else 0
                 # Or prob_nothing = 1 - max(original_probs)
                 # Let's use prob_nothing = 1 - max(original_probs) for a continuous score.
                 max_probs = np.max(pred_scores_np, axis=1, keepdims=True) # Shape (N, 1)
                 prob_nothing = 1 - max_probs # Shape (N, 1)
                 scores_for_plot = np.concatenate([pred_scores_np, prob_nothing], axis=1) # Shape (N, 5)
        else:
            scores_for_plot = pred_scores_np # Use original scores if 'nothing' not expected


        # Create figure with custom grid for image + bar plot layout
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(
            num_examples, 1, height_ratios=[3] * num_examples, hspace=0.4
        )

        # Use a color map or list that can handle the number of classes
        num_plot_classes = len(class_names)
        colors = plt.cm.get_cmap('tab10', num_plot_classes)(range(num_plot_classes))
        # If you have a specific color list, ensure it's long enough
        # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"] # Example for 5 classes

        for i in range(min(num_examples, len(images))):
            img = images[i][0]  # Grayscale channel
            true_label = true_labels_np[i] # Use potentially updated labels
            pred_label = pred_labels_np[i] # Use potentially updated preds
            pred_score = scores_for_plot[i] # Use potentially updated scores

            # Create sub-grid for each example (image + bar plot)
            sub_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, gs[i], height_ratios=[2, 1], hspace=0.3
            )

            # Image subplot
            ax_img = fig.add_subplot(sub_gs[0])
            ax_img.imshow(img, cmap="gray")
            # Create title with true and predicted labels
            true_actions = [
                class_names[j] for j, val in enumerate(true_label) if val > 0
            ]
            pred_actions = [
                class_names[j] for j, val in enumerate(pred_label) if val > 0
            ]
            if not true_actions:
                true_actions = ["none"] # Or class_names[-1] if always including 'nothing'
            if not pred_actions:
                pred_actions = ["none"] # Or class_names[-1] if always including 'nothing'
            title = f"True: {', '.join(true_actions)} | Pred: {', '.join(pred_actions)}"
            ax_img.set_title(title, fontsize=10)
            ax_img.axis("off")

            # Bar plot subplot
            ax_bar = fig.add_subplot(sub_gs[1])
            bars = ax_bar.bar(
                class_names,
                pred_score,
                color=colors,
                alpha=0.7,
            )
            ax_bar.set_ylim(0, 1)
            ax_bar.set_ylabel("Prob", fontsize=8)
            ax_bar.tick_params(axis="x", labelsize=8)
            ax_bar.tick_params(axis="y", labelsize=8)
            # Add value labels on bars
            for j, (bar, score) in enumerate(zip(bars, pred_score)):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.02,
                    f"{score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            # Add grid for better readability
            ax_bar.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        return fig


    def plot_wrong_prediction_examples(model, test_loader, class_names, num_examples=4):
        """
        Visualize only false prediction examples (where any class is mispredicted).
        For multi-label classification, a prediction is considered false if
        predicted vector != true vector.
        """
        model.eval()
        images, true_labels, pred_labels, pred_scores = [], [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                # Convert to numpy for comparison
                pred_np = predictions.cpu().numpy()
                target_np = target.cpu().numpy()
                data_np = data.cpu().numpy()

                # Find indices where prediction != target (any class differs)
                mismatches = np.any(pred_np != target_np, axis=1)

                # Collect only the false examples
                for i in range(len(mismatches)):
                    if mismatches[i] and len(images) < num_examples:
                        images.append(data_np[i][0])  # Grayscale channel
                        true_labels.append(target_np[i])
                        pred_labels.append(pred_np[i])
                        pred_scores.append(probabilities[i].cpu().numpy())

                if len(images) >= num_examples:
                    break

        if len(images) == 0:
            print("No false predictions found in the provided data.")
            return None

        # Ensure labels/preds/scores have 'nothing' column if needed
        true_labels_np = np.array(true_labels)
        pred_labels_np = np.array(pred_labels)
        pred_scores_np = np.array(pred_scores)

        if len(class_names) == 5:
            true_labels_np = add_nothing_class(true_labels_np)
            pred_labels_np = add_nothing_class(pred_labels_np)
            # Similar logic for scores as in plot_prediction_examples if needed
            max_probs = np.max(pred_scores_np, axis=1, keepdims=True) # Shape (N, 1)
            prob_nothing = 1 - max_probs # Shape (N, 1)
            scores_for_plot = np.concatenate([pred_scores_np, prob_nothing], axis=1) # Shape (N, 5)
        else:
            scores_for_plot = pred_scores_np


        # Create figure
        fig = plt.figure(figsize=(12, 3 * len(images)))
        gs = gridspec.GridSpec(len(images), 1, hspace=0.4)

        # Use a color map or list that can handle the number of classes
        num_plot_classes = len(class_names)
        colors = plt.cm.get_cmap('tab10', num_plot_classes)(range(num_plot_classes))
        # If you have a specific color list, ensure it's long enough
        # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"] # Example for 5 classes

        for i in range(len(images)):
            img = images[i]
            true_label = true_labels_np[i] # Use potentially updated labels
            pred_label = pred_labels_np[i] # Use potentially updated preds
            pred_score = scores_for_plot[i] # Use potentially updated scores

            # Sub-grid: image + bar plot
            sub_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, gs[i], height_ratios=[2, 1], hspace=0.3
            )

            # Image subplot
            ax_img = fig.add_subplot(sub_gs[0])
            ax_img.imshow(img, cmap="gray")
            true_actions = [
                class_names[j] for j, val in enumerate(true_label) if val > 0
            ] or ["none"] # Or class_names[-1] if always including 'nothing'
            pred_actions = [
                class_names[j] for j, val in enumerate(pred_label) if val > 0
            ] or ["none"] # Or class_names[-1] if always including 'nothing'
            title = f"True: {', '.join(true_actions)} | Pred: {', '.join(pred_actions)}"
            ax_img.set_title(title, fontsize=10, color="red")  # Red to highlight error
            ax_img.axis("off")

            # Bar plot subplot
            ax_bar = fig.add_subplot(sub_gs[1])
            bars = ax_bar.bar(
                class_names,
                pred_score,
                color=colors,
                alpha=0.7,
            )
            ax_bar.set_ylim(0, 1)
            ax_bar.set_ylabel("Prob", fontsize=8)
            ax_bar.tick_params(axis="x", labelsize=8)
            ax_bar.tick_params(axis="y", labelsize=8)
            ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
            # Add score labels on bars
            for bar, score in zip(bars, pred_score):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.02,
                    f"{score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        plt.tight_layout()
        return fig


    def create_accuracy_per_class_plot(
        train_acc_per_class, val_acc_per_class, epoch, class_names
    ):
        """Create plots showing accuracy per class - Updated for N classes"""
        num_classes = len(class_names)
        # Calculate subplot grid dimensions
        n_cols = 2
        n_rows = (num_classes + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if num_classes == 1 else axes
        else:
            axes = axes.ravel()

        for i, class_name in enumerate(class_names):
            epochs_range = range(1, epoch + 2)
            axes[i].plot(
                epochs_range, train_acc_per_class[i], "b-", label=f"Training Accuracy"
            )
            axes[i].plot(
                epochs_range, val_acc_per_class[i], "r-", label=f"Validation Accuracy"
            )
            axes[i].set_title(f"Accuracy - {class_name}")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel("Accuracy (%)")
            axes[i].set_ylim(0, 100)
            axes[i].legend()
            axes[i].grid(True)
            # Add final values as text
            if len(val_acc_per_class[i]) > 0:
                final_val_acc = val_acc_per_class[i][-1]
                axes[i].text(
                    0.02,
                    0.98,
                    f"Final Val Acc: {final_val_acc:.1f}%",
                    transform=axes[i].transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig


    def train_model(
        model,
        train_loader,
        val_loader,
        epochs=10,
        learning_rate=0.001,
        experiment_name="cnn_experiment",
        run_name=None,
        weights=None,
    ):
        """
        Enhanced training with comprehensive MLflow logging and graphics including weighted F1-score
        """
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        if run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{model.arch_name}_lr{learning_rate}_ep{epochs}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            # Log comprehensive parameters
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("model_architecture", model.arch_name)
            mlflow.log_param("input_channels", model.input_channels)
            mlflow.log_param("input_height", model.input_height) # Log input height
            mlflow.log_param("input_width", model.input_width)  # Log input width
            mlflow.log_param("feature_output_size", model.feature_output_size)
            mlflow.log_param("batch_size", train_loader.batch_size)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("loss_function", "BCEWithLogitsLoss")

            # Model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            mlflow.log_param("total_parameters", total_params)
            mlflow.log_param("trainable_parameters", trainable_params)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            if weights is not None:
                if len(weights) != 4:  # Assuming 4 classes initially
                    raise ValueError(
                        f"Expected 4 weights for 4 classes, got {len(weights)}"
                    )
                # Convert weights to tensor
                pos_weights = torch.tensor(weights, dtype=torch.float32)
                # Create the loss function with pos_weight
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
                print(
                    f"Using weighted BCEWithLogitsLoss with pos_weights: {pos_weights}"
                )
            else:
                # Default behavior if no weights provided
                criterion = nn.BCEWithLogitsLoss()
                print("Using standard BCEWithLogitsLoss (no weights).")

            # Track metrics for plotting - ADD F1 scores
            train_losses, val_losses = [], []
            train_accs, val_accs = [], []
            train_f1s, val_f1s = [], []  # New: track overall F1 scores
            num_original_classes = 4 # Assuming original classes are forward, back, left, right
            num_total_classes = 5 # Including the 'nothing' class
            train_f1_per_class = [
                [] for _ in range(num_total_classes)
            ]  # Track F1 per class for training
            val_f1_per_class = [
                [] for _ in range(num_total_classes)
            ]  # Track F1 per class for validation
            train_acc_per_class = [
                [] for _ in range(num_total_classes)
            ]  # Track accuracy per class for training
            val_acc_per_class = [
                [] for _ in range(num_total_classes)
            ]  # Track accuracy per class for validation
            class_names = ["forward", "back", "left", "right", "nothing"] # Update class names

            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                train_pred_list, train_target_list = [], []

                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    # Use sigmoid probabilities with 0.5 threshold (like original)
                    probabilities = torch.sigmoid(output)
                    predicted = (probabilities > 0.5).float()
                    train_total += target.numel()
                    train_correct += (predicted == target).sum().item()

                    # Store predictions and targets for F1 calculation (original 4 classes)
                    train_pred_list.append(predicted.cpu().numpy())
                    train_target_list.append(target.cpu().numpy())

                # Calculate training metrics
                train_pred_all = np.vstack(train_pred_list)
                train_target_all = np.vstack(train_target_list)

                # Add the 'nothing' class column to both targets and predictions based on original 4 classes
                train_target_all_with_nothing = add_nothing_class(train_target_all)
                train_pred_all_with_nothing = add_nothing_class(train_pred_all)

                # Overall weighted F1 (using 5 classes now)
                try:
                    train_f1_weighted = f1_score(
                        train_target_all_with_nothing,
                        train_pred_all_with_nothing,
                        average="weighted",
                        zero_division=0,
                    )
                except Exception as e:
                    print(f"Warning: Could not calculate training weighted F1: {e}")
                    train_f1_weighted = 0.0
                train_f1s.append(train_f1_weighted)

                # Per-class F1 scores (for 5 classes)
                try:
                    train_f1_class = f1_score(
                        train_target_all_with_nothing, train_pred_all_with_nothing, average=None, zero_division=0
                    )
                    for i in range(num_total_classes): # Loop for 5 classes now
                        train_f1_per_class[i].append(train_f1_class[i])
                except Exception as e:
                    print(f"Warning: Could not calculate training per-class F1: {e}")
                    # If there are no positive samples for a class, set F1 to 0
                    for i in range(num_total_classes): # Loop for 5 classes now
                        train_f1_per_class[i].append(0.0)

                # Per-class accuracy (for 5 classes)
                for i in range(num_total_classes): # Loop for 5 classes now
                    class_pred = train_pred_all_with_nothing[:, i]
                    class_target = train_target_all_with_nothing[:, i]
                    class_acc = accuracy_score(class_target, class_pred) * 100 # Use sklearn accuracy
                    train_acc_per_class[i].append(class_acc)

                train_acc = 100.0 * train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                train_accs.append(train_acc)

                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_predictions, val_targets, val_scores = [], [], []

                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()

                        # Use sigmoid probabilities with 0.5 threshold (like original)
                        probabilities = torch.sigmoid(output)
                        predicted = (probabilities > 0.5).float()
                        scores = probabilities  # Use probabilities for ROC curves
                        val_total += target.numel()
                        val_correct += (predicted == target).sum().item()

                        val_predictions.extend(predicted.cpu().numpy())
                        val_targets.extend(target.cpu().numpy())
                        val_scores.extend(scores.cpu().numpy())

                # Calculate validation metrics
                val_pred_all = np.array(val_predictions)
                val_target_all = np.array(val_targets)

                # Add the 'nothing' class column to both targets and predictions based on original 4 classes
                val_target_all_with_nothing = add_nothing_class(val_target_all)
                val_pred_all_with_nothing = add_nothing_class(val_pred_all)

                # Overall weighted F1 (using 5 classes now)
                try:
                    val_f1_weighted = f1_score(
                        val_target_all_with_nothing,
                        val_pred_all_with_nothing,
                        average="weighted",
                        zero_division=0,
                    )
                except Exception as e:
                    print(f"Warning: Could not calculate validation weighted F1: {e}")
                    val_f1_weighted = 0.0
                val_f1s.append(val_f1_weighted)

                # Per-class F1 scores (for 5 classes)
                try:
                    val_f1_class = f1_score(
                        val_target_all_with_nothing, val_pred_all_with_nothing, average=None, zero_division=0
                    )
                    for i in range(num_total_classes): # Loop for 5 classes now
                        val_f1_per_class[i].append(val_f1_class[i])
                except Exception as e:
                    print(f"Warning: Could not calculate validation per-class F1: {e}")
                    # If there are no positive samples for a class, set F1 to 0
                    for i in range(num_total_classes): # Loop for 5 classes now
                        val_f1_per_class[i].append(0.0)

                # Per-class accuracy (for 5 classes)
                for i in range(num_total_classes): # Loop for 5 classes now
                    class_pred = val_pred_all_with_nothing[:, i]
                    class_target = val_target_all_with_nothing[:, i]
                    class_acc = accuracy_score(class_target, class_pred) * 100 # Use sklearn accuracy
                    val_acc_per_class[i].append(class_acc)

                val_acc = 100.0 * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                val_accs.append(val_acc)

                # Log metrics including F1 scores
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("train_f1_weighted", train_f1_weighted, step=epoch)
                # Log per-class training F1 scores (updated loop)
                for i, class_name in enumerate(class_names): # Now iterates 5 times
                    mlflow.log_metric(
                        f"train_f1_{class_name}", train_f1_per_class[i][-1], step=epoch
                    )
                    mlflow.log_metric(
                        f"train_acc_{class_name}",
                        train_acc_per_class[i][-1],
                        step=epoch,
                    )

                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                mlflow.log_metric("val_f1_weighted", val_f1_weighted, step=epoch)
                # Log per-class validation F1 scores (updated loop)
                for i, class_name in enumerate(class_names): # Now iterates 5 times
                    mlflow.log_metric(
                        f"val_f1_{class_name}", val_f1_per_class[i][-1], step=epoch
                    )
                    mlflow.log_metric(
                        f"val_acc_{class_name}", val_acc_per_class[i][-1], step=epoch
                    )

                # Print with per-class F1 scores and per-class accuracy (updated loops)
                class_f1_str = ", ".join(
                    [
                        f"{class_names[i]}: {val_f1_per_class[i][-1]:.3f}" # Loop 5 times
                        for i in range(num_total_classes)
                    ]
                )
                class_acc_str = ", ".join(
                    [
                        f"{class_names[i]}: {val_acc_per_class[i][-1]:.1f}%" # Loop 5 times
                        for i in range(num_total_classes)
                    ]
                )
                print(
                    f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1_weighted:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1_weighted:.4f}"
                )
                print(f"  Val F1 per class: {class_f1_str}")
                print(f"  Val Acc per class: {class_acc_str}")

            # Final comprehensive evaluation
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_scores = np.array(val_scores) # Keep original scores for ROC

            # Add 'nothing' class for final metric calculation
            val_targets_with_nothing = add_nothing_class(val_targets)
            val_predictions_with_nothing = add_nothing_class(val_predictions)

            # Calculate final metrics (using updated variables)
            try:
                final_f1_weighted = f1_score(
                    val_targets_with_nothing, val_predictions_with_nothing, average="weighted", zero_division=0
                )
                final_f1_per_class = f1_score(
                    val_targets_with_nothing, val_predictions_with_nothing, average=None, zero_division=0
                )
            except Exception as e:
                print(f"Warning: Could not calculate final weighted/per-class F1: {e}")
                final_f1_weighted = 0.0
                final_f1_per_class = [0.0] * num_total_classes # Update length to 5

            # Calculate final per-class accuracy (using updated variables)
            final_acc_per_class = []
            for i in range(num_total_classes): # Loop for 5 classes
                class_pred = val_predictions_with_nothing[:, i]
                class_target = val_targets_with_nothing[:, i]
                class_acc = accuracy_score(class_target, class_pred) * 100 # Use sklearn accuracy
                final_acc_per_class.append(class_acc)

            mlflow.log_metric(
                "final_val_f1_weighted", final_f1_weighted
            )  # Log final F1
            # Log per-class final F1 scores (updated loop)
            for i, class_name in enumerate(class_names): # Now iterates 5 times
                mlflow.log_metric(f"final_val_f1_{class_name}", final_f1_per_class[i])
                mlflow.log_metric(f"final_val_acc_{class_name}", final_acc_per_class[i])

            # Calculate and log classification report (using updated variables)
            report = classification_report(
                val_targets_with_nothing,
                val_predictions_with_nothing,
                target_names=class_names, # Use updated class names
                output_dict=True,
                zero_division=0,
            )
            # Log metrics for each class (including macro F1) (updated loop)
            for i, class_name in enumerate(class_names): # Now iterates 5 times
                mlflow.log_metric(
                    f"val_precision_{class_name}", report[class_name]["precision"]
                )
                mlflow.log_metric(
                    f"val_recall_{class_name}", report[class_name]["recall"]
                )
                mlflow.log_metric(
                    f"val_f1_{class_name}", report[class_name]["f1-score"]
                )

            # Also log macro and weighted averages
            mlflow.log_metric("val_f1_macro", report["macro avg"]["f1-score"])
            mlflow.log_metric("val_f1_weighted", report["weighted avg"]["f1-score"])

            # Create and log classification report in temporary file - CORRECTED VERSION (using updated variables)
            report_text = classification_report(
                val_targets_with_nothing, val_predictions_with_nothing, target_names=class_names, zero_division=0 # Use updated variables
            )
            # Create temp file, write content, and close the file handle before logging
            tmp_report_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            )
            tmp_report_file.write(report_text)
            tmp_report_file.close()  # Close the file handle
            mlflow.log_artifact(
                tmp_report_file.name, "reports/classification_report.txt"
            )
            os.unlink(tmp_report_file.name)  # Delete the temporary file after logging

            # Create and log model summary in temporary file - CORRECTED VERSION
            final_f1_per_class_str = ", ".join(
                [f"{class_names[i]}: {final_f1_per_class[i]:.4f}" for i in range(num_total_classes)] # Loop 5 times
            )
            final_acc_per_class_str = ", ".join(
                [f"{class_names[i]}: {final_acc_per_class[i]:.2f}%" for i in range(num_total_classes)] # Loop 5 times
            )
            model_summary = f"""
            Model Architecture: {model.arch_name}
            Input Channels: {model.input_channels}
            Input Height: {model.input_height}
            Input Width: {model.input_width}
            Feature Output Size: {model.feature_output_size}
            Total Parameters: {total_params:,}
            Trainable Parameters: {trainable_params:,}
            Layers: {len(model.layers_config)}
            Training Configuration:
            - Epochs: {epochs}
            - Learning Rate: {learning_rate}
            - Batch Size: {train_loader.batch_size}
            - Optimizer: Adam
            - Loss Function: BCEWithLogitsLoss
            - Final Validation Accuracy: {val_acc:.2f}%
            - Final Validation Weighted F1-Score: {final_f1_weighted:.4f}
            - Final Validation Per-Class F1-Scores: {final_f1_per_class_str}
            - Final Validation Per-Class Accuracy: {final_acc_per_class_str}
            Class-wise Performance:
            {report_text}
            """
            # Create temp file, write content, and close the file handle before logging
            tmp_summary_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            )
            tmp_summary_file.write(model_summary)
            tmp_summary_file.close()  # Close the file handle
            mlflow.log_artifact(tmp_summary_file.name, "reports/model_summary.txt")
            os.unlink(tmp_summary_file.name)  # Delete the temporary file after logging

            # Create and log confusion matrices
            cm_fig = plot_confusion_matrices(val_targets_with_nothing, val_predictions_with_nothing, class_names) # Use updated variables
            tmp_cm_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cm_fig.savefig(tmp_cm_file.name, dpi=150, bbox_inches="tight")
            tmp_cm_file.close()  # Close the file handle
            mlflow.log_artifact(tmp_cm_file.name, "confusion_matrices")
            os.unlink(tmp_cm_file.name)
            plt.close(cm_fig)

            # Create and log ROC curves (only for original 4 classes as 'nothing' is derived)
            roc_fig = plot_roc_curves(val_targets, val_scores, class_names[:4]) # Use original 4 classes for ROC
            tmp_roc_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            roc_fig.savefig(tmp_roc_file.name, dpi=150, bbox_inches="tight")
            tmp_roc_file.close()  # Close the file handle
            mlflow.log_artifact(tmp_roc_file.name, "roc_curves")
            os.unlink(tmp_roc_file.name)
            plt.close(roc_fig)

            # Create and log prediction examples (including 'nothing' in plots if applicable)
            pred_examples_fig = plot_wrong_prediction_examples(
                model, val_loader, class_names # Pass updated class names
            )
            if pred_examples_fig is not None: # Check if any wrong predictions were found
                tmp_pred_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                pred_examples_fig.savefig(tmp_pred_file.name, dpi=150, bbox_inches="tight")
                tmp_pred_file.close()  # Close the file handle
                mlflow.log_artifact(tmp_pred_file.name, "wrong_prediction_examples")
                os.unlink(tmp_pred_file.name)
                plt.close(pred_examples_fig)

            # Create final training plot (with global F1 only) - ONLY FINAL PLOT IN training_plots
            final_main_fig = create_training_plots(
                train_losses,
                val_losses,
                train_f1s,
                val_f1s,
                train_f1_per_class,
                val_f1_per_class,
                epochs - 1,
                class_names, # Pass updated class names
            )
            tmp_train_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            final_main_fig.savefig(tmp_train_file.name, dpi=150, bbox_inches="tight")
            tmp_train_file.close()  # Close the file handle
            mlflow.log_artifact(
                tmp_train_file.name, "training_plots"
            )  # Only final plot goes here
            os.unlink(tmp_train_file.name)
            plt.close(final_main_fig)

            # Create final per-class F1 plot
            final_per_class_fig = create_f1_per_class_plot(
                train_f1_per_class, val_f1_per_class, epochs - 1, class_names # Pass updated class names
            )
            tmp_f1_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            final_per_class_fig.savefig(tmp_f1_file.name, dpi=150, bbox_inches="tight")
            tmp_f1_file.close()  # Close the file handle
            mlflow.log_artifact(tmp_f1_file.name, "f1_scores")  # New directory
            os.unlink(tmp_f1_file.name)
            plt.close(final_per_class_fig)

            # Create final per-class accuracy plot
            final_per_class_acc_fig = create_accuracy_per_class_plot(
                train_acc_per_class, val_acc_per_class, epochs - 1, class_names # Pass updated class names
            )
            tmp_acc_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            final_per_class_acc_fig.savefig(
                tmp_acc_file.name, dpi=150, bbox_inches="tight"
            )
            tmp_acc_file.close()  # Close the file handle
            mlflow.log_artifact(tmp_acc_file.name, "accuracy_scores")  # New directory
            os.unlink(tmp_acc_file.name)
            plt.close(final_per_class_acc_fig)

            # ADD: Save architecture configuration as JSON
            architecture_config = {
                "arch_name": model.arch_name,
                "input_channels": model.input_channels,
                "input_height": model.input_height, # Use the stored attribute
                "input_width": model.input_width,   # Use the stored attribute
                "feature_output_size": model.feature_output_size,
                "layers_config": model.layers_config,
                "total_params": total_params,
                "trainable_params": trainable_params,
            }
            # Save architecture config as JSON
            tmp_arch_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(architecture_config, tmp_arch_file, indent=2)
            tmp_arch_file.close()
            mlflow.log_artifact(tmp_arch_file.name, "model_architecture.json")
            os.unlink(tmp_arch_file.name)

            # Log model with input example and include architecture info
            sample_batch, _ = next(iter(train_loader))
            input_example = sample_batch[:1].numpy()
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=f"{run_name}_model",
                input_example=input_example,
                pip_requirements=[
                    "torch>=2.0.0",
                    "numpy>=1.21.0",
                    "mlflow>=2.0.0",
                    "matplotlib>=3.5.0",
                    "seaborn>=0.11.0",
                    "scikit-learn>=1.0.0",
                ],
            )

            print(
                f"Enhanced training completed. MLflow run ID: {mlflow.active_run().info.run_id}"
            )
            print(f"Final Weighted F1-Score: {final_f1_weighted:.4f}")
            print(f"Final Per-Class F1-Scores: {final_f1_per_class_str}")
            print(f"Final Per-Class Accuracy: {final_acc_per_class_str}")
            print(
                f"Logged artifacts: training_plots (final only), f1_scores, accuracy_scores, confusion_matrices, roc_curves, wrong_prediction_examples, reports/, model_architecture.json"
            )
            return model


    def evaluate_model(model, test_loader):
        """
        Evaluate the model and log metrics
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        # Log evaluation metrics if MLflow is active
        if mlflow.active_run():
            mlflow.log_metric("test_accuracy", accuracy)
        return accuracy


    def predict_single_image(model, image_array, threshold=0.0):
        """
        Predict a single image with your trained multi-label model
        Args:
            model: Trained PyTorch model
            image_array: numpy array of shape (H, W) or (H, W, C)
            threshold: threshold for converting logits to binary predictions (default 0.0)
        Returns:
            predictions: numpy array of shape (4,) with binary values [0, 1, 0, 1]
            logits: raw model outputs before thresholding
        """
        model.eval()  # Set model to evaluation mode

        # Preprocess the image to match training format
        if len(image_array.shape) == 2:  # (H, W) - Grayscale
            processed_image = np.expand_dims(image_array, axis=0)  # -> (1, H, W)
        elif len(image_array.shape) == 3:  # (H, W, 3) - RGB
            # Convert RGB to grayscale if needed
            processed_image = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
            processed_image = np.expand_dims(processed_image, axis=0)  # -> (1, H, W)

        # Normalize and convert to tensor
        processed_image = torch.FloatTensor(processed_image) / 255.0
        processed_image = processed_image.unsqueeze(
            0
        )  # Add batch dimension -> (1, 1, H, W)

        with torch.no_grad():  # Disable gradient computation for inference
            logits = model(processed_image)
            predictions = (
                logits > threshold
            ).float()  # Apply threshold to get binary predictions

        return predictions.numpy()[0], logits.numpy()[0]  # Remove batch dimension


    def predict_batch(model, image_batch, threshold=0.0):
        """
        Predict a batch of images
        Args:
            model: Trained PyTorch model
            image_batch: numpy array of shape (N, H, W) or (N, H, W, 1) or (N, H, W, 3)
            threshold: threshold for converting logits to binary predictions
        Returns:
            predictions: numpy array of shape (N, 4) with binary values
            logits: raw model outputs before thresholding
        """
        model.eval()

        # Convert to tensor and preprocess
        if isinstance(image_batch, np.ndarray):
            image_batch = torch.FloatTensor(image_batch)

        # Normalize
        image_batch = image_batch / 255.0

        # Add channel dimension if needed
        if len(image_batch.shape) == 3:  # (N, H, W)
            image_batch = image_batch.unsqueeze(1)  # -> (N, 1, H, W)
        elif len(image_batch.shape) == 4 and image_batch.shape[3] == 3:  # (N, H, W, 3)
            # Convert RGB to grayscale
            image_batch = torch.matmul(
                image_batch, torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
            )
            image_batch = image_batch.unsqueeze(1)  # -> (N, 1, H, W)

        with torch.no_grad():
            logits = model(image_batch)
            predictions = (logits > threshold).float()

        return predictions.numpy(), logits.numpy()


    # To interpret the results:
    def interpret_predictions(
        predictions, class_names=["forward", "back", "left", "right"]
    ):
        """
        Interpret the model predictions
        """
        result = {}
        for i, pred in enumerate(predictions):
            result[class_names[i]] = int(pred)

        # Print human-readable result
        active_actions = [
            class_names[i] for i, pred in enumerate(predictions) if pred > 0
        ]
        print(f"Predicted actions: {active_actions}")
        print(f"Detailed: {result}")
        return result
    return (FlexibleCNN,)


@app.cell
def _(Dataset, np, torch):
    class CnnImageDataset(Dataset):
        def __init__(self, images_df, labels_df):
            """
            images_df: Polars DataFrame with image arrays
            labels_df: Polars DataFrame with one-hot encoded labels (multi-label format)
            """
            self.images = images_df[images_df.columns[0]].to_list()
            # Keep one-hot encoded labels for multi-label classification
            self.labels = labels_df.to_numpy().astype(np.float32)  # Convert to float32

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]  # This is already one-hot encoded

            # Handle image dimensions
            if len(image.shape) == 2:  # (H, W) - Grayscale
                image = np.expand_dims(image, axis=0)  # -> (1, H, W)
            else:
                Exception("The image isn't gray scale")

            # Convert to tensor and normalize
            image = torch.FloatTensor(image) / 255.0
            label = torch.FloatTensor(label)  # Keep as one-hot vector for multi-label

            return image, label
    return (CnnImageDataset,)


@app.cell
def _(
    CnnImageDataset,
    DataLoader,
    cnn_X_test,
    cnn_X_train,
    cnn_y_test,
    cnn_y_train,
):
    # Create datasets and data loaders
    cnn_train_dataset = CnnImageDataset(cnn_X_train, cnn_y_train)
    cnn_test_dataset = CnnImageDataset(cnn_X_test, cnn_y_test)

    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True)
    cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=32, shuffle=False)
    return


@app.cell
def _(cnn_X_train):
    sample_image = cnn_X_train[cnn_X_train.columns[0]].to_list()[0]
    print(f"Actual image shape: {sample_image.shape}")
    return


@app.cell
def _(load_dotenv, os):
    load_dotenv()
    cnn_experiment_name = os.getenv(
        "MLFLOW_CNN_EXPERIMENT_NAME", "default_cnn_experiment"
    )
    return


@app.cell
def _(FlexibleCNN, images_height, images_width, mlflow, os):
    cnn_architectures = [
        (
            "simple_cnn",
            [
                # Feature extraction
                (
                    "conv",
                    {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                ),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.2}),
                (
                    "conv",
                    {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                ),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.2}),
                # Classifier
                ("linear", {"out_features": 16}),
                ("relu", {}),
                ("dropout", {"p": 0.2}),
                ("linear", {"out_features": 4}),
            ],
        ),
        (
            "not_so_simple",
            [
                # Feature extraction
                (
                    "conv",
                    {"out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2},
                ),
                ("batchnorm", {}),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.3}),
                (
                    "conv",
                    {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                ),
                ("batchnorm", {}),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.3}),
                (
                    "conv",
                    {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1},
                ),
                ("batchnorm", {}),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.3}),
                (
                    "conv",
                    {"out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1},
                ),
                ("batchnorm", {}),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.4}),
                # Classifier
                ("flatten", {}),
                ("linear", {"out_features": 256}),
                ("relu", {}),
                ("dropout", {"p": 0.4}),
                ("linear", {"out_features": 64}),
                ("relu", {}),
                ("dropout", {"p": 0.4}),
                ("linear", {"out_features": 4}),
            ],
        ),
    ]

    cnn_model = FlexibleCNN(
        cnn_architectures[1],
        input_channels=1,
        input_height=images_height,
        input_width=images_width,
    )

    print(
        f"Model '{cnn_model.arch_name}' created with {cnn_model.input_channels} input channel(s) and MLflow integration."
    )
    # Use a local path relative to your script's directory
    _local_mlruns_path = os.path.abspath("./mlruns")
    mlflow.set_tracking_uri(f"file://{_local_mlruns_path}")

    # Ensure the directory exists and is writable
    os.makedirs(_local_mlruns_path, exist_ok=True)
    assert os.access(_local_mlruns_path, os.W_OK), (
        f"MLflow directory '{_local_mlruns_path}' not writable!"
    )

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    cnn_weights = [1.5, 1.0, 2.0, 2.0]  # Forward / Back / Left / Right

    """
    train_model(
        cnn_model,
        cnn_train_loader,
        cnn_test_loader,
        epochs=1,
        learning_rate=0.001,
        experiment_name=cnn_experiment_name,
        run_name="test",
        weights=cnn_weights
    )
    """

    return


@app.cell
def _(mo):
    mo.md(r"""### Temporal CNN""")
    return


@app.cell(disabled=True)
def _():
    num_previous_frames = 5
    return (num_previous_frames,)


@app.cell(disabled=True)
def _(pl):
    def create_df_previous_frames(df_augmented, num_previous_frames):
        # Sort by record and frame_idx to ensure proper ordering
        df_sorted = df_augmented.sort(["record", "frame_idx"])

        # Create the DataFrame that will hold the results
        result_rows = []

        for record in df_sorted["record"].unique():
            # Separate augmented and non-augmented frames
            df_record_augmented = df_sorted.filter(
                (pl.col("record") == record) & (pl.col("is_augmented") == True)
            ).sort("frame_idx")
            df_record_original = df_sorted.filter(
                (pl.col("record") == record) & (pl.col("is_augmented") == False)
            ).sort("frame_idx")

            # Process augmented frames (skip first num_previous_frames)
            for current_idx in range(num_previous_frames, len(df_record_augmented)):
                current_row = df_record_augmented.row(current_idx, named=True)
                current_frame = current_row["frame_idx"]

                new_row = current_row.copy()

                for prev_frame in range(1, num_previous_frames + 1):
                    prev_frame_idx = current_frame - prev_frame
                    prev_rows = df_record_augmented.filter(
                        pl.col("frame_idx") == prev_frame_idx
                    )
                    if len(prev_rows) > 0:
                        prev_row = prev_rows.row(0, named=True)
                        # Add only the image and control columns from the previous augmented frame
                        new_row[f"image_preprocessed_prev_{prev_frame}"] = prev_row[
                            "image_preprocessed"
                        ]
                        new_row[f"forward_prev_{prev_frame}"] = prev_row["forward"]
                        new_row[f"back_prev_{prev_frame}"] = prev_row["back"]
                        new_row[f"left_prev_{prev_frame}"] = prev_row["left"]
                        new_row[f"right_prev_{prev_frame}"] = prev_row["right"]
                    else:
                        new_row[f"image_preprocessed_prev_{prev_frame}"] = None

                result_rows.append(new_row)

            # Process original (non-augmented) frames (skip first num_previous_frames)
            for current_idx in range(num_previous_frames, len(df_record_original)):
                current_row = df_record_original.row(current_idx, named=True)
                current_frame = current_row["frame_idx"]

                new_row = current_row.copy()

                for prev_frame in range(1, num_previous_frames + 1):
                    prev_frame_idx = current_frame - prev_frame
                    prev_rows = df_record_original.filter(
                        pl.col("frame_idx") == prev_frame_idx
                    )
                    if len(prev_rows) > 0:
                        prev_row = prev_rows.row(0, named=True)
                        # Add only the image and control columns from the previous original frame
                        new_row[f"image_preprocessed_prev_{prev_frame}"] = prev_row[
                            "image_preprocessed"
                        ]
                        new_row[f"forward_prev_{prev_frame}"] = prev_row["forward"]
                        new_row[f"back_prev_{prev_frame}"] = prev_row["back"]
                        new_row[f"left_prev_{prev_frame}"] = prev_row["left"]
                        new_row[f"right_prev_{prev_frame}"] = prev_row["right"]
                    else:
                        new_row[f"image_preprocessed_prev_{prev_frame}"] = None

                result_rows.append(new_row)

        # Create final DataFrame and sort
        df_result = pl.DataFrame(result_rows)
        return df_result.sort(["record", "frame_idx", "is_augmented"])
    return (create_df_previous_frames,)


@app.cell(disabled=True)
def _(create_df_previous_frames, df_augmented, num_previous_frames):
    df_previous_frames = create_df_previous_frames(df_augmented, num_previous_frames)
    df_previous_frames.head()
    return (df_previous_frames,)


@app.cell(disabled=True)
def _(df_previous_frames, np, num_previous_frames, pl):
    # Get all image columns that need to be converted
    image_columns = ["image_preprocessed"] + [
        f"image_preprocessed_prev_{i}" for i in range(1, num_previous_frames + 1)
    ]

    # Convert all image columns to numpy arrays
    df_previous_frames_array_image = df_previous_frames.with_columns(
        [
            pl.col(col)
            .map_elements(
                lambda pil_img: np.array(pil_img) if pil_img is not None else None,
                return_dtype=pl.Object,
            )
            .alias(f"2d_array_{col}")
            for col in image_columns
        ]
    )
    return (df_previous_frames_array_image,)


@app.cell
def _(df_previous_frames_array_image):
    df_previous_frames_array_image.head()
    return


@app.cell(disabled=True)
def _(df_previous_frames_array_image, np, num_previous_frames, pl):
    def stack_images(row_dict, num_prev_frames):
        """Stack images along the channel dimension"""
        images = []

        # Add previous frames in reverse order (prev_n, ..., prev_1) then current
        for i in range(num_prev_frames, 0, -1):
            img = row_dict.get(f"2d_array_image_preprocessed_prev_{i}")
            if img is not None:
                images.append(img)

        # Add current frame
        current_img = row_dict.get("2d_array_image_preprocessed")
        if current_img is not None:
            images.append(current_img)

        if not images:
            return None

        # Stack along channel dimension (last axis)
        # Assuming images are 2D (grayscale) or 3D (H, W, C)
        try:
            stacked = np.stack(images, axis=-1)
            return stacked
        except ValueError as e:
            print(f"Error stacking images: {e}")
            print(f"Number of images to stack: {len(images)}")
            for i, img in enumerate(images):
                print(f"Image {i} shape: {img.shape if img is not None else 'None'}")
            return None

    # Now create the stacked arrays by iterating through rows
    stacked_arrays = []
    for _row in df_previous_frames_array_image.iter_rows(named=True):
        stacked = stack_images(_row, num_previous_frames)
        stacked_arrays.append(stacked)

    # Add the stacked column to the DataFrame
    df_previous_frames_array_stacked = df_previous_frames_array_image.with_columns(
        pl.Series("stacked_array_image_sequence", stacked_arrays, dtype=pl.Object)
    )
    return (df_previous_frames_array_stacked,)


@app.cell(disabled=True)
def _(df_previous_frames_array_stacked):
    df_previous_frames_array_stacked["stacked_array_image_sequence"][0].shape
    return


@app.cell(disabled=True)
def _(df_previous_frames_array_stacked):
    df_previous_frames_array_stacked.head()
    return


@app.cell(disabled=True)
def _(df_previous_frames_array_stacked, train_test_split):
    temporal_cnn_df_train, temporal_cnn_df_test = train_test_split(
        df_previous_frames_array_stacked, test_size=0.2, random_state=42
    )

    # --- 2. Prepare feature and label columns
    temporal_cnn_feature_col = "stacked_array_image_sequence"

    # --- 3. Create X (inputs) as Polars DataFrames
    temporal_cnn_X_train = temporal_cnn_df_train.select(temporal_cnn_feature_col)
    temporal_cnn_X_test = temporal_cnn_df_test.select(temporal_cnn_feature_col)

    temporal_cnn_y_train = temporal_cnn_df_train.select(
        "forward", "back", "left", "right"
    )

    temporal_cnn_y_test = temporal_cnn_df_test.select(
        "forward", "back", "left", "right"
    )
    return (
        temporal_cnn_X_test,
        temporal_cnn_X_train,
        temporal_cnn_y_test,
        temporal_cnn_y_train,
    )


@app.cell(disabled=True)
def _(temporal_cnn_X_test):
    temporal_cnn_X_test.head()
    return


@app.cell(disabled=True)
def _(temporal_cnn_y_test):
    temporal_cnn_y_test.head()
    return


@app.cell(disabled=True)
def _(temporal_cnn_X_train):
    temporal_sample_image = temporal_cnn_X_train[
        temporal_cnn_X_train.columns[0]
    ].to_list()[0]
    print(f"Actual image shape: {temporal_sample_image.shape}")
    return


@app.cell(disabled=True)
def _(Dataset, np, torch):
    class TemporalCnnImageDataset(Dataset):
        def __init__(self, images_df, labels_df):
            """
            images_df: Polars DataFrame with stacked image arrays in 'stacked_array_image_sequence' column
            labels_df: Polars DataFrame with one-hot encoded labels (multi-label format)
            """
            self.images = images_df["stacked_array_image_sequence"].to_list()
            # Keep one-hot encoded labels for multi-label classification
            self.labels = labels_df.to_numpy().astype(np.float32)  # Convert to float32

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]  # This is already one-hot encoded

            # Handle image dimensions
            if len(image.shape) == 3:  # (H, W, C) - Stacked sequence
                # Transpose from (H, W, C) to (C, H, W) for PyTorch
                image = np.transpose(image, (2, 0, 1))
            else:
                raise Exception(
                    f"Expected 3D image (H, W, C), got shape: {image.shape}"
                )

            # Convert to tensor and normalize
            image = torch.FloatTensor(image) / 255.0
            label = torch.FloatTensor(label)  # Keep as one-hot vector for multi-label

            return image, label
    return (TemporalCnnImageDataset,)


@app.cell(disabled=True)
def _(
    DataLoader,
    TemporalCnnImageDataset,
    temporal_cnn_X_test,
    temporal_cnn_X_train,
    temporal_cnn_y_test,
    temporal_cnn_y_train,
):
    # Create datasets and data loaders
    temporal_cnn_train_dataset = TemporalCnnImageDataset(
        temporal_cnn_X_train, temporal_cnn_y_train
    )
    temporal_cnn_test_dataset = TemporalCnnImageDataset(
        temporal_cnn_X_test, temporal_cnn_y_test
    )

    temporal_cnn_train_loader = DataLoader(
        temporal_cnn_train_dataset, batch_size=32, shuffle=True
    )
    temporal_cnn_test_loader = DataLoader(
        temporal_cnn_test_dataset, batch_size=32, shuffle=False
    )
    return


@app.cell
def _(load_dotenv, os):
    load_dotenv()
    temporal_cnn_experiment_name = os.getenv(
        "MLFLOW_TEMPORAL_CNN_EXPERIMENT_NAME", "default_cnn_experiment"
    )
    return


@app.cell(disabled=True)
def _(
    FlexibleCNN,
    images_height,
    images_width,
    mlflow,
    num_previous_frames,
    os,
):
    lstm_architectures = [
        (
            "simple_cnn",
            [
                # Feature extraction
                (
                    "conv",
                    {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                ),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.2}),
                (
                    "conv",
                    {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                ),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.2}),
                # Classifier
                ("linear", {"out_features": 16}),
                ("relu", {}),
                ("dropout", {"p": 0.2}),
                ("linear", {"out_features": 4}),
            ],
        ),
        (
            "not_so_simple_cnn",
            [
                # Feature extraction
                (
                    "conv",
                    {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                ),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.2}),
                (
                    "conv",
                    {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                ),
                ("relu", {}),
                ("maxpool", {"kernel_size": 2, "stride": 2}),
                ("dropout", {"p": 0.2}),
                # Classifier
                ("linear", {"out_features": 32}),
                ("relu", {}),
                ("dropout", {"p": 0.2}),
                ("linear", {"out_features": 32}),
                ("relu", {}),
                ("dropout", {"p": 0.2}),
                ("linear", {"out_features": 4}),
            ],
        ),
    ]

    temporal_cnn_model = FlexibleCNN(
        lstm_architectures[1],
        input_channels=num_previous_frames + 1,
        input_height=images_height,
        input_width=images_width,
    )

    print(
        f"Model '{temporal_cnn_model.arch_name}' created with {temporal_cnn_model.input_channels} input channel(s) and MLflow integration."
    )
    # Use a local path relative to your script's directory
    _local_mlruns_path = os.path.abspath("./mlruns")
    mlflow.set_tracking_uri(f"file://{_local_mlruns_path}")

    # Ensure the directory exists and is writable
    os.makedirs(_local_mlruns_path, exist_ok=True)
    assert os.access(_local_mlruns_path, os.W_OK), (
        f"MLflow directory '{_local_mlruns_path}' not writable!"
    )

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    temporal_cnn_weights = [1.5, 1.0, 2.0, 2.0]  # Forward / Back / Left / Right

    """"
    train_model(
        temporal_cnn_model,
        temporal_cnn_train_loader,
        temporal_cnn_test_loader,
        epochs=1,
        learning_rate=0.001,
        experiment_name=temporal_cnn_experiment_name,
        run_name="test",
        weights=temporal_cnn_weights
    )
    """

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
