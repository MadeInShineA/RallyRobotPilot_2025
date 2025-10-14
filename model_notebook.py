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
    return lzma, mo, os, pickle, pl


@app.cell
def _(mo):
    mo.md(r"""## Load the different records""")
    return


@app.cell
def _(lzma, os, pickle, pl):
    record_dir = "./records/"

    output_dir = "./records_csv/"
    os.makedirs(output_dir, exist_ok=True)

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

        # Build list of records WITHOUT 'image'
        records = []
        for idx, s in enumerate(snapshots):
            record = {
                "forward": s.current_controls[0],
                "back": s.current_controls[1],
                "left": s.current_controls[2],
                "right": s.current_controls[3],
                "car_speed": s.car_speed,
                "car_angle": s.car_angle,
                **{f"raycast_{i}": float(d) for i, d in enumerate(s.raycast_distances)},
                # Optionally save image to disk and store path
                # "image_path": f"{filename}_frame_{idx}.npy"
            }
            records.append(record)

            # Optional: Save image separately
            # image_path = os.path.join(image_dir, f"{filename}_frame_{idx}.npy")
            # np.save(image_path, s.image)

        try:
            df = pl.DataFrame(records)
            csv_filename = filename.replace(".npz", ".csv")
            output_path = os.path.join(output_dir, csv_filename)
            df.write_csv(output_path)
            print(f"✅ Converted {filename} → {csv_filename} (excluded image data)")
        except Exception as e:
            print(f"❌ Error converting {filename}: {e}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
