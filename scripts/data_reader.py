if __name__ == "__main__":
    import numpy
    import pickle
    import lzma

    with lzma.open("record_1.npz", "rb") as file:
        data = pickle.load(file)

        print("Read", len(data), "snapshotwas")
        for data_point in data:
            print(f"Angle: {data_point.car_angle}")
