import numpy as np
import pandas as pd

from tracking.kalman_filter import KalmanFilter

class Tracklet:
    def __init__(self, track_id=0):
        self.track_id = track_id
        self.positions = []

        vx = 1
        vy = 1

        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0

        # Create a new Kalman filter instance for this track
        self.kf = KalmanFilter()

        # Set initial state
        initial_state = np.array([x1, y1, x2, y2, vx, vy]).reshape(
            6, 1
        )  # Include zero initial velocities

        self.kf.kf.x = initial_state  # Assigns the initial state directly to the filter's state vector

    def update_position(self, x1, y1):
        self.positions.append((x1, y1))  # Append new position to the history

    def get_width(self):
        return self.x2 - self.x1

    def get_height(self):
        return self.y2 - self.y1

    def get_bottom_right(self):
        return self.x1 + self.w, self.y1 + self.h

    def get_z(self):
        """
        Get measurement vector z for Kalman filtering.
        Only positions are measured.
        """
        return np.array([self.x1, self.y1, self.x2, self.y2]).reshape(-1, 1)

    def set_state(self, state):
        self.x1 = state[0]
        self.y1 = state[1]
        self.x2 = state[2]
        self.y2 = state[3]
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1

    def predict(self):
        # Predict the new state
        pred_state = self.kf.predict()

        # Update the track with the new predicted state
        self.set_state(pred_state)
        self.update_position(pred_state[0][0], pred_state[1][0])
        return pred_state

    def update_and_predict(self, x1, y1, x2, y2):

        z = np.array([x1, y1, x2, y2]).reshape(-1, 1)
        # Update Kalman filter with the measurement vector (detection)
        self.kf.update(z)

        # Predict the next state
        return self.predict()


# sample usage
if __name__ == "__main__":
    
    # load sample bounding boxes data
    base_path = "drone-yolov7-dataset/valid"
    df = pd.read_csv(f"{base_path}/converted_labels/processed_data.csv")

    def convert_bbox_format(df):
        # Calculate the corner coordinates
        df["x1"] = df["x_center"] - df["width"] / 2
        df["y1"] = df["y_center"] - df["height"] / 2
        df["x2"] = df["x_center"] + df["width"] / 2
        df["y2"] = df["y_center"] + df["height"] / 2

        # Creating a dictionary format for each row if needed
        df["bbox_corners"] = df.apply(
            lambda row: {
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
            },
            axis=1,
        )
        return df

    # Convert and print new DataFrame
    converted_df = convert_bbox_format(df)

    movie = (
        converted_df[
            (converted_df["image_number"] >= 2) & (converted_df["image_number"] <= 39)
        ]
        .sort_values(by="image_number", ascending=True)
        .reset_index(drop=False)
    )
    frames = movie[["bbox_corners", "image"]]

    # main loop
    tracklet = Tracklet() 
    for i, frame in frames.iterrows():
        # only measure every 5 frames
        if i % 5 == 0:
            tracklet.update_and_predict(**frame["bbox_corners"])
        # else we only predict / estimate
        else:
            tracklet.predict()
        print(i)
