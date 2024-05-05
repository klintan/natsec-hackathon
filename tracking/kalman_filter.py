from filterpy.kalman import KalmanFilter as KF
import numpy as np


class KalmanFilter:
    def __init__(self, dt=1 / 24.0):
        self.kf = KF(dim_x=6, dim_z=4)  # 6 state variables, 4 measurements

        # State Transition Matrix for constant velocity model
        self.kf.F = np.eye(6)
        self.kf.F[0, 4] = dt  # x1 velocity influence
        self.kf.F[1, 5] = dt  # y1 velocity influence
        self.kf.F[2, 4] = dt  # x2 velocity influence
        self.kf.F[3, 5] = dt  # y2 velocity influence

        # Measurement function: we only measure positions
        self.kf.H = np.zeros((4, 6))
        self.kf.H[0, 0] = 1  # Measure x1
        self.kf.H[1, 1] = 1  # Measure y1
        self.kf.H[2, 2] = 1  # Measure x2
        self.kf.H[3, 3] = 1  # Measure y2

        # Initial state covariance
        self.kf.P *= 1000  # Start with a large uncertainty

        # Measurement noise covariance
        self.kf.R = np.diag(
            [1, 1, 1, 1]
        )  # Smaller values indicate high confidence in the measurements

        # Process noise covariance
        self.kf.Q = np.eye(6) * 0.1  # Adjust based on movement variability

    def update(self, z):
        """Update the Kalman Filter from a new measurement"""
        self.kf.update(z)

    def predict(self):
        """Predict the next state of the Kalman Filter"""
        self.kf.predict()
        return self.kf.x  # Return the predicted state
