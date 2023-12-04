import numpy as np


class KalmanFilter:
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):

        self.n = F.shape[1]

        self.F = F
        self.H = H
        self.B = B  # Control matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P if (P is not None) else np.eye(
            self.n)  # error covariance matrix
        self.x = x0 if (x0 is not None) else np.zeros(
            (self.n, 1))  # initial state

    def predict(self, u=None):
        if self.B is not None and u is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


def example():
    dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    measurements = - (x**2 + 2*x - 2) + np.random.normal(0, 2, 100)

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
    predictions = []

    for z in measurements:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    import matplotlib.pyplot as plt
    plt.plot(range(len(measurements)), measurements, label='Measurements')
    plt.plot(range(len(predictions)), np.array(
        predictions), label='Kalman Filter Prediction')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example()
