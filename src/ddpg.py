import numpy as np


# thanks to Machine Learning with Phil on YouTube for the video guide
# on implementing this algorithm
# https://www.youtube.com/watch?v=6Yd5WnYls_Y

# implement the noise class
class QUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    # calculate the noise (I dont know the math behind it, this is a wrote Copy paste of the formulas)
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
        self.sigma * np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)