import numpy as np


# thanks to Machine Learning with Phil on YouTube for the video guide
# on implementing this algorithm
# https://www.youtube.com/watch?v=6Yd5WnYls_Y

# implement the noise class
# use this to create the exploratory behaviour policy beta based on the
# deterministic action network
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

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        # store pieces of the memory in different np arrays
        # * operator unpacks tuples in python, FYI
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    # state in this case is basically the obs
    def store_transition(self, state, action, reward, next_state, done):
        # wrapping index to override oldest memory when you reach limit
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = 1-done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):

        # generate indices for random indexing
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        # grab random samples from the buffer
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminals
