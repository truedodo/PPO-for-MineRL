import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


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

# Buffer to store <state, action, reward, next_state, terminal> tuples from episodes
# will sample from this to train the networks with mini-batches
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


# Example critic network for DDPG that has 2 fully connected hidden layers
# TODO figure how to integrate this with MineRL...
class ExampleCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ExampleCriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        # define NN layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # not sure why the paramters are initialized like this
        # sure it's in the paper somewhere, but not that relevent if we are tuning VPT
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        # add a batch norm layer
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        # add a batch norm layer
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # a linear layer for processing the action value
        self.action_value = nn.Linear(self.n_actions, fc2_dims)


        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)


        ## other important NN initialization steps for pytorch
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # actually pass data through the network layers
        # notice how we manually call activation functions
        # instead of defining them with layers
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        # double relu on action value? weird?
        action_value = F.relu(self.action_value(action))
        # this is how states and actions are integrated into this network
        # we just add them together after passing them through the state
        # and action layers, interesting
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value
    
    def save_checkpoint(self):
        print("...saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))
        