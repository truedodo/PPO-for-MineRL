
import pickle
import sys
import time
from typing import List
import gym
import torch as th
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from datetime import datetime
from tqdm import tqdm
from memory import Memory, MemoryDataset
from util import to_torch_tensor, normalize, safe_reset, hard_reset, calculate_gae
from vectorized_minerl import *

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8
from lib.tree_util import tree_map  # nopep8

# For debugging purposes
# th.autograd.set_detect_anomaly(True)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# device = th.device("mps")  # apple silicon

TRAIN_WHOLE_MODEL = False



class PhasicPolicyGradient:
    def __init__(
            self,
            env_name: str,
            model: str,
            weights: str,
            out_weights: str,
            save_every: int,


            # Hyperparameters
            num_rollouts: int,
            num_steps: int,  # the number of steps per rollout, T
            epochs: int,
            minibatch_size: int,
            lr: float,
            weight_decay: float,
            betas: tuple,
            beta_s: float,
            eps_clip: float,
            value_clip: float,
            value_loss_weight: float,
            gamma: float,
            lam: float,


            mem_buffer_size: int,
            # tau: float = 0.95,

            # Optional: plot stuff
            plot: bool,
            num_envs: int

    ):
        model_path = f"models/{model}.model"
        weights_path = f"weights/{weights}.weights"
        self.out_weights_path = f"weights/{out_weights}.weights"
        self.training_name = f"ppo-{env_name}-{out_weights}-{int(time.time())}"

        self.env_name = env_name
        self.num_envs = num_envs
        self.envs = init_vec_envs(self.env_name, self.num_envs)

        self.save_every = save_every

        # Load hyperparameters unchanged
        self.num_rollouts = num_rollouts
        self.num_steps = num_steps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.value_loss_weight = value_loss_weight
        self.gamma = gamma
        self.lam = lam
        # self.tau = tau

        self.plot = plot

        self.mem_buffer_size = mem_buffer_size

        if self.plot:
            self.pi_loss_history = []
            self.v_loss_history = []
            self.total_loss_history = []

            self.entropy_history = []
            self.expl_var_history = []

            self.surr1_history = []
            self.surr2_history = []

            self.reward_history = []

            # These statistics are calculated live during the episode running (rollout)
            self.live_reward_history = []
            self.live_value_history = []
            self.live_gae_history = []

        # Load the agent parameters from the weight files
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])


        # Make our agents

        self.agent = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        actor = self.agent.policy
        policy_params = actor.parameters()

        self.optim = th.optim.Adam(
            policy_params, lr=lr, betas=betas, weight_decay=weight_decay)
        
        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.optim, lambda x: 1 - x / num_rollouts)
        
        
        # Create a SEPARATE VPT agent just for the critic
        self.critic = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
                                pi_head_kwargs=pi_head_kwargs)
        self.critic.load_weights(weights_path)
        critic_params = list(self.critic.policy.value_head.parameters()) + list(self.critic.policy.net.parameters())

        # separate optimizer for the critic
        self.critic_optim = th.optim.Adam(critic_params, lr=lr, betas=betas, weight_decay=weight_decay)

        self.scheduler_critic = th.optim.lr_scheduler.LambdaLR(
        self.critic_optim, lambda x: 1 - x / num_rollouts)

        # Internal buffer of the most recent episode memories
        # This will be a relatively large chunk of data
        # Potential memory issues / optimizations around here...
        self.memories: List[Memory] = []


    def policy(self):
        '''
        Returns the policy network head, aux value head, and base
        '''

        return self.agent.policy.pi_head, self.agent.policy.value_head, self.agent.policy.net


    def value(self):
        '''
        Return the current value network  head and base
        '''
        return self.critic.policy.value_head, self.critic.policy.net


    def pi_and_v(self, agent_obs, policy_hidden_state, value_hidden_state, dummy_first):
        """
        Returns the correct policy and value outputs
        """
        # Shorthand for networks
        policy, _, policy_base = self.policy()
        value, value_base = self.value()

        (pi_h, _), p_state_out = policy_base(
            agent_obs, policy_hidden_state, context={"first": dummy_first})
        (_, v_h), v_state_out = value_base(
            agent_obs, value_hidden_state, context={"first": dummy_first})
            
        return policy(pi_h), value(v_h), p_state_out, v_state_out


    def init_plots(self):
        plt.ion()
        self.main_fig, self.main_ax = plt.subplots(2, 3, figsize=(12, 8))
        self.live_fig, self.live_ax = plt.subplots(1, 1, figsize=(6, 4))

        # Set up policy loss plot
        self.main_ax[0, 0].set_autoscale_on(True)
        self.main_ax[0, 0].autoscale_view(True, True, True)

        self.main_ax[0, 0].set_title("Policy Loss")

        self.pi_loss_plot, = self.main_ax[0, 0].plot(
            [], [], color="blue")

        # Setup value loss plot
        self.main_ax[0, 1].set_autoscale_on(True)
        self.main_ax[0, 1].autoscale_view(True, True, True)

        self.main_ax[0, 1].set_title("Value Loss")

        self.v_loss_plot, = self.main_ax[0, 1].plot(
            [], [], color="orange")

        # Set up total loss plot
        self.main_ax[0, 2].set_autoscale_on(True)
        self.main_ax[0, 2].autoscale_view(True, True, True)

        self.main_ax[0, 2].set_title("Total Loss")

        self.total_loss_plot, = self.main_ax[0, 2].plot(
            [], [], color="purple"
        )

        # Setup entropy plot
        self.main_ax[1, 0].set_autoscale_on(True)
        self.main_ax[1, 0].autoscale_view(True, True, True)
        self.main_ax[1, 0].set_title("Entropy")

        self.entropy_plot, = self.main_ax[1, 0].plot([], [], color="green")

        # Setup explained variance plot
        self.main_ax[1, 1].set_autoscale_on(True)
        self.main_ax[1, 1].autoscale_view(True, True, True)
        self.main_ax[1, 1].set_title("Explained Vaiance")

        self.expl_var_plot, = self.main_ax[1, 1].plot([], [], color="grey")

        # Setup reward plot
        self.main_ax[1, 2].set_autoscale_on(True)
        self.main_ax[1, 2].autoscale_view(True, True, True)
        self.main_ax[1, 2].set_title("Reward per Rollout Phase")

        self.reward_plot,  = self.main_ax[1, 2].plot([], [], color="red")

        # Setup live plots
        self.live_ax.set_autoscale_on(True)
        self.live_ax.autoscale_view(True, True, True)
        self.live_ax.set_title("Episode Progress")
        self.live_ax.set_xlabel("steps")

        self.live_reward_plot, = self.live_ax.plot(
            [], [], color="red", label="Reward")
        self.live_value_plot, = self.live_ax.plot(
            [], [], color="blue", label="Value")
        self.live_gae_plot, = self.live_ax.plot(
            [], [], color="green", label="GAE")

        self.live_ax.legend(loc="upper right")

    def rollout(self, env, next_obs=None, next_done=False, next_policy_hidden_state=None, hard_reset: bool = False):
        """
        Runs a rollout on the vectorized environments for `num_steps` timesteps and records the memories

        Returns `next_obs` and `next_done` for starting the next section of rollouts
        """
        start = datetime.now()

        # Temporary buffer to put the memories in before extending self.memories
        rollout_memories: List[Memory] = []

        # Only run this in the beginning
        if next_obs is None:
            # Initialize the hidden state vector
            next_policy_hidden_state = self.agent.policy.initial_state(1)
            next_critic_hidden_state = self.critic.policy.initial_state(1)

            # MineRL specific technical setup

            # Need to do a little bit of augmentation so the dataloader accepts the initial hidden state
            # This shouldn't affect anything; initial_state just uses None instead of empty tensors
            for i in range(len(next_policy_hidden_state)):
                next_policy_hidden_state[i] = list(next_policy_hidden_state[i])
                next_policy_hidden_state[i][0] = th.from_numpy(np.full(
                    (1, 1, 128), False)).to(device)
                
            for i in range(len(critic_hidden_state)):
                critic_hidden_state[i] = list(critic_hidden_state[i])
                critic_hidden_state[i][0] = th.from_numpy(np.full(
                    (1, 1, 128), False)).to(device)

            if hard_reset:
                env.close()
                env = gym.make(self.env_name)
                next_obs = env.reset()
            else:
                next_obs = env.reset()

            next_done = False

        # This is a dummy tensor of shape (batchsize, 1) which was used as a mask internally
        dummy_first = th.from_numpy(np.array((False,))).to(device)
        dummy_first = dummy_first.unsqueeze(1)

        # This is not really used in training
        # More just for us to estimate the success of an episode
        episode_reward = 0

        if self.plot:
            self.live_reward_history.clear()
            self.live_value_history.clear()
            self.live_gae_history.clear()

        for _ in range(self.num_steps):
            obs = next_obs
            done = next_done
            policy_hidden_state = next_policy_hidden_state
            critic_hidden_state = next_critic_hidden_state

            # We have to do some resetting...
            if done:
                next_obs = env.reset()
                policy_hidden_state = self.agent.policy.initial_state(1)
                for i in range(len(policy_hidden_state)):
                    policy_hidden_state[i] = list(policy_hidden_state[i])
                    policy_hidden_state[i][0] = th.from_numpy(np.full(
                        (1, 1, 128), False)).to(device)
                    
                critic_hidden_state = self.critic.policy.initial_state(1)
                for i in range(len(critic_hidden_state)):
                    critic_hidden_state[i] = list(critic_hidden_state[i])
                    critic_hidden_state[i][0] = th.from_numpy(np.full(
                        (1, 1, 128), False)).to(device)

            # Preprocess image
            agent_obs = self.agent._env_obs_to_agent(obs)

            # Basically just adds a dimension to both camera and button tensors
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            with th.no_grad():
                pi_distribution, v_prediction, next_policy_hidden_state, next_critic_hidden_state \
                    = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first)

            action = self.agent.policy.pi_head.sample(
                pi_distribution, deterministic=False)

            # Get log probability of taking this action given pi
            action_log_prob = self.agent.policy.get_logprob_of_action(
                pi_distribution, action)

            # Process this so the env can accept it
            minerl_action = self.agent._agent_action_to_env(action)

            # Take action step in the environment
            next_obs, reward, next_done, info = env.step(minerl_action)

            # Immediately disregard the reward function from the environment
            # reward = self.rc.get_rewards(obs)
            episode_reward += reward

            # Important! When we store a memory, we want the hidden state at the time of the observation as input! Not the step after
            # This is because we need to fully recreate the input when training the LSTM part of the network
            memory = Memory(agent_obs, None, None, None, action, action_log_prob,
                            reward, 0, next_done, v_prediction)

            rollout_memories.append(memory)

            # Finally, render the environment to the screen
            # Comment this out if you are boring
            env.render()

            if self.plot:
                # Calculate the GAE up to this point
                v_preds = list(map(lambda mem: mem.value, rollout_memories))
                rewards = list(map(lambda mem: mem.reward, rollout_memories))
                masks = list(
                    map(lambda mem: 1 - float(mem.done), rollout_memories))

                returns = calculate_gae(
                    rewards, v_preds, masks, self.gamma, self.lam)

                # Update data
                self.live_reward_history.append(reward)
                self.live_value_history.append(v_prediction.item())
                self.live_gae_history = returns

                # Update the plots
                self.live_reward_plot.set_ydata(self.live_reward_history)
                self.live_reward_plot.set_xdata(
                    range(len(self.live_reward_history)))

                self.live_value_plot.set_ydata(self.live_value_history)
                self.live_value_plot.set_xdata(
                    range(len(self.live_value_history)))

                self.live_gae_plot.set_ydata(self.live_gae_history)
                self.live_gae_plot.set_xdata(
                    range(len(self.live_gae_history)))

                self.live_ax.relim()
                self.live_ax.autoscale_view(True, True, True)

                # Actually draw everything
                self.live_fig.canvas.draw()
                self.live_fig.canvas.flush_events()

        # Reset the reward calculator once we are done with the episode

        # Calculate the generalized advantage estimate
        # TODO need to use next_obs and next_done for this
        v_preds = list(map(lambda mem: mem.value, rollout_memories))
        rewards = list(map(lambda mem: mem.reward, rollout_memories))
        masks = list(map(lambda mem: 1 - float(mem.done), rollout_memories))

        returns = calculate_gae(rewards, v_preds, masks, self.gamma, self.lam)

        # Make changes to the memories for this episode before adding them to main buffer
        for i in range(len(rollout_memories)):
            # Replace raw reward with the GAE
            rollout_memories[i].reward = returns[i]

            # Remember the total reward for this episode
            rollout_memories[i].total_reward = episode_reward

        # Update internal memory buffer
        self.memories.extend(rollout_memories)

        if self.plot:
            # Update the reward plot
            self.reward_history.append(episode_reward)
            self.reward_plot.set_ydata(self.reward_history)
            self.reward_plot.set_xdata(range(len(self.reward_history)))

            self.main_ax[1, 2].relim()
            self.main_ax[1, 2].autoscale_view(True, True, True)

            self.main_fig.canvas.draw()
            self.main_fig.canvas.flush_events()

        end = datetime.now()
        print(
            f"✅ Rollout finished (duration: {end - start} | memories: {len(rollout_memories)} | total reward: {episode_reward})")

        return next_obs, next_done, next_policy_hidden_state

    def learn_ppo_phase(self):

        # Create dataloader from the memory buffer
        data = MemoryDataset(self.memories)

        # Shuffle is FALSE here because we are training the entire networks and rollout out again...
        dl = DataLoader(data, batch_size=self.minibatch_size, shuffle=False)

        # Shorthand
        policy, aux, policy_base = self.policy()
        value, value_base = self.value()

        for _ in tqdm(range(self.epochs), desc="🧠 Epochs"):

            # Note: These are batches, not individual samples
            for mem in dl:
                agent_obs, state, recorded_pi_h, recorded_v_h, actions, old_action_log_probs,\
                    rewards, total_rewards, dones, v_old = mem
                
                batch_size = len(dones)
                v_old = v_old.to(device)

                # DATALOADER WHY DO YOU LOVE ADDING DIMENSIONS?!!
                for i in range(len(state)):
                    state[i][0] = state[i][0].squeeze(1)
                    state[i][1][0] = state[i][1][0].squeeze(1)
                    state[i][1][1] = state[i][1][1].squeeze(1)
                agent_obs = tree_map(lambda x: x.squeeze(1), agent_obs)

                policy_hidden_state, critic_hidden_state = state
                pi_distribution, v_prediction = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state)
                v_prediction.to(device)

                # The returns are stored in the `reward` field in memory, for some reason
                returns = normalize(rewards)

                # Calculate the explained variance, to see how accurate the GAE really is...
                explained_variance = 1 - \
                    th.sub(returns, v_prediction).var() / returns.var()

                # Get log probs
                action_log_probs = self.agent.policy.get_logprob_of_action(
                    pi_distribution, actions)

                entropy = self.agent.policy.pi_head.entropy(
                    pi_distribution).to(device)

                # Calculate clipped surrogate objective loss
                ratios = (action_log_probs -
                          old_action_log_probs).exp().to(device)
                advantages = normalize(
                    returns - v_prediction.detach().to(device))
                surr1 = ratios * advantages
                surr2 = ratios.clamp(
                    1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - th.min(surr1, surr2) - self.beta_s * entropy

                # Backprop for policy
                self.optim.zero_grad()
                policy_loss.backward()
                self.optim.step()

                # Calculate unclipped value loss
                value_loss = 0.5 * (v_prediction.squeeze() - returns) ** 2
                # Backprop for critic
                self.optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step()

                if self.plot:
                    self.pi_loss_history.append(policy_loss.mean().item())
                    self.v_loss_history.append(value_loss.item())

                    self.expl_var_history.append(explained_variance.item())

                    self.entropy_history.append(entropy.mean().item())

            # Update plot at the end of every epoch
            if self.plot:
                # Update policy loss plot
                self.pi_loss_plot.set_ydata(self.pi_loss_history)
                self.pi_loss_plot.set_xdata(
                    range(len(self.pi_loss_history)))
                self.main_ax[0, 0].relim()
                self.main_ax[0, 0].autoscale_view(True, True, True)

                # Update value loss plot
                self.v_loss_plot.set_ydata(self.v_loss_history)
                self.v_loss_plot.set_xdata(
                    range(len(self.v_loss_history)))
                self.main_ax[0, 1].relim()
                self.main_ax[0, 1].autoscale_view(True, True, True)

                # Update total loss plot
                self.total_loss_plot.set_ydata(self.total_loss_history)
                self.total_loss_plot.set_xdata(
                    range(len(self.total_loss_history)))
                self.main_ax[0, 2].relim()
                self.main_ax[0, 2].autoscale_view(True, True, True)

                # Update the entropy plot
                self.entropy_plot.set_ydata(self.entropy_history)
                self.entropy_plot.set_xdata(range(len(self.entropy_history)))
                self.main_ax[1, 0].relim()
                self.main_ax[1, 0].autoscale_view(True, True, True)

                # Update the explained variance plot
                self.expl_var_plot.set_ydata(self.expl_var_history)
                self.expl_var_plot.set_xdata(
                    range(len(self.expl_var_history)))

                self.main_ax[1, 1].relim()
                self.main_ax[1, 1].autoscale_view(True, True, True)

                # Actually draw everything
                self.main_fig.canvas.draw()
                self.main_fig.canvas.flush_events()
            # update_network(value_loss, self.optim_v)

        # Update learning rate
        self.scheduler.step()
        self.scheduler_critic.step()

    def run_train_loop(self):
        """
        Runs the basic PPG training loop
        """
        if self.plot:
            self.init_plots()

        obss = [None]*self.num_envs
        dones = [False]*self.num_envs
        states = [None]*self.num_envs

        for i in range(self.num_rollouts):

            if i % self.save_every == 0:
                state_dict = self.agent.policy.state_dict()
                th.save(state_dict, self.out_weights_path)

                data_path = f"data/{self.training_name}.csv"
                df = pd.DataFrame(
                    data={
                        "pi_loss": self.pi_loss_history,
                        "v_loss": self.v_loss_history,
                        "total_loss": self.total_loss_history,
                        "entropy": self.entropy_history,
                        "expl_var": self.expl_var_history
                    })
                df.to_csv(data_path, index=False)

                fig_path = f"data/{self.training_name}.png"
                self.main_fig.savefig(fig_path)
                print(f"💾 Saved checkpoint data")
                print(f"   - {self.out_weights_path}")
                print(f"   - {data_path}")
                print(f"   - {fig_path}")

            print(
                f"🎬 Starting {self.env_name} rollout {i + 1}/{self.num_rollouts}")

            # Do a server restart every 10 rollouts
            # Note: This is not one-to-one with the episodes
            # At T = 100, this is every 5 episodes
            should_hard_reset = i % 10 == 0 and i != 0

            obss_buffer = []
            dones_buffer = []
            states_buffer = []
            for env, next_obs, next_done, next_hidden_state in zip(self.envs, obss, dones, states):
                next_obs, next_done, next_hidden_state = self.rollout(
                    env, next_obs, next_done, next_hidden_state, hard_reset=should_hard_reset)
                obss_buffer.append(next_obs)
                dones_buffer.append(next_done)
                states_buffer.append(next_hidden_state)

            obss = obss_buffer
            dones = dones_buffer
            states = states_buffer

            self.learn_ppo_phase()

            

            for _ in range(self.sleep_cycles):
                # optimize Ljoint wrt policy weights

                # optimize Lvalue wrt value weights



            # clear memories after every rollout
            self.memories.clear()




if __name__ == "__main__":

    ppg = PhasicPolicyGradient(
        env_name="MineRLPunchCowEz-v0",
        model="foundation-model-1x",
        weights="foundation-model-1x",
        out_weights="cow-deleter-1x",
        save_every=5,
        num_envs=4,
        num_rollouts=500,
        num_steps=50,
        epochs=6,
        minibatch_size=48,
        lr=2.5e-5,
        weight_decay=0,
        betas=(0.9, 0.999),
        beta_s=0.2,
        eps_clip=0.2,
        value_clip=0.2,
        value_loss_weight=0.2,
        gamma=0.99,
        lam=0.95,
        mem_buffer_size=10000,
        plot=True,
    )

    ppg.run_train_loop()
