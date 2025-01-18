import logging
import coloredlogs
import sys
import pickle
import psutil
import gym
import minerl
import os
sys.path.insert(0, "vpt")
from vpt.agent import MineRLAgent
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3 import PPO
import numpy as np
from gym.spaces import MultiDiscrete
from vpt.lib.policy import MinecraftPolicy, MinecraftAgentPolicy
coloredlogs.install(logging.DEBUG)
coloredlogs.install()

# copied from example in SB3 documentation
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

MINERL_GYM_ENV = 'MineRLObtainDiamondShovel-v0'
MODEL = 'foundation-model-3x.model'
WEIGHTS = 'foundation-model-3x.weights'

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

with open("models/foundation-model-3x.model", "rb") as f:
    agent_parameters = pickle.load(f)

policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

print(policy_kwargs)

# Create the base environment for validation
base_env = gym.make(MINERL_GYM_ENV)

# Create vectorized environment with custom wrapper
vec_env = make_vec_env(
    env_id=MINERL_GYM_ENV,
    n_envs=2,
    seed=42,
)

# Initialize the MineRLAgent, passing the base environment for validation
agent = MineRLAgent(base_env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)

# Filter invalid keys from policy_kwargs
# valid_keys = ["net_arch", "activation_fn", "ortho_init", "log_std_init", "optimizer_class", "optimizer_kwargs"]
# policy_kwargs = {key: value for key, value in policy_kwargs.items() if key in valid_keys}

agent = PPO(
    MinecraftPolicy,  # Assuming a CNN-based policy for MineRL observations
    env=base_env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log=log_dir  # Logs for TensorBoard
)
agent.load_weights(WEIGHTS)

# model.set_parameters(WEIGHTS)

print("wweee start twaining nowww")

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
TIMESTEPS = 100000
agent.learn(total_timesteps=TIMESTEPS)



# Save the trained model
agent.save("ppo_minerl")
print("Model saved as ppo_minerl.zip.")




### IGNORE ###
# def main():
#     env = gym.make(MINERL_GYM_ENV)

#     # Load the model
#     agent_parameters = pickle.load(open(MODEL, "rb"))
#     policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
#     pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
#     pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
#     agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
#     agent.load_weights(WEIGHTS)

#     for i in range(10):
#         obs = env.reset()
#         agent.reset()
#         # for step_counter in range(EVAL_MAX_STEPS):
#         for step_counter in range(10000):

#             # Step your model here.
#             minerl_action = agent.get_action(obs)

#             obs, reward, done, info = env.step(minerl_action)

#             # Uncomment the line below to see the agent in action:
#             env.render()

#             if done:
#                 break
#         print(f"[{i}] Episode complete")

#     # Close environment and clean up any bigger memory hogs.
#     # Otherwise, you might start running into memory issues
#     # on the evaluation server.
#     env.close()


# if __name__ == "__main__":
#     main()
