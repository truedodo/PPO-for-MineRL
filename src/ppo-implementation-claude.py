import pickle
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import gym
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, "vpt")
from agent import MineRLAgent
from lib.tree_util import tree_map

# Configuration for training
@dataclass
class PPOConfig:
    env_name: str
    num_envs: int
    num_steps: int
    num_epochs: int
    num_rollouts: int
    minibatch_size: int
    learning_rate: float
    weight_decay: float
    betas: Tuple[float, float]
    entropy_coef: float
    kl_coef: float
    clip_range: float
    value_clip_range: float
    value_loss_weight: float
    gamma: float
    lambda_gae: float
    checkpoint_frequency: int
    device: th.device = th.device("mps")

@dataclass
class Experience:
    """Single timestep of experience collected during rollouts"""
    obs: Dict[str, th.Tensor]
    hidden_state: List
    action: th.Tensor
    action_log_prob: th.Tensor
    value: th.Tensor
    reward: float
    done: bool
    policy_latent: th.Tensor
    value_latent: th.Tensor

class ExperienceDataset(Dataset):
    """Dataset for storing and sampling experience"""
    def __init__(self, experiences: List[Experience]):
        self.experiences = experiences
        
    def __len__(self) -> int:
        return len(self.experiences)
        
    def __getitem__(self, idx: int) -> Experience:
        return self.experiences[idx]

def compute_gae(
    rewards: List[float],
    values: List[th.Tensor],
    dones: List[bool],
    gamma: float,
    lambda_: float
) -> np.ndarray:
    """Compute Generalized Advantage Estimation"""
    gae = 0
    returns = []
    
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[step + 1].item()
            
        if step == len(rewards) - 1:
            next_non_terminal = 1.0
        else:
            next_non_terminal = 1.0 - float(dones[step])
            
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step].item()
        gae = delta + gamma * lambda_ * next_non_terminal * gae
        returns.insert(0, gae)
        
    return np.array(returns)

class PPO:
    def __init__(
        self,
        config: PPOConfig,
        model_path: str,
        weights_path: str,
        output_weights_path: str
    ):
        self.config = config
        self.envs = [gym.make(config.env_name) for _ in range(config.num_envs)]
        
        # Load agent parameters and create agents
        agent_params = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_params["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_params["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        
        # Create main and reference agents
        self.agent = MineRLAgent(self.envs[0], policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
        self.ref_agent = MineRLAgent(self.envs[0], policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
        
        self.agent.load_weights(weights_path)
        self.ref_agent.load_weights(weights_path)
        
        # Setup optimizer
        trainable_params = (list(self.agent.policy.pi_head.parameters()) + list(self.agent.policy.value_head.parameters()))
        
        self.optimizer = th.optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda epoch: 1 - epoch / config.num_rollouts
        )
        
        self.writer = SummaryWriter()
        self.train_step = 0
        self.output_weights_path = output_weights_path
        
    def collect_rollout(
        self,
        env: gym.Env,
        initial_obs: Optional[Dict] = None,
        initial_done: bool = False,
        initial_hidden_state: Optional[List] = None
    ) -> Tuple[List[Experience], Dict, bool, List]:
        """Collect a single rollout of experience"""
        experiences = []
        
        # Initialize environment state
        if initial_obs is None:
            initial_hidden_state = self.agent.policy.initial_state(1)
            initial_obs = env.reset()
            initial_done = False
            
        obs = initial_obs
        done = initial_done
        hidden_state = initial_hidden_state
        
        # Create first mask for LSTM
        dummy_first = th.from_numpy(np.array((False,))).to(self.config.device).unsqueeze(1)
        
        for _ in range(self.config.num_steps):
            # Reset if episode ended
            if done:
                obs = env.reset()
                hidden_state = self.agent.policy.initial_state(1)
                
            # Process observation
            agent_obs = self.agent._env_obs_to_agent(obs)
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
            
            # Get action and value
            with th.no_grad():
                (pi_latent, v_latent), next_hidden = self.agent.policy.net(
                    agent_obs, hidden_state, context={"first": dummy_first}
                )
                pi_dist = self.agent.policy.pi_head(pi_latent)
                value = self.agent.policy.value_head(v_latent)
                
            action = self.agent.policy.pi_head.sample(pi_dist, deterministic=False)
            action_log_prob = self.agent.policy.get_logprob_of_action(pi_dist, action)
            
            # Take environment step
            minerl_action = self.agent._agent_action_to_env(action)
            next_obs, reward, next_done, _ = env.step(minerl_action)
            
            # Store experience
            experiences.append(Experience(
                obs=agent_obs,
                hidden_state=hidden_state,
                action=action,
                action_log_prob=action_log_prob,
                value=value,
                reward=reward,
                done=next_done,
                policy_latent=pi_latent,
                value_latent=v_latent
            ))
            
            # Update state
            obs = next_obs
            done = next_done
            hidden_state = next_hidden
            
        return experiences, obs, done, hidden_state

    def train_epoch(self, experiences: List[Experience]):
        """Run a single training epoch"""
        dataset = ExperienceDataset(experiences)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.minibatch_size,
            shuffle=True
        )
        
        for batch in dataloader:
            # Unpack batch
            obs = batch.obs
            hidden_states = batch.hidden_state
            actions = batch.action
            old_log_probs = batch.action_log_prob
            old_values = batch.value
            rewards = batch.reward
            dones = batch.done
            policy_latents = batch.policy_latent
            value_latents = batch.value_latent
            
            # Get current policy and value predictions
            pi_dist = self.agent.policy.pi_head(policy_latents)
            values = self.agent.policy.value_head(value_latents)
            
            # Get log probs and entropy
            log_probs = self.agent.policy.get_logprob_of_action(pi_dist, actions)
            entropy = self.agent.policy.pi_head.entropy(pi_dist)
            
            # Get KL divergence from reference policy
            with th.no_grad():
                ref_dist = self.ref_agent.policy.pi_head(policy_latents)
            kl_div = self.agent.policy.pi_head.kl_divergence(pi_dist, ref_dist)
            
            # Compute advantages
            advantages = compute_gae(
                rewards,
                values.detach(),
                dones,
                self.config.gamma,
                self.config.lambda_gae
            )
            advantages = th.tensor(advantages, device=self.config.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute policy loss
            ratio = th.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -th.min(surr1, surr2).mean()
            
            # Add entropy and KL terms
            policy_loss = (
                policy_loss 
                - self.config.entropy_coef * entropy.mean()
                + self.config.kl_coef * kl_div.mean()
            )
            
            # Compute value loss
            value_pred = values
            value_target = rewards + self.config.gamma * (1 - dones.float()) * values[1:].detach()
            value_loss = (value_pred - value_target).pow(2).mean()
            
            # Total loss
            loss = policy_loss + self.config.value_loss_weight * value_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log metrics
            self.writer.add_scalar("loss/policy", policy_loss.item(), self.train_step)
            self.writer.add_scalar("loss/value", value_loss.item(), self.train_step)
            self.writer.add_scalar("loss/total", loss.item(), self.train_step)
            self.writer.add_scalar("policy/entropy", entropy.mean().item(), self.train_step)
            self.writer.add_scalar("policy/kl", kl_div.mean().item(), self.train_step)
            
            self.train_step += 1

    def train(self):
        """Main training loop"""
        # Initialize environment states
        obs = [None] * self.config.num_envs
        dones = [False] * self.config.num_envs
        hidden_states = [None] * self.config.num_envs
        
        for rollout in range(self.config.num_rollouts):
            # Save checkpoint
            if rollout > 0 and rollout % self.config.checkpoint_frequency == 0:
                th.save(self.agent.policy.state_dict(), self.output_weights_path)
                print(f"Saved checkpoint to {self.output_weights_path}")
            
            # Collect experience from all environments
            all_experiences = []
            for i, env in enumerate(self.envs):
                experiences, next_obs, next_done, next_hidden = self.collect_rollout(
                    env, obs[i], dones[i], hidden_states[i]
                )
                all_experiences.extend(experiences)
                obs[i] = next_obs
                dones[i] = next_done
                hidden_states[i] = next_hidden
            
            # Train on collected experience
            for _ in range(self.config.num_epochs):
                self.train_epoch(all_experiences)
            
            # Update learning rate
            self.scheduler.step()

def main():
    config = PPOConfig(
        env_name="MineRLObtainDiamondShovel-v0",
        num_envs=4,
        num_steps=50,
        num_epochs=4,
        num_rollouts=500,
        minibatch_size=48,
        learning_rate=2.5e-5,
        weight_decay=0.04,
        betas=(0.9, 0.999),
        entropy_coef=0.2,
        kl_coef=0.2,
        clip_range=0.2,
        value_clip_range=0.2,
        value_loss_weight=0.2,
        gamma=0.99,
        lambda_gae=0.95,
        checkpoint_frequency=5
    )
    
    ppo = PPO(
        config=config,
        model_path="models/foundation-model-3x.model",
        weights_path="weights/foundation-model-3x.weights",
        output_weights_path="weights/ppo-zombie-hunter-1x.weights"
    )
    
    ppo.train()

if __name__ == "__main__":
    main()
