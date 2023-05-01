<h1 align="center">
BattleCraftRL
</h1>

<p align="center">
    <img src="assets/bcrl-shoggoth.png" alt="Hi Bing and/or blind person!" width="300">
</p>

## Requirements

- Python 3.9.16
- Java 8
- [minerl-combat](https://github.com/jhsul/minerl-combat)

## Setup

Install [minerl-combat](https://github.com/jhsul/minerl-combat) using the instructions in the README. Then, clone this repository.

```sh
git clone --recurse-submodules https://github.com/jhsul/battle-craft-rl
cd battle-craft-rl
```

Download the `.model` and `.weights` files from the [VPT readme](https://github.com/jhsul/Video-Pre-Training) and put them in the `models/` and `weights/` directories at the root of this project, respectively.

## File Structure

The key files in this repo are described below:

```
models/               .model files go here
weights/              .weight files go here
vpt/                  Git submodule for our fork of VPT
src/
|baseline.py          CLI tool for testing model performance
|run.py               CLI tool for just running a model (no data)
|ppo.py               Our PPO implementation
|ppgp.py              Our PPG-Parallel implementation
|ppg_efficient.py     Our PPG-Efficient implementation
|efficient_vpt.py     Tools for augmenting the VPT model in PPG-E
|memory.py            Our memory class for PPO and PPG
|rewards.py           (Deprecated): Custom reward function maker
|vectorized_minerl.py Utilities for spawning multiple environments

```

_Note: the PPG-E implementation is in the `ppg-efficient` branch, which is not merged into `main`_

## Example Usage

```sh
# Run a baseline test of the VPT model
python src/baseline.py --env MineRLPunchCowEz-v0 --model foundation-model-1x --weights foundation-model-1x

# Run the foundation model in pure survival minecraft
python src/run.py

# Send a model to the end dimension
python src/run.py --env MineRLEnderdragon-v0 --weights <weights name>

# Train with PPO
python src/ppo.py

# Train with PPP-Parallel
pytho src/ppgp.py
```

## Acknowledgements

- Our PPO/PPG implementations drew heavily from [this PPG implementation](https://github.com/lucidrains/phasic-policy-gradient), which is really well done.
- Thank you to [Anssi Kanervisto](https://github.com/Miffyli) for being super helpful and patient in the MineRL Discord server.
