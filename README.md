[![Build Status](https://travis-ci.org/AdamStelmaszczyk/rl-tutorial.svg?branch=master)](https://travis-ci.org/AdamStelmaszczyk/rl-tutorial)

Solving Mountain Car environment with TensorFlow & Keras implementation of DQN.  
For similar code solving some Atari games, look [here](https://github.com/AdamStelmaszczyk/dqn).

---

<img src="https://github.com/AdamStelmaszczyk/rl-tutorial/blob/master/images/tensorboard.png">

## Install

1. Clone this repo: `git clone https://github.com/AdamStelmaszczyk/rl-tutorial.git`.
2. [Install `conda`](https://conda.io/docs/user-guide/install/index.html) for dependency management.
3. Create `tutorial` conda environment: `conda create -n tutorial python=3.6.5 -y`.
4. Activate `tutorial` conda environment: `source activate tutorial`. All the following commands should be run in the activated `tutorial` environment.
5. Install basic dependencies: `pip install -r requirements.txt`.

There is an automatic build on Travis which [does the same](https://github.com/AdamStelmaszczyk/rl-tutorial/blob/master/.travis.yml).

## Run

`python run.py --help`

```
usage: run.py [-h] [--eval] [--model MODEL] [--name NAME] [--seed SEED]
              [--test] [--view]

optional arguments:
  -h, --help     show this help message and exit
  --eval         run evaluation with log only (default: False)
  --images       save images during evaluation (default: False)
  --model MODEL  model filename to load (default: None)
  --name NAME    name for saved files (default: 06-08-21-53)
  --seed SEED    pseudo random number generator seed (default: None)
  --test         run tests (default: False)
  --view         view the model playing the game (default: False)
```

## Generate GIFs

<img src="https://github.com/AdamStelmaszczyk/rl-tutorial/blob/master/images/random.gif">

1. Generate images: `python run.py --model 06-08-18-42-log/06-08-18-42-200000.h5 --images`.
2. We will use `convert` tool, which is part of ImageMagick, [here](https://www.imagemagick.org/script/download.php) are the installation instructions.
3. Convert images from episode 1 to GIF: `convert -layers optimize-frame 1_*.png 1.gif`

<img src="https://github.com/AdamStelmaszczyk/rl-tutorial/blob/master/images/good.gif">

## Uninstall

1. Deactivate conda environment: `source deactivate`.
2. Remove `tutorial` conda environment: `conda env remove -n tutorial -y`.

## Links

- Playing Atari with Deep Reinforcement Learning, https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- Human-level control through deep reinforcement learning, https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
- MountainCar description: https://github.com/openai/gym/wiki/MountainCar-v0
- MountainCar source code: https://github.com/openai/gym/blob/4c460ba6c8959dd8e0a03b13a1ca817da6d4074f/gym/envs/classic_control/mountain_car.py
