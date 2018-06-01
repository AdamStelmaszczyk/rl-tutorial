[![Build Status](https://travis-ci.org/AdamStelmaszczyk/rl-tutorial.svg?branch=master)](https://travis-ci.org/AdamStelmaszczyk/rl-tutorial)


TensorFlow & Keras implementation of DQN.

## Install

1. Clone this repo: `git clone https://github.com/AdamStelmaszczyk/rl-tutorial.git`.
2. [Install `conda`](https://conda.io/docs/user-guide/install/index.html) for dependency management.
3. Create `tutorial` conda environment: `conda create -n tutorial python=3.6.5 -y`.
4. Activate `tutorial` conda environment: `source activate tutorial`. All the following commands should be run in the activated `tutorial` environment.
5. Install basic dependencies: `pip install -r requirements.txt`.

There is an automatic build on Travis which [does the same](https://github.com/AdamStelmaszczyk/rl-tutorial/blob/master/.travis.yml).

## Uninstall

1. Deactivate conda environment: `source deactivate`.
2. Remove `tutorial` conda environment: `conda env remove -n tutorial -y`.

## Links

- MountainCar description: https://github.com/openai/gym/wiki/MountainCar-v0
- MountainCar source code: https://github.com/openai/gym/blob/4c460ba6c8959dd8e0a03b13a1ca817da6d4074f/gym/envs/classic_control/mountain_car.py