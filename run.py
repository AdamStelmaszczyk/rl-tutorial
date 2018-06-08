import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import strftime, time

import cv2
import gym
import numpy as np
import psutil
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tqdm import tqdm

from loggers import TensorBoardLogger
from replay_buffer import ReplayBuffer

DISCOUNT_FACTOR_GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TARGET_UPDATE_EVERY = 1000
TRAIN_START = 2000
REPLAY_BUFFER_SIZE = 50000
MAX_STEPS = 200000
LOG_EVERY = 2000
SNAPSHOT_EVERY = 50000
EVAL_EVERY = 20000
EVAL_STEPS = 10000
EVAL_EPSILON = 0
TRAIN_EPSILON = 0.01
Q_VALIDATION_SIZE = 10000


def one_hot_encode(n, action):
    one_hot = np.zeros(n)
    one_hot[int(action)] = 1
    return one_hot


def predict(env, model, observations):
    action_mask = np.ones((len(observations), env.action_space.n))
    return model.predict(x=[observations, action_mask])


def fit_batch(env, model, target_model, batch):
    observations, actions, rewards, next_observations, dones = batch
    # Predict the Q values of the next states. Passing ones as the action mask.
    next_q_values = predict(env, target_model, next_observations)
    # The Q values of terminal states is 0 by definition.
    next_q_values[dones] = 0.0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    q_values = rewards + DISCOUNT_FACTOR_GAMMA * np.max(next_q_values, axis=1)
    one_hot_actions = np.array([one_hot_encode(env.action_space.n, action) for action in actions])
    history = model.fit(
        x=[observations, one_hot_actions],
        y=one_hot_actions * q_values[:, None],
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    return history.history['loss'][0]


def create_model(env):
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    observations_input = keras.layers.Input(obs_shape, name='observations_input')
    action_mask = keras.layers.Input((n_actions,), name='action_mask')
    hidden = keras.layers.Dense(32, activation='relu')(observations_input)
    hidden_2 = keras.layers.Dense(32, activation='relu')(hidden)
    output = keras.layers.Dense(n_actions)(hidden_2)
    filtered_output = keras.layers.multiply([output, action_mask])
    model = keras.models.Model([observations_input, action_mask], filtered_output)
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer, loss='mean_squared_error')
    return model


def greedy_action(env, model, observation):
    next_q_values = predict(env, model, observations=[observation])
    return np.argmax(next_q_values)


def epsilon_greedy_action(env, model, observation, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_action(env, model, observation)
    return action


def save_model(model, step, logdir, name):
    filename = '{}/{}-{}.h5'.format(logdir, name, step)
    model.save(filename)
    print('Saved {}'.format(filename))
    return filename


def save_image(env, episode, step):
    frame = env.render(mode='rgb_array')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # following cv2.imwrite assumes BGR
    filename = "{}_{:06d}.png".format(episode, step)
    cv2.imwrite(filename, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])


def evaluate(env, model, view=False, images=False):
    print("Evaluation")
    done = True
    episode = 0
    episode_return_sum = 0.0
    for step in tqdm(range(1, EVAL_STEPS + 1)):
        if done:
            if episode > 0:
                episode_return_sum += episode_return
            obs = env.reset()
            episode += 1
            episode_return = 0.0
            episode_steps = 0
            if view:
                env.render()
            if images:
                save_image(env, episode, step)
        else:
            obs = next_obs
        action = epsilon_greedy_action(env, model, obs, epsilon=EVAL_EPSILON)
        next_obs, reward, done, _ = env.step(action)
        episode_return += reward
        episode_steps += 1
        if view:
            env.render()
        if images:
            save_image(env, episode, step)
    assert episode > 0
    episode_return_avg = episode_return_sum / episode
    return episode_return_avg


def train(env, model, max_steps, name, logdir, logger):
    target_model = create_model(env)
    replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
    done = True
    episode = 0
    steps_after_logging = 0
    loss = 0.0
    for step in range(1, max_steps + 1):
        try:
            if step % SNAPSHOT_EVERY == 0:
                save_model(model, step, logdir, name)
            if done:
                if episode > 0:
                    if steps_after_logging >= LOG_EVERY:
                        steps_after_logging = 0
                        episode_end = time()
                        episode_seconds = episode_end - episode_start
                        episode_steps = step - episode_start_step
                        steps_per_second = episode_steps / episode_seconds
                        memory = psutil.virtual_memory()
                        to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
                        print(
                            "episode {} "
                            "steps {}/{} "
                            "loss {:.7f} "
                            "return {} "
                            "in {:.2f}s "
                            "{:.1f} steps/s "
                            "{:.1f}/{:.1f} GB RAM".format(
                                episode,
                                episode_steps,
                                step,
                                loss,
                                episode_return,
                                episode_seconds,
                                steps_per_second,
                                to_gb(memory.used),
                                to_gb(memory.total),
                            ))
                        logger.log_scalar('episode_return', episode_return, step)
                        logger.log_scalar('episode_steps', episode_steps, step)
                        logger.log_scalar('episode_seconds', episode_seconds, step)
                        logger.log_scalar('steps_per_second', steps_per_second, step)
                        logger.log_scalar('memory_used', to_gb(memory.used), step)
                        logger.log_scalar('loss', loss, step)
                episode_start = time()
                episode_start_step = step
                obs = env.reset()
                episode += 1
                episode_return = 0.0
            else:
                obs = next_obs

            action = epsilon_greedy_action(env, model, obs, epsilon=TRAIN_EPSILON)
            next_obs, reward, done, _ = env.step(action)
            episode_return += reward
            replay.add(obs, action, reward, next_obs, done)

            if step >= TRAIN_START:
                if step % TARGET_UPDATE_EVERY == 0:
                    target_model.set_weights(model.get_weights())
                batch = replay.sample(BATCH_SIZE)
                loss = fit_batch(env, model, target_model, batch)
            if step == Q_VALIDATION_SIZE:
                q_validation_observations, _, _, _, _ = replay.sample(Q_VALIDATION_SIZE)
            if step >= TRAIN_START and step % EVAL_EVERY == 0:
                episode_return_avg = evaluate(env, model)
                q_values = predict(env, model, q_validation_observations)
                max_q_values = np.max(q_values, axis=1)
                avg_max_q_value = np.mean(max_q_values)
                print(
                    "episode {} "
                    "step {} "
                    "episode_return_avg {:.3f} "
                    "avg_max_q_value {:.3f}".format(
                        episode,
                        step,
                        episode_return_avg,
                        avg_max_q_value,
                    ))
                logger.log_scalar('episode_return_avg', episode_return_avg, step)
                logger.log_scalar('avg_max_q_value', avg_max_q_value, step)
            steps_after_logging += 1
        except KeyboardInterrupt:
            save_model(model, step, logdir, name)
            break


def load_or_create_model(env, model_filename):
    if model_filename:
        model = keras.models.load_model(model_filename)
        print('Loaded {}'.format(model_filename))
    else:
        model = create_model(env)
    model.summary()
    return model


def set_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)


def main(args):
    assert BATCH_SIZE <= TRAIN_START <= Q_VALIDATION_SIZE <= REPLAY_BUFFER_SIZE
    print('args', args)
    env = gym.make('MountainCar-v0')
    set_seed(env, args.seed)
    model = load_or_create_model(env, args.model)
    if args.view or args.eval or args.images:
        episode_return_avg = evaluate(env, model, args.view, args.images)
        print("episode_return_avg {:.3f}".format(episode_return_avg))
    else:
        max_steps = 100 if args.test else MAX_STEPS
        logdir = '{}-log'.format(args.name)
        logger = TensorBoardLogger(logdir)
        print('Created {}'.format(logdir))
        train(env, model, max_steps, args.name, logdir, logger)
        if args.test:
            filename = save_model(model, EVAL_STEPS, logdir='.', name='test')
            load_or_create_model(env, filename)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eval', action='store_true', default=False, help='run evaluation with log only')
    parser.add_argument('--images', action='store_true', default=False, help='save images during evaluation')
    parser.add_argument('--model', action='store', default=None, help='model filename to load')
    parser.add_argument('--name', action='store', default=strftime("%m-%d-%H-%M"), help='name for saved files')
    parser.add_argument('--seed', action='store', type=int, help='pseudo random number generator seed')
    parser.add_argument('--test', action='store_true', default=False, help='run tests')
    parser.add_argument('--view', action='store_true', default=False, help='view the model playing the game')
    main(parser.parse_args())
