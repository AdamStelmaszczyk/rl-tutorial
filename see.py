import gym

env = gym.make('MountainCar-v0')
done = True
episode = 0
episode_return = 0.0
for episode in range(5):
    for step in range(200):
        if done:
            if episode > 0:
                print("Episode return: ", episode_return)
            obs = env.reset()
            episode += 1
            episode_return = 0.0
            env.render()
        else:
            obs = next_obs
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        episode_return += reward
        env.render()
