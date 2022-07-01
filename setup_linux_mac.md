# Ubuntu
* swig
```bash
$ sudo apt-get install swig
```
# Mac
* swig
```bash
$ brew install swig
```
* gym instlal
```bash
$ pip install 'gym[all]'
$ pip install 'gym[box2d]'
```

# test gym
```python
import gym
env = gym.make('CartPole-v1')

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
```

```python

```