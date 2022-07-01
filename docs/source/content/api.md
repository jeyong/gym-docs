---
layout: "contents"
title: API
---

# API

## Environments 초기화
Gym에서 Environments 초기화는 다음과 같이 쉽게 가능:

```python
import gym
env = gym.make('CartPole-v0')
```

## Environments와 상호 작용
Gym은 전형적인 "agent-environment loop" 를 구현 :

```{image} /_static/diagrams/AE_loop.png
:width: 50%
:align: center
:class: only-light
```

```{image} /_static/diagrams/AE_loop_dark.png
:width: 50%
:align: center
:class: only-dark
```

agent는 해당 environment 내에서 어떤 action들을 수행한다.(보통은 control 입력을 environment로 전달하는 방식으로 예로는 모터의 torque 입력) 그리고 environment의 state가 어떻게 변경되는지를 관찰한다. 이러한 action-observation 교환을 *timestep*이라고 부른다.

RL에서 목표는 특정 방식으로 environment를 다루는 것이다. 예제로 agent가 robot을 공간에서 특정 지점으로 이동시키기를 원한다. 만약 이것이 성공하면 각 timestemp마다 observation에 따라서 positive reward를 받게 된다. agent가 성공하지 못했다면 reward는 negative이거나 0일 수도 있다. agent는 많은 timesteps 동안 누적 reward를 극대화 시키도록 train시킨다.

일부 timesteps 이후에 environment는 terminal 상태로 들어갈 수도 잇다. 예제로 robot은 파괴될 수도 있다. 이 경우 environment를 새로운 초기 상태로 리셋되기를 원한다. 이러한 terminal 상태로 들어가게 되면 environment은 done signal을 agent로 제공한다. "catastrophic failure"가 모든 done signal들을 발생시키지는 않는다.:
가끔 고정된 timesteps 이후에 done signal이 발생되기를 원할때도 있다. 혹은 agent가 environment내에서 어떤 task를 완료시키는 것에 성공한 경우일 수도 있다.

Gym 내에서 agent-environment loop은 어떤 형태인지 살펴보자.
이 예제는 `LunarLander-v2` environment의 인스턴스를 1000 timesteps 실행시키고 각 step에서 environment를 랜더링한다. environment를 랜더링하는 것을 윈도우가 뜨게 된다.

```python
import gym
env = gym.make("LunarLander-v2")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()

    if done:
        observation, info = env.reset(return_info=True)

env.close()
```

출력은 아래와 같은 형태로 나타난다.

```{figure} https://user-images.githubusercontent.com/15806078/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif
:width: 50%
:align: center
```

모든 environment는 `env.action_space` 속성을 제공하기 위해서 유효한 action 포맷을 지정한다. 유사하게 유효한 observation 포맷은 `env.observation_space`로 지정한다. 위에 예제에서 임의의 actions을 `env.action_space.sample()`를 통해서 샘플링했다. 중요한 점은 해당 environment에서 재생성이 가능한 샘플을 위해서 별도의 action space를 seed로 줄수 있어야 한다.

## 표준 방법

### Stepping
```{eval-rst}
.. autofunction:: gym.Env.step
```

### Resetting
```{eval-rst}
.. autofunction:: gym.Env.reset
```

### Rendering

```{eval-rst}
.. autofunction:: gym.Env.render
```

## 추가적인 Environment API

### Attributes

```{eval-rst}
.. autoattribute:: gym.Env.action_space

    이 attirubte는 유효한 action의 포맷을 준다. 이는 Gym이 제공하는 datatype `Space`이다. 예제로 만약 action space가 type `Discrete`이고 value `Discrete(2)`을 주면 2개의 유효한 discrete actions이 있다. : 0 & 1.

    .. code::
    
        >>> env.action_space
        Discrete(2)
        >>> env.observation_space
        Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
```

```{eval-rst}
.. autoattribute:: gym.Env.observation_space

    이 attribute는 유효한 observations의 포맷을 준다. Gym이 제공하는 datatype :class:`Space`이다. 예제로 observation space가 type :class:`Box`이고 object의 shape은 ``(4,)`이면 이는 유효한 observation은 4 numbers의 array가 된다. attributes로 box bounds를 검사할 수 있다.

    .. code::

        >>> env.observation_space.high
        array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=float32)
        >>> env.observation_space.low
        array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38], dtype=float32)
``` 

```{eval-rst}
.. autoattribute:: gym.Env.reward_range

    이 attribute는 min과 max reward에 대한 tuple이다. 기본 범위는 ``(-inf,+inf)``로 설정된다. 좀더 좁은 범위를 원한다면 그렇게 설정할 수 있다.
``` 

### Methods

```{eval-rst}
.. autofunction:: gym.Env.close
``` 
 
```{eval-rst}
.. autofunction:: gym.Env.seed
```  

## Checking API-Conformity
custom environmnet를 구현하고 API에 부합하는지를 확인하기 위해서 sanity 체크를 수행할 수 있다. 이렇게 실행 : 

```python
>>> from gym.utils.env_checker import check_env
>>> check_env(env)
```

이 함수는 여러분의 environment가 Gym API를 따르지 않는거 같이 보이면 exception을 throw한다. 여러분이 실수를 했거나 원칙을 따르지 않는 것으로 보인다면 warning을 생성시킨다.(e.g. `observation_space`가 image로 보이지만 올바른 dtype을 가지지 않는 경우) warning은 `warn=False`을 전달해서 끄기가 가능하다. 기본적으로 `check_env`은 `render` method를 검사하지 않는다. 이런 동작을 변경하고자 한다면 `skip_render_check=False`을 전달할 수 있다.

> environment에서 `check_env`를 실행한 후에, 검사한 instance를 재사용하지 말아야 한다.  이미 closed되어 있을 수 있다.

## Spaces
Spaces are usually used to specify the format of valid actions and observations.
Every environment should have the attributes `action_space` and `observation_space`, both of which should be instances
of classes that inherit from `Space`.
There are multiple `Space` types available in Gym:

- `Box`: describes an n-dimensional continuous space. It's a bounded space where we can define the upper and lower limits which describe the valid values our observations can take.
- `Discrete`: describes a discrete space where {0, 1, ..., n-1} are the possible values our observation or action can take. Values can be shifted to {a, a+1, ..., a+n-1} using an optional argument.
- `Dict`: represents a dictionary of simple spaces.
- `Tuple`: represents a tuple of simple spaces.
- `MultiBinary`: creates a n-shape binary space. Argument n can be a number or a `list` of numbers.
- `MultiDiscrete`: consists of a series of `Discrete` action spaces with a different number of actions in each element.

```python
>>> from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
>>> 
>>> observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
>>> observation_space.sample()
[ 1.6952509 -0.4399011 -0.7981693]
>>>
>>> observation_space = Discrete(4)
>>> observation_space.sample()
1
>>> 
>>> observation_space = Discrete(5, start=-2)
>>> observation_space.sample()
-2
>>> 
>>> observation_space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
>>> observation_space.sample()
OrderedDict([('position', 0), ('velocity', 1)])
>>>
>>> observation_space = Tuple((Discrete(2), Discrete(3)))
>>> observation_space.sample()
(1, 2)
>>>
>>> observation_space = MultiBinary(5)
>>> observation_space.sample()
[1 1 1 0 1]
>>>
>>> observation_space = MultiDiscrete([ 5, 2, 2 ])
>>> observation_space.sample()
[3 0 0]
 ```

## Wrappers
Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.
Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular. Wrappers can 
also be chained to combine their effects. Most environments that are generated via `gym.make` will already be wrapped by default.

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along
with (possibly optional) parameters to the wrapper's constructor:
```python
>>> import gym
>>> from gym.wrappers import RescaleAction
>>> base_env = gym.make("BipedalWalker-v3")
>>> base_env.action_space
Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)
>>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
>>> wrapped_env.action_space
Box([0. 0. 0. 0.], [1. 1. 1. 1.], (4,), float32)
```


There are three very common things you might want a wrapper to do:

- Transform actions before applying them to the base environment
- Transform observations that are returned by the base environment
- Transform rewards that are returned by the base environment

Such wrappers can be easily implemented by inheriting from `ActionWrapper`, `ObservationWrapper`, or `RewardWrapper` and implementing the
respective transformation.

However, sometimes you might need to implement a wrapper that does some more complicated modifications (e.g. modify the
reward based on data in `info`). Such wrappers
can be implemented by inheriting from `Wrapper`.
Gym already provides many commonly used wrappers for you. Some examples:

- `TimeLimit`: Issue a done signal if a maximum number of timesteps has been exceeded (or the base environment has issued a done signal).
- `ClipAction`: Clip the action such that it lies in the action space (of type `Box`).
- `RescaleAction`: Rescale actions to lie in a specified interval
- `TimeAwareObservation`: Add information about the index of timestep to observation. In some cases helpful to ensure that transitions are Markov.

If you have a wrapped environment, and you want to get the unwrapped environment underneath all of the layers of wrappers (so that you can manually call a function or change some underlying aspect of the environment), you can use the `.unwrapped` attribute. If the environment is already a base environment, the `.unwrapped` attribute will just return itself.

```python
>>> wrapped_env
<RescaleAction<TimeLimit<BipedalWalker<BipedalWalker-v3>>>>
>>> wrapped_env.unwrapped
<gym.envs.box2d.bipedal_walker.BipedalWalker object at 0x7f87d70712d0>
```

## Playing within an environment
You can also play the environment using your keyboard using the `play` function in `gym.utils.play`. 
```python
from gym.utils.play import play
play(gym.make('Pong-v0'))
```
This opens a window of the environment and allows you to control the agent using your keyboard.

Playing using the keyboard requires a key-action map. This map should have type `dict[tuple[int], int | None]`, which maps the keys pressed to action performed.
For example, if pressing the keys `w` and `space` at the same time is supposed to perform action `2`, then the `key_to_action` dict should look like:
```python
{
    # ...
    (ord('w'), ord(' ')): 2,
    # ...
}
```
As a more complete example, let's say we wish to play with `CartPole-v0` using our left and right arrow keys. The code would be as follows:
```python
import gym
import pygame
from gym.utils.play import play
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
play(gym.make("CartPole-v0"), keys_to_action=mapping)
```
where we obtain the corresponding key ID constants from pygame. If the `key_to_action` argument is not specified, then the default `key_to_action` mapping for that env is used, if provided.

Furthermore, if you wish to plot real time statistics as you play, you can use `gym.utils.play.PlayPlot`. Here's some sample code for plotting the reward for last 5 second of gameplay:
```python
def callback(obs_t, obs_tp1, action, rew, done, info):
    return [rew,]
plotter = PlayPlot(callback, 30 * 5, ["reward"])
env = gym.make("Pong-v0")
play(env, callback=plotter.callback)
```
