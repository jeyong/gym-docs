---
layout: "contents"
title: Environment Creation
---
# Environment 생성

이 문서는 Gym내부에서 새로운 environment를 생성하기 위해서 새 환경 생성과 관련된 유용한 wrappers, 유틸리티, 테스트들에 대해서 알아본다.
여기에 있는 코드로 실행하기 위해서 gym-examples를 clone할 수 있다. 가상환경을 사용하는 것을 추천한다.:

```console
git clone https://github.com/Farama-Foundation/gym-examples
cd gym-examples
python -m venv .env
source .env/bin/activate
pip install -e .
```

## Subclassing gym.Env

나만의 environment를 생성하는 방법을 배우기 전에 [Gym's API 문서](https://www.gymlibrary.ml/content/api/)를 먼저 확인해야 한다.

아래와 같은 gym-examples의 하부를 살펴보자.:

```sh
gym-examples/
  README.md
  setup.py
  gym_examples/
    __init__.py
    envs/
      __init__.py
      grid_world.py
    wrappers/
      __init__.py
      relative_position.py
 ```

`gym.Env`의 하부를 보여주기 위해서 `GridWorldEnv`라는 아주 간단한 game을 구현해 볼 것이다.
`gym-examples/gym_examples/envs/grid_world.py` 내에 있는 커스텀 environment에 대한 코드를 작성해 보자.
environment는 고정 크기의 2차원 사각 그리드로 구성되어 있다.(생성자에서 `size`를 지정)
agent는 각 timestep에서 그리드 셀 사이에서 수평이나 수직으로 이동 가능하다. agent의 목표는 episode 시작 시점에 그리드 상에 임의의 위치에 놓여진 물체를 목표지점으로 이동시키는 것이다.

- observations은 target과 agent의 위치를 제공한다.
- 우리 environment 내에 4가지 action이 있다. "right", "up", "left", "down"
- done signal은 target이 위치한 그리드 셀로 agent가 이동하게 되면 발생
- rewards는 binary(0 or 1)와 sparse로 agent가 target에 도달하지 못하면 reward는 항상 0이고 도달하면 1이다.

이 environment에서 episode(`size=5`)는 아래와 같다:

<img src="https://user-images.githubusercontent.com/15806078/160155148-253a05ae-25c1-4fcf-9a72-f72362a64225.gif" width="35%">

파란점은 agent이고 빨간 사각형은 target이다.


`GridWorldEnv` 코드를 한줄씩 살펴보자.: 

### 선언과 초기화(Declaration and Initialization)
커스텀 environment는 추상 class `gym.Env`으로부터 상속받는다. `metadata` attribute를 여러분의 class에 추가하는 것을 잊지말자.
여러분의 environment에서 지원하는 render-modes를 지정해야한다.(e.g. `"human"`, `"rgb_array"`, `"ansi"`) 그리고 framerate는 environment가 랜더링하는 속도이다.
`GridWorldEnv` 에서는 "rgb_array"와 "human" 모드와 4 FPS로 랜더링된다.

우리 environment의 `__init__` method는 integer `size`를 받아서 사각 grid의 크기를 결정한다.
랜더링을 위한 일부 변수를 설정하고 `self.observation_space` 와 `self.action_space`를 정의한다.
우리의 경우에 observations는 2차원 grid에서 agent와 target의 location에 대한 정보를 제공한다.
observations를 표현하는데 `"agent"` 와 `"target"` keys를 가지는 dictionaries의 형태를 선택한다. observation은 ` {"agent": array([1, 0]), "target": array([0, 3])}`와 같은 형태다.
우리 environment에서 4 actions를("right", "up", "left", "down") 가지므로 action space로 `Discrete(4)`를 사용한다.
여기서는 `GridWorldEnv` 선언과 `__init__`의 구현을 보자.:
```python
import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
```

### Environment States로부터 Observations 구성하기(Constructing Observations From Environment States)

`reset`과 `step` 모두에서 observations를 계산하기 때문에 `_get_obs` method를 갖고 있으면 편리하다. 이 method는 environment의 state를 observation으로 변환한다.
하지만 이는 강제사항은 아니며 `reset`과 `step`에서 각각 observations를 계산할 수도 있다.:
```python
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
```
`step`과 `reset`에서 반환하는 추가 정보를 위해서 유사한 method를 구현할 수도 있다.
우리의 경우 agent와 target 사이의 manhattan distance를 제공한다.:
```python
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
```
자주 info는 `step` method 내부에서만 유효한 일부 데이터를 포함한다.(e.g. individual reward terms) 이 경우 `step`에서 `get_info`가 반환하는 dictionary를 업데이트해야만 한다.

### Reset
`reset` method는 새로운 episode를 초기화하기 위해서 호출된다. `step` method는 `reset`이 호출되기 전에 호출되지는 않는다고 가정한다. 더우기 `reset`은 done signal이 발생할때마다 호출되어야만 한다.
사용자는 `seed` 키워드를 `reset`으로 전달하여 environment가 사용하는 랜덤 상수 생성기를 초기화한다. environment가 제공하는 `gym.Env`의 `self.np_random` 랜덤 상수 생성기를 사용하는 것을 추천한다. 만약에 RNG만 사용한다면 seeding에 대해서 크게 거정할 필요가 없다. 하지만  `gym.Env`가 제대로 RNG를 seeding하려면 `super().reset(seed=seed)`*를 호출해야한다는 것을 명심하자.
일단 이게 완료되면 environment의 state를 임의로 설정할 수 있다.
이 경우 agent의 location과 샘플 target 위치를 서로 겹치지 않게 임의로 선택할 수 있다.

`reset` method는 초기 state의 observation을 리턴하거나 초기 observation의 tuple과 일부 추가 정보를 리턴해야만 한다. 이는 `return_info`가 `True`인가에 의존하낟. 일찍이 구현한 `get_obs`와 `get_info` method를 사용할 수 있다.:

```python
    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2)

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation
```

### Step
`step` method는 보통 environment의 로직 대부분을 포함하고 있다. `action`을 받아서 action이 적용된 후에 environment의 state를 계산하고 4-tuple인 `(observation, reward, done, info)`를 반환한다.
일단 새로운 environment의 state가 계산되면 이 state가 종료 state인지 검사해서 맞으면 `done`으로 설정한다. `GridWorldEnv`내에서는 sparse binary rewards를 사용하므로 일단 `done`이 되면 `reward` 계산은 의미가 없다. `observation`과 `info`를 모으기 위해서 `_get_obs`와 `_get_info`를 다시 사용할 수 있다.:

```python
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done if the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if done else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info
```

### Rendering
여기서 랜더링을 위해서 PyGame을 사용한다. 랜더링에 유사한 접근법은 다양한 environments에서 사용된다. 이런 environments는 Gym에 포함되어 있고 여러분의 environments을 위한 기본 뼈대로 사용할 수 있다.:

```python
    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
```

### Close
`close` method는 environment가 사용했던 open resources를 close해야만 한다. 많은 경우에서 이 method를 구현하는데 귀찮을 필요가 없다. 하지만 우리 예제에서는 `render`는 `mode="human"`과 함께 호출되고 열린 창을 닫는데 필요하다.:

```python
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
```

다른 environments에서 `close`는 open되어 있는 리소스를 close할 수 있다. `close`를 호출한 후에 environment와 상호작용하지 않는다.


## Registering Envs

Gym이 검출할 수 있는 커스텀 environment를 위해서 다음과 같이 등록을 해야만 한다.
`gym-examples/gym_examples/__init__.py` 내에 이 코드를 넣기 위해서 선택한다. 

```python
from gym.envs.registration import register

register(
    id='gym_examples/GridWorld-v0',
    entry_point='gym_examples.envs:GridWorldEnv',
    max_episode_steps=300,
)
```
environment ID는 3개 컴포넌트로 구성되어 있고 이중 2개는 옵션이다: 옵션인 namespace(여기서는: `gym_examples`), 필수 이름(여기서는: `GridWorld`), 옵션이지만 추천하는 version(여기서는: v0).
`GridWorld-v0`(추천하는 방식)나 `GridWorld` 혹은 `gym_examples/GridWorld`로 등록할 수 있다. 그리고 적절한 ID는 environment 생성 동안에 사용된다. 절삭과 종료를 구분하기 위해서 `info["TimeLimit.truncated"]`를 검사한다.

`id`와 `entrypoint`를 분리하여 다음과 같은 추가 키워드 인자를 `register`에 전달할 수 있다.:

| Name                | Type     | Default  | Description                                                                                               |
|---------------------|----------|----------|-----------------------------------------------------------------------------------------------------------|
| `reward_threshold`  | `float`  | `None`   | The reward threshold before the task is considered solved                                                 |
| `nondeterministic`  | `bool`   | `False`  | Whether this environment is non-deterministic even after seeding                                          |
| `max_episode_steps` | `int`    | `None`   | The maximum number of steps that an episode can consist of. If not `None`, a `TimeLimit` wrapper is added |
| `order_enforce`     | `bool`   | `True`   | Whether to wrap the environment in an `OrderEnforcing` wrapper                                            |
| `autoreset`         | `bool`   | `False`  | Whether to wrap the environment in an `AutoResetWrapper`                                                  |
| `kwargs`            | `dict`   | `{}`     | The default kwargs to pass to the environment class                                                       |

이런 keywords의 대부분은(`max_episode_steps`, `order_enforce`, `kwargs` 제외) environment 인스턴스의 동작을 변경하지 않지만 environment에 대한 추가적인 정보만 제공한다.
registration 이후에 커스텀 `GridWorldEnv` environment는 `env = gym.make('gym_examples/GridWorld-v0')`로 생성할 수 있다.

`gym-examples/gym_examples/envs/__init__.py`는 다음을 가진다:

```python
from gym_examples.envs.grid_world import GridWorldEnv
```

## Package 생성하기

마지막 단계로 코드를 Python package로 구성하는 것이다. 이는 `gym-examples/setup.py` 설정과 관련된다. 어떻게 하는지에 간략한 예제는 다음과 같다:

```python
from setuptools import setup

setup(name='gym_examples',
    version='0.0.1',
    install_requires=['gym==0.23.1', 'pygame==2.1.0']
)
```

## Environment Instances 생성하기
package를 `pip install -e gym-examples` 를 사용하여 local에 설치한 후에 environment의 인스턴서를 생성하는 방법은 다음과 같다:

```python
import gym_examples
env = gym.make('gym_examples/GridWorld-v0')
```

environment 생성자의 keyword 인자를 `gym.make`에 전달하여 environment를 커스텀으로 만들 수 있다.
이 경우 다음과 같이 할 수 있다:

```python
env = gym.make('gym_examples/GridWorld-v0', size=10)
```

가끔 registration을 건너띄는 편리한 방법을 찾아서 environment 생성자를 호출할 수 있다. 어떤 사람은 이 접근법이 더 파이썬스럽고 이렇게 인스턴스화된 environment가 더 알맞아 보인다고 생각할 수 있다.(하지만 wrappers를 추가하는 방법도 기억해두자!)

## Wrappers 사용하기
다양한 커스텀 environment을 사용하려는 경우 혹은 Gym에서 제공하는 environment의 동작을 수정하는 경우가 자주 있다.
Wrappers를 이용하면 environment 구현을 변경이나 추가 없고 boilerplate 코드 추가 없이 커스텀으로 사용할 수 있다.
자세한 내용은 [wrapper documentation](https://www.gymlibrary.ml/content/wrappers/)를 참고하자.
예제로 observations는 학습 코드에서 집적 사용할수는 없다. 왜냐하면 이것들은 ditionaries이기 때문이다.
하지만 우리는 이것을 고치기 위해서 environment 구현을 건드릴 필요가 없다! 단순하게 environment 인스턴스의 꼭대기에 wrapper를 추가하여 observations를 단일 array로 변경할 수 있다:

```python
import gym_examples
from gym.wrappers import FlattenObservation

env = gym.make('gym_examples/GridWorld-v0')
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())     # E.g.  [3 0 3 3]
```

Wrappers가 큰 장점은 environments를 고도로 모듈화할 수 있다는 것이다.
예로 GridWorld로부터 observations를 flattening하는 대신에 target과 agent의 상대 위치를 보고 싶을 수도 있다.
[ObservationWrappers](https://www.gymlibrary.ml/content/wrappers/#observationwrapper)에 있는 섹션에서 wrapper를 구현하여 이 작업을 수행한다. 이 wrapper는 gym-examples에서도 유효하다.:

```python
import gym_examples
from gym_examples.wrappers import RelativePosition

env = gym.make('gym_examples/GridWorld-v0')
wrapped_env = RelativePosition(env)
print(wrapped_env.reset())     # E.g.  [-3  3]
```

