# Gym-docs

이 저장소는 [Gym](https://github.com/openai/gym)에 대한 [새로운 웹사이트](http://www.gymlibrary.ml)에 대한 내용을 포함하고 있다. 이 사이트는 현재 베타 버전이고 추가/수정을 진행 중이다.


이 문서는 Sphinx를 사용하지만 문서는 markdown으로 작성되었다.

If you are modifying a non-environment page or an atari environment page, please PR this repo. Otherwise, follow the steps below:

## environment 페이지를 수정하기 위한 절차

### environment 페이지 수정하기

Atari environment를 수정하고자 한다면 이 repo에서 직접 md 파일을 수정한다.

그렇지 않은 경우 Gym을 fork하고 environment의 Python 파일에 있는 docstring을 수정한다. 다음으로 여러분의 Gym fork을 pip install 하고 이 repo 내에 있는 `docs/scripts/gen_mds.py`를 실행한다. 이것은 자동으로 해당 environment에 대한 md 문서 파일을 생성한다.

### Adding a new environment

#### Atari env

Atari envs에 대해서 `pages/environments/atari`에 md 파일을 추가하고 난 후에 **다른 단계들**을 진행한다.

#### Non-Atari env

Gym에 있는 해당 environment를 확인한다. 해당 environment의 Python 파일은 markdown 형식인지 확인한다. pip install gym을 수행하고 `docs/scripts/gen_mds.py`를 실행한다. 이렇게 하면 자동으로 해당 environment에 대한 md 페이지를 생성한다. 다음으로 [다른 단계들](#other-steps)를 완료한다.

#### 다른 단계들

- Add the corresponding gif into the `docs/source/_static/videos/{ENV_TYPE}` folder, where `ENV_TYPE` is the category of your new environment (e.g. mujoco). Follow snake_case naming convention. Alternatively, run `docs/scripts/gen_gifs.py`.
- Edit `docs/source/environments/{ENV_TYPE}/index.md`, and add the name of the file corresponding to your new environment to the `toctree`.

## 문서 빌드하기

Gym과 필요한 package들을 설치:

```
pip install -r requirements.txt
pip install gym
```

문서를 빌드:

```
cd docs
make dirhtml
```

문서에 변경이 생길때마다 자동으로 빌드하게 만들기:

```
cd docs
sphinx-autobuild -b dirhtml ./source build/html
```
