# hrc_role_assignment

Repo used for code related to the legible role assignment project, which uses the Overcooked environment. Includes the following submodules:
`overcooked_ai` : Fork-of-a-fork of the original Overcooked gym implementation. Merges modifications by @StephAO to allow for HRL and @yi-shiuan-tung to implement additional ingredients and recipe types.
<!-- `overcooked_hrl`: Fork of @StephAO's repo which implements multiple RL agent types and training approaches, including a heirarchical RL implementation. -->

## first time setup
```
git clone --recurse-submodules git@github.com:cairo-robotics/hrc_role_assignment.git
cd hrc_role_assignment/overcooked_ai && pip install -e .
cd ../overcooked_hrl && pip install -e .
```


Note: LLM caller script expects config.yaml containing
```
OPENAI_API_KEY:
  <your_key_here>
```