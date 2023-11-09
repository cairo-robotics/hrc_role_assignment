#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='overcooked_role_assignment',
      version='0.1.0',
      description='Cooperative multi-agent environment based on Overcooked',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Kaleb Bishop',
      author_email='kaleb.bishop@colorado.edu',
      url='https://github.com/StephAO/overcooked_role_assignment',
      download_url='https://github.com/StephAO/overcooked_role_assignment',
      keywords=['Overcooked', 'AI', 'Reinforcement Learning', 'Human Agent Collaboration'],
      # packages=find_packages('overcooked_role_assignment'),
      # package_dir={"": "overcooked_role_assignment"},
      packages=['overcooked_role_assignment', 'overcooked_role_assignment.agents', 'overcooked_role_assignment.gym_environments', 'overcooked_role_assignment.common'],
      package_dir={
          'overcooked_role_assignment': 'overcooked_role_assignment',
          'overcooked_role_assignment.agents': 'overcooked_role_assignment/agents',
          'overcooked_role_assignment.gym_environments': 'overcooked_role_assignment/gym_environments',
          'overcooked_role_assignment.common': 'overcooked_role_assignment/common'
      },
      package_data={
        'overcooked_role_assignment' : [
          'data/*.pickle'
        ],
      },
      install_requires=[
        'numpy',
        'tqdm',
        'wandb',
        'gym',
        'pygame',
      ],
      tests_require=['pytest']
    )