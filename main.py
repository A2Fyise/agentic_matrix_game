import gymnasium as gym 
import numpy as np 
from typing import Optional
from stable_baselines3 import PPO

from print import print_grid
from step import agent_step

class agentic_world(gym.Env):
    def __init__(self, size: int = 4):
        self.size = size
        self.agent_location = np.array([-1, -1], dtype=np.int32)
        self.target_location = np.array([-1, -1], dtype=np.int32)


        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = gym.spaces.Discrete(4)

        self.action_directions = {
            0: np.array([1, 0]), #right
            1: np.array([0, 1]), #up 
            2: np.array([-1, 0]), #left 
            3: np.array([0, -1]), #down
        }

    def get_observation(self):
        return {"agent": self.agent_location, "target": self.target_location}

    def get_information(self):
        return {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.target_location = self.agent_location

        while np.array_equal(self.target_location, self.agent_location):
            self.target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self.get_observation()
        information = self.get_information()
        return observation, information

    def step(self, action):
        agent_step(self, action)
    
    def print(self):
        print_grid(self)

env = agentic_world()
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

obs, _ = env.reset()

for _ in range(500):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(env.get_observation())
    env.print()
    if done:
        obs, _ = env.reset()
        break
