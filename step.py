import numpy as np

def agent_step(self, action):
    action = action.item() if isinstance(action, np.ndarray) else action
    direction = self.action_directions[action]

    self.agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)

    terminated = np.array_equal(self.agent_location, self.target_location)
    truncated = False
    reward = 1 if terminated else 0
    observation = self.get_observation()
    information = self.get_information()
    return observation, reward, terminated, truncated, information