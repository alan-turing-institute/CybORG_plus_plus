import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minimal import SimplifiedCAGE

MAX_STEPS = 20

class CAGEBlueWrapper(gym.Env):
    """
    Wrapper for Stable Baselines3 to train a **blue agent**.
    - The red agent follows a fixed strategy.
    - The RL policy controls only the blue agent.
    """
    def __init__(self, red_agent):
        super(CAGEBlueWrapper, self).__init__()
        
        self.env = SimplifiedCAGE(num_envs=1, num_nodes=13, red_agent=red_agent)
        self.episode_length = 0
        temp_state, _ = self.env.reset()
        blue_obs_shape = temp_state['Blue'].shape
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=blue_obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.env.action_mapping['Blue'])) 


    def reset(self, seed=None):
        """ Reset environment and return the blue observation. """
        state, info = self.env.reset()
        self.episode_length = 0
        self.state_po = state
        return state['Blue'].astype(np.float32), info  # **Ensure correct format**

    def step(self, action):
        """ Take a step where the RL agent controls the blue team. """
        blue_action = np.array([[action]])  # Format action correctly
        self.episode_length += 1
        red_obs = self.state_po['Red']
        red_action = self.env.red_agent.get_action(red_obs)
        
        state, rewards, done, info = self.env.step(blue_action=blue_action, red_action=red_action)
        self.state_po = state
        done = self.episode_length >= MAX_STEPS
        return state['Blue'].astype(np.float32), float(rewards['Blue'][0][0]), done, False, info


class CAGERedWrapper(gym.Env):
    """
    Wrapper for Stable Baselines3 to train a **red agent**.
    - The blue agent follows a fixed strategy.
    - The RL policy controls only the red agent.
    """
    def __init__(self, blue_agent):
        super(CAGERedWrapper, self).__init__()

        self.env = SimplifiedCAGE(num_envs=1, num_nodes=13, blue_agent=blue_agent)

        temp_state, _ = self.env.reset()
        red_obs_shape = temp_state['Red'].shape  # Ensure this is actually (40,)!
    
        self.observation_space = spaces.Box(low=-1, high=1, shape=red_obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(56)  # Red action space
        self.state_po = None

    def reset(self, seed=None):
        """ Reset environment and return the red observation. """
        state, info = self.env.reset()
        self.state_po = state
        self.episode_length = 0
        return state['Red'].astype(np.float32), info  # **Ensure correct shape**

    def step(self, action):
        """ Take a step where the RL agent controls the red team. """
        red_action = np.array([[action]])  # Format action correctly
        self.episode_length += 1

        blue_obs = self.state_po['Blue']
        blue_action = self.env.blue_agent.get_action(blue_obs)
        state, rewards, done, info = self.env.step(blue_action=blue_action, red_action=red_action)

        self.state_po = state
        done = self.episode_length >= MAX_STEPS

        return state['Red'].astype(np.float32), float(rewards['Red'][0][0]), done, False, info 
