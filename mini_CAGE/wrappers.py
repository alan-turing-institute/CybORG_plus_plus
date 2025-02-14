import numpy as np
import gym
from gym import spaces
from minimal import SimplifiedCAGE

class CAGEBlueWrapper(gym.Env):
    """
    Wrapper for Stable Baselines3 to train a **blue agent**.
    - The red agent follows a fixed strategy.
    - The RL policy controls only the blue agent.
    """
    def __init__(self, red_agent):
        super(CAGEBlueWrapper, self).__init__()
        
        self.env = SimplifiedCAGE(num_envs=1, num_nodes=13, red_agent=red_agent)

        # **Fix the observation space to match the actual Blue state**
        obs_shape = self.env.reset()[0]['Blue'].shape
        self.observation_space = spaces.Box(low=-1, high=1, shape=obs_shape, dtype=np.float32)
        
        # Define Blue action space (Discrete: number of possible actions)
        self.action_space = spaces.Discrete(56)

    def reset(self):
        """ Reset environment and return the blue observation. """
        state, _ = self.env.reset()
        return state['Blue'].astype(np.float32)  # **Ensure correct format**

    def step(self, action):
        """ Take a step where the RL agent controls the blue team. """
        blue_action = np.array([[action]])  # Format action correctly
        red_action = self.env.red_agent.get_action(self.env.state['Red'])  # Fixed red agent
        
        state, rewards, done, info = self.env.step(blue_action=blue_action, red_action=red_action)
        
        return state['Blue'].astype(np.float32), float(rewards['Blue'][0][0]), False, info

class CAGERedWrapper(gym.Env):
    """
    Wrapper for Stable Baselines3 to train a **red agent**.
    - The blue agent follows a fixed strategy.
    - The RL policy controls only the red agent.
    """
    def __init__(self, blue_agent):
        super(CAGERedWrapper, self).__init__()

        self.env = SimplifiedCAGE(num_envs=1, num_nodes=13, blue_agent=blue_agent)

        # **🔹 Fix: Get Red observation shape dynamically**
        temp_state, _ = self.env.reset()
        red_obs_shape = temp_state['Red'].shape  # **Ensure this is actually (40,)!**
        
        print(f"🔍 Detected Red Observation Shape: {red_obs_shape}")  # Debugging

        self.observation_space = spaces.Box(low=-1, high=1, shape=red_obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(56)  # Red action space

    def reset(self):
        """ Reset environment and return the red observation. """
        state, _ = self.env.reset()
        return state['Red'].astype(np.float32)  # **Ensure correct shape**

    def step(self, action):
        """ Take a step where the RL agent controls the red team. """
        red_action = np.array([[action]])  # Format action correctly
        blue_action = self.env.blue_agent.get_action(self.env.state['Blue'])  # Fixed blue agent

        state, rewards, done, info = self.env.step(blue_action=blue_action, red_action=red_action)

        return state['Red'].astype(np.float32), float(rewards['Red'][0][0]), False, info
