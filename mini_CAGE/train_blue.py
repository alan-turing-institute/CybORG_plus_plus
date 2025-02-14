from stable_baselines3 import PPO
from agents import Meander_minimal, B_line_minimal  # Fixed Red agent
from wrappers import CAGEBlueWrapper

# Initialize environment with fixed Red agent
red_agent = B_line_minimal()
env = CAGEBlueWrapper(red_agent=red_agent)

# Train PPO with optimized parameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0005,  #  Slightly reduce learning rate for stability
    batch_size=1024,       #  Larger batch size for better updates
    # clip_range=0.1,        #  Lower clip range to stabilize training
    # gae_lambda=0.98,       #  Higher lambda for better advantage estimation
    # ent_coef=0.01,         #  Reduce entropy regularization (less randomness)
)

model.learn(total_timesteps=100_000)  
model.save("ppo_blue_defender")
