from stable_baselines3 import PPO
from agents import Meander_minimal, B_line_minimal  # Fixed Red agent
from wrappers import CAGEBlueWrapper

# ---------------------------
# Load Trained PPO Blue Agent
# ---------------------------
print("\n✅ Loading Trained PPO Blue Agent...")
blue_model = PPO.load("ppo_blue_defender")

# Initialize environment with fixed Red agent
red_agent = Meander_minimal()
env = CAGEBlueWrapper(red_agent=red_agent)

# **Extract action mapping from the environment**
blue_action_map = env.env.action_mapping['Blue']  # 🔥 Get the mapping

# ---------------------------
# Run Test Episode
# ---------------------------
print("\n🎯 Running Test Episode with PPO Blue vs Red...")
obs, _ = env.reset()
total_reward = 0

for step in range(20): 
    action, _ = blue_model.predict(obs, deterministic=False)  # Predict action for Blue
    
    obs, reward, done, truncated, info = env.step(action)  # Step environment
    total_reward += reward

    # **Map the action index to its corresponding name**
    action_name = blue_action_map[int(action)]


    print(f"Step {step+1}:")
    print(f"  🔵 Blue Action: {action_name} ({action})")
    print(f"  🎯 Reward: {reward}")

print(f"\nFinal Blue Total Reward: {total_reward}")
