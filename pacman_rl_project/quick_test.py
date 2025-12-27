from environments.pacman_env import PacmanEnvironment
from agents.dqn_agent import DQNAgent
from gui.pygame_visualizer import PyGameVisualizer
from training.trainer import PacmanTrainer

# Quick test with fewer episodes
env = PacmanEnvironment(width=10, height=10)  # Smaller grid for faster training
state_shape = env._get_state().shape
num_actions = env.get_action_space()

agent = DQNAgent(state_shape, num_actions)
visualizer = PyGameVisualizer(env)
trainer = PacmanTrainer(env, agent, visualizer)

# Train for just 50 episodes to test
trainer.train(num_episodes=50)