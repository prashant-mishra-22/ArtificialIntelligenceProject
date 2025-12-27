import pygame
from environments.pacman_env import PacmanEnvironment
from agents.dqn_agent import DQNAgent
from gui.pygame_visualizer import PyGameVisualizer
from training.trainer import PacmanTrainer

def main():
    # Initialize environment
    env = PacmanEnvironment(width=15, height=15)
    state_shape = env._get_state().shape  # (height, width, channels)
    num_actions = env.get_action_space()
    
    # Initialize agent
    agent = DQNAgent(state_shape, num_actions)
    
    # Initialize visualizer
    visualizer = PyGameVisualizer(env)
    
    # Initialize trainer
    trainer = PacmanTrainer(env, agent, visualizer)
    
    try:
        # Start training
        trainer.train(num_episodes=1000)
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        visualizer.close()
        print("Training completed!")

if __name__ == "__main__":
    main()