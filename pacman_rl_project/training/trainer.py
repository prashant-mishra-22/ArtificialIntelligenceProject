import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.replay_buffer import ReplayBuffer
import numpy as np
import torch
import os

class PacmanTrainer:
    def __init__(self, env, agent, visualizer=None):
        self.env = env
        self.agent = agent
        self.visualizer = visualizer
        
        # Training parameters
        self.batch_size = 32
        self.target_update = 100
        self.replay_buffer_size = 10000
        
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        # Tracking
        self.episode_rewards = []
        self.episode_scores = []
    
    def train(self, num_episodes=1000):
        print("Starting Training...")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            total_score = 0
            steps = 0
            
            while True:
                # Select and perform action
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Train agent
                loss = self.agent.train_step(self.replay_buffer, self.batch_size)
                
                state = next_state
                total_reward += reward
                total_score = info.get('score', 0)
                steps += 1
                
                # Render if visualizer available
                if self.visualizer:
                    self.visualizer.render(episode, total_score, self.agent.epsilon)
                
                # Update target network
                if steps % self.target_update == 0:
                    self.agent.update_target_network()
                
                if done:
                    break
            
            # Update epsilon
            self.agent.update_epsilon()
            
            # Track progress
            self.episode_rewards.append(total_reward)
            self.episode_scores.append(total_score)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_score = np.mean(self.episode_scores[-10:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model periodically
            if episode % 100 == 0:
                self.save_model(episode)
    
    def save_model(self, episode):
        if not os.path.exists('models'):
            os.makedirs('models')
        
        torch.save({
            'episode': episode,
            'policy_state_dict': self.agent.policy_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon,
            'rewards': self.episode_rewards,
            'scores': self.episode_scores
        }, f'models/pacman_dqn_episode_{episode}.pth')
        
        print(f"Model saved at episode {episode}")