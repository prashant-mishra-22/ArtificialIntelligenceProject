import numpy as np
import pygame
import random
from enum import Enum

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STOP = (0, 0)

class PacmanEnvironment:
    def __init__(self, width=15, height=15):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        # Initialize empty grid
        self.grid = np.zeros((self.height, self.width))
        
        # Place Pacman in center
        self.pacman_pos = [self.width // 2, self.height // 2]
        
        # Place food randomly
        self.food_positions = []
        for _ in range(20):  # 20 food pellets
            while True:
                pos = [random.randint(1, self.width-2), random.randint(1, self.height-2)]
                if pos != self.pacman_pos and pos not in self.food_positions:
                    self.food_positions.append(pos)
                    break
        
        # Initialize ghosts
        self.ghosts = [
            {'pos': [2, 2], 'direction': Direction.RIGHT},
            {'pos': [self.width-3, self.height-3], 'direction': Direction.LEFT}
        ]
        
        # Game state
        self.score = 0
        self.steps = 0
        self.done = False
        
        return self._get_state()
    
    def _get_state(self):
        """Convert game state to image representation"""
        state = np.zeros((self.height, self.width, 3))
        
        # Walls (boundaries)
        state[0, :, :] = [0, 0, 1]  # Blue walls
        state[-1, :, :] = [0, 0, 1]
        state[:, 0, :] = [0, 0, 1]
        state[:, -1, :] = [0, 0, 1]
        
        # Pacman (yellow)
        state[self.pacman_pos[1], self.pacman_pos[0], :] = [1, 1, 0]
        
        # Food (white)
        for food in self.food_positions:
            state[food[1], food[0], :] = [1, 1, 1]
        
        # Ghosts (red)
        for ghost in self.ghosts:
            state[ghost['pos'][1], ghost['pos'][0], :] = [1, 0, 0]
        
        return state
    
    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}
        
        reward = -0.1  # Small negative reward for each step
        self.steps += 1
        
        # Move Pacman
        direction = list(Direction)[action]
        new_pos = [
            self.pacman_pos[0] + direction.value[0],
            self.pacman_pos[1] + direction.value[1]
        ]
        
        # Check wall collision
        if (1 <= new_pos[0] < self.width-1 and 
            1 <= new_pos[1] < self.height-1):
            self.pacman_pos = new_pos
        
        # Check food collection
        if self.pacman_pos in self.food_positions:
            self.food_positions.remove(self.pacman_pos)
            reward += 10
            self.score += 10
        
        # Move ghosts (simple random movement)
        for ghost in self.ghosts:
            # 30% chance to change direction
            if random.random() < 0.3:
                ghost['direction'] = random.choice(list(Direction))
            
            new_ghost_pos = [
                ghost['pos'][0] + ghost['direction'].value[0],
                ghost['pos'][1] + ghost['direction'].value[1]
            ]
            
            # Ghost wall collision check
            if (1 <= new_ghost_pos[0] < self.width-1 and 
                1 <= new_ghost_pos[1] < self.height-1):
                ghost['pos'] = new_ghost_pos
        
        # Check ghost collision
        for ghost in self.ghosts:
            if self.pacman_pos == ghost['pos']:
                reward -= 50
                self.done = True
                break
        
        # Check win condition
        if len(self.food_positions) == 0:
            reward += 100
            self.done = True
        
        # Check step limit
        if self.steps >= 1000:
            self.done = True
        
        return self._get_state(), reward, self.done, {'score': self.score}
    
    def get_action_space(self):
        return len(Direction)
    
    def render(self, screen, cell_size=30):
        screen.fill((0, 0, 0))
        
        # Draw grid
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                
                # Draw walls
                if (x == 0 or x == self.width-1 or 
                    y == 0 or y == self.height-1):
                    pygame.draw.rect(screen, (0, 0, 255), rect)
                
                # Draw food
                if [x, y] in self.food_positions:
                    pygame.draw.circle(screen, (255, 255, 255), 
                                     (x * cell_size + cell_size//2, 
                                      y * cell_size + cell_size//2), 
                                     cell_size//8)
                
                # Draw Pacman
                if [x, y] == self.pacman_pos:
                    pygame.draw.circle(screen, (255, 255, 0), 
                                     (x * cell_size + cell_size//2, 
                                      y * cell_size + cell_size//2), 
                                     cell_size//2)
                
                # Draw ghosts
                for ghost in self.ghosts:
                    if [x, y] == ghost['pos']:
                        pygame.draw.circle(screen, (255, 0, 0), 
                                         (x * cell_size + cell_size//2, 
                                          y * cell_size + cell_size//2), 
                                         cell_size//2)