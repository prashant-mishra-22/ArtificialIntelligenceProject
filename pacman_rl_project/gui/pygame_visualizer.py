import pygame
import numpy as np

class PyGameVisualizer:
    def __init__(self, env, cell_size=30):
        self.env = env
        self.cell_size = cell_size
        self.width = env.width * cell_size
        self.height = env.height * cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pacman RL Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
    
    def render(self, episode, score, epsilon, fps=10):
        self.env.render(self.screen, self.cell_size)
        
        # Display training info
        info_text = f"Episode: {episode} | Score: {score} | Epsilon: {epsilon:.2f}"
        text_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 5))
        
        pygame.display.flip()
        self.clock.tick(fps)
    
    def close(self):
        pygame.quit()