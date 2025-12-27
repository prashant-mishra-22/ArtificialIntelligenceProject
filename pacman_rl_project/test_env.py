from environments.pacman_env import PacmanEnvironment
from gui.pygame_visualizer import PyGameVisualizer

env = PacmanEnvironment(width=10, height=10)
visualizer = PyGameVisualizer(env)

state = env.reset()
for i in range(100):
    action = 0  # Move right
    state, reward, done, info = env.step(action)
    visualizer.render(i, info.get('score', 0), 0.5, fps=5)
    
    if done:
        break

visualizer.close()