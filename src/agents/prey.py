import numpy as np
from .agent import BaseAgent

class Prey(BaseAgent):
    #SN: assuming that unlike Predator, Prey doesn't need DOING_OBSTACLES, obstacle dims, etc
    def __init__(self, num_dims, max_velocity, workspace_size):
        super().__init__(num_dims=num_dims, max_velocity=max_velocity, workspace_size=workspace_size)
