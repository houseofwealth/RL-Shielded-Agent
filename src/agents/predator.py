from .agent import BaseAgent


class Predator(BaseAgent):
    def __init__(self, num_dims, max_velocity, workspace_size, doing_obstacles, doing_geofence, LObs, RObs, TObs, BObs):
        super().__init__(num_dims, max_velocity=max_velocity, workspace_size=workspace_size, 
                         doing_obstacles=doing_obstacles, doing_geofence=doing_geofence,
                         LObs=LObs, RObs=RObs, TObs=TObs, BObs=BObs)
