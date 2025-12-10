from gymnasium import spaces
from evogym import EvoWorld, EvoSim
from evogym.envs import EvoGymBase
from evogym.envs import register

from typing import Optional, Dict, Any, Tuple
import numpy as np
import os
import random


class SimpleWalkerEnvClass(EvoGymBase):

    def __init__(
        self,
        body: Optional[np.ndarray] = None,
        bodies: Optional[list[np.ndarray]] = None,
        connections: Optional[list[np.ndarray]] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):
        
        # Handle single body case (from gym.make) vs multiple bodies case (from our task)
        if bodies is not None:
            self.bodies = bodies
            self.connections = connections if connections else [None] * len(bodies)
            self.current_morph_idx = 0
            body = bodies[0]  # Use first body for initial setup
            connection = self.connections[0] if connections else None
        else:
            self.bodies = [body] if body is not None else []
            self.connections = [None]
            self.current_morph_idx = 0
            connection = None
        
        # Store rendering options
        self._render_mode = render_mode
        self._render_options = render_options
        
        # Forced obs size so the network gets the right dims in
        self.FIXED_OBS_SIZE = 66
        self.FIXED_ACTION_SIZE = 20

        # Create world template first
        world_json = os.path.join(os.path.dirname(__file__), 'world_data', 'simple_walker_env.json')
        world = EvoWorld.from_json(world_json)
            
        # Add robot to world before initializing base class
        if body is not None:
            world.add_from_array('robot', body, 1, 1, connections=connection)
            
        # Initialize base class with prepared world
        super().__init__(world=world, render_mode=render_mode, render_options=render_options)
            

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(self.FIXED_ACTION_SIZE,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(self.FIXED_OBS_SIZE,), dtype=float)

        # set viewer to track objects
        self.default_viewer.track_objects('robot')


    def step(self, action):

        # Handle case where network outputs more actions than actuators
        num_actuators = self.get_actuator_indices('robot').size
        action = action[:num_actuators]  # Only use the first N actions where N is number of actuators


        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
            
        # check goal met
        if com_2[0] > 28:
            done = True
            reward += 1.0

        # observation
        raw_obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))
        
        if raw_obs.shape[0] < self.FIXED_OBS_SIZE:
            obs = np.pad(raw_obs, (0, self.FIXED_OBS_SIZE - raw_obs.shape[0]))
        else:
            obs = raw_obs[:self.FIXED_OBS_SIZE]

        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and optionally switch morphology"""
        
        # Reset base sim first 
        super().reset(seed=seed, options=options)
        
        # Now get observation after sim is reset
        state = self.get_obs()
        return state, {}
    
    def get_obs(self) -> np.ndarray:
        """Get current observation"""
        raw_obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
        ))

        if raw_obs.shape[0] < self.FIXED_OBS_SIZE:
            obs = np.pad(raw_obs, (0, self.FIXED_OBS_SIZE - raw_obs.shape[0]))
        else:
            obs = raw_obs[:self.FIXED_OBS_SIZE]

        return obs

    
    def set_morphology(self):
        """Switch to a different morphology"""
        morph_idx = random.randrange(0, len(self.bodies) - 1)

        print(f'Morph idx shuffled to: {morph_idx}')
            
        if morph_idx != self.current_morph_idx:
            self.current_morph_idx = morph_idx
            
            # Create clean world template
            world_json = os.path.join(os.path.dirname(__file__), 'world_data', 'simple_walker_env.json')
            world = EvoWorld.from_json(world_json)
            
            # Add robot to world
            world.add_from_array('robot', self.bodies[morph_idx], 1, 1, 
                               connections=self.connections[morph_idx])
            
            # Create new simulation with updated world
            self._sim = EvoSim(world)
            
            # Reset to initialize sim and get new sizes
       
            self.action_space = spaces.Box(low=0.6, high=1.6, shape=(self.FIXED_ACTION_SIZE,), dtype=float)
            self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(self.FIXED_OBS_SIZE,), dtype=float)

            
    def get_num_morphologies(self) -> int:
        """Return the number of available morphologies"""
        return len(self.bodies)