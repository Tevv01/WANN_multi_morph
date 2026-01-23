from evogym.envs.base import EvoGymBase
from evogym.envs.traverse import StairsBase
import gymnasium as gym
from gymnasium import error, spaces
from gymnasium import utils
from gymnasium.utils import seeding

from evogym import *
from evogym.envs import BenchmarkBase

import random
import math
import numpy as np
import os
from typing import Dict, Any, Optional



class StairsBaseNew(BenchmarkBase):
    
    def __init__(
        self,
        world: EvoWorld,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(world=world, render_mode=render_mode, render_options=render_options)
        # Forced obs size so the network gets the right dims in
        self.FIXED_OBS_SIZE = 80
        self.FIXED_ACTION_SIZE = 25

    def get_reward(self, robot_pos_init, robot_pos_final):
        
        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        reward = (robot_com_pos_final[0] - robot_com_pos_init[0])
        return reward

    def get_obs(self) -> np.ndarray:
        """Get current observation"""
        # observation
        raw_obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        
        if raw_obs.shape[0] < self.FIXED_OBS_SIZE:
            obs = np.pad(raw_obs, (0, self.FIXED_OBS_SIZE - raw_obs.shape[0]), mode='constant')
        else:
            obs = raw_obs[:self.FIXED_OBS_SIZE]
        

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        super().reset(seed=seed, options=options)

        # observation
        robot_ort = self.object_orientation_at_time(self.get_time(), "robot")
        raw_obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))

        if raw_obs.shape[0] < self.FIXED_OBS_SIZE:
            obs = np.pad(raw_obs, (0, self.FIXED_OBS_SIZE - raw_obs.shape[0]))
        else:
            obs = raw_obs[:self.FIXED_OBS_SIZE]

        return obs, {}



class WalkingBumpy(StairsBaseNew):

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ObstacleTraverser-v0.json'))
        self.world.add_from_array('robot', body, 2, 1, connections=connections)

        # init sim
        StairsBaseNew.__init__(self, world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        # Forced obs size so the network gets the right dims in
        self.FIXED_OBS_SIZE = 100
        self.FIXED_ACTION_SIZE = 25

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=float)


    def step(self, action):

        # collect pre step information
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # Handle case where network outputs more actions than actuators
        num_actuators = self.get_actuator_indices('robot').size
        action = action[:num_actuators]  # Only use the first N actions where N is number of actuators

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort_final]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
       
        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
        
        # check termination condition
        com_pos = np.mean(robot_pos_final, axis=1)
        if com_pos[0] > (79)*self.VOXEL_SIZE:
            done = True
            reward += 2.0 
        if robot_ort_final > math.pi/2 and robot_ort_final < 3*math.pi/2:
            done = True
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}
    







class WalkingBumpy2(StairsBaseNew):

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ObstacleTraverser-v1.json'))
        self.world.add_from_array('robot', body, 2, 4, connections=connections)

        # init sim
        StairsBaseNew.__init__(self, world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        # Forced obs size so the network gets the right dims in
        self.FIXED_OBS_SIZE = 80
        self.FIXED_ACTION_SIZE = 25

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(self.FIXED_ACTION_SIZE,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(self.FIXED_OBS_SIZE,), dtype=float)

    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and optionally switch morphology"""
        
        # Reset base sim first 
        super().reset(seed=seed, options=options)
        
        # Now get observation after sim is reset
        state = self.get_obs()
        return state, {}

    def get_obs(self) -> np.ndarray:
        """Get current observation"""
        # observation
        raw_obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        
        if raw_obs.shape[0] < self.FIXED_OBS_SIZE:
            obs = np.pad(raw_obs, (0, self.FIXED_OBS_SIZE - raw_obs.shape[0]))
        else:
            obs = raw_obs[:self.FIXED_OBS_SIZE]
        

        return obs

    def step(self, action):

        # collect pre step information
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # Handle case where network outputs more actions than actuators
        num_actuators = self.get_actuator_indices('robot').size
        raw_action = action[:num_actuators]  # Only use the first N actions where N is number of actuators
        action = np.clip(raw_action, 0.6, 1.6)
        
        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")

        obs = self.get_obs()

       
        # compute reward
        reward = super().get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

         # check termination condition
        com_pos = np.mean(robot_pos_final, axis=1)
        if com_pos[0] > (59)*self.VOXEL_SIZE:
            done = True
            reward += 2.0 

        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}