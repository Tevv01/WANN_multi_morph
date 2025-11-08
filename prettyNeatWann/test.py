import gymnasium as gym
import evogym.envs
from evogym import sample_robot
import numpy as np

ACTUATOR_IDS = [3,4]

def count_actuators(morphology):
    """Return coordinates and count of actuator cells."""
    actuator_positions = list(zip(*np.where(np.isin(morphology, ACTUATOR_IDS))))
    return actuator_positions, len(actuator_positions)

def adjust_actuators(morphology, target=5):
    """Ensure morphology has exactly `target` actuators."""
    morphology = morphology.copy()
    actuator_positions, n = count_actuators(morphology)
    
    # If more than target → randomly remove some actuators
    if n > target:
        remove_count = n - target
        remove_indices = np.random.choice(len(actuator_positions), remove_count, replace=False)
        for i in remove_indices:
            y, x = actuator_positions[i]
            morphology[y, x] = 1  # replace with rigid or soft material (pick 1 as example)
    
    # If fewer than target → randomly add actuators
    elif n < target:
        add_count = target - n
        # Find all non-actuator, non-empty cells
        candidates = list(zip(*np.where((morphology != 0) & (~np.isin(morphology, ACTUATOR_IDS)))))
        if len(candidates) < add_count:
            print("Not enough valid cells to add actuators.")
        else:
            add_indices = np.random.choice(len(candidates), add_count, replace=False)
            for i in add_indices:
                y, x = candidates[i]
                morphology[y, x] = np.random.choice(ACTUATOR_IDS)  # random actuator type
    
    return morphology


def generate_morphs(num_morphs, size = 5):
    morphologies = []
    for _ in range(num_morphs):
        body, connections = sample_robot((size, size))
        body = adjust_actuators(body)
        morphologies.append(body)
    
    return morphologies


if __name__ == '__main__':
    morphs = generate_morphs(5)

    for morph in morphs:
        print(morph)
        env = gym.make('Walker-v0', body=morph, render_mode='human')
        env.reset()

        terminated = False
        for _ in range(500):
            action = env.action_space.sample()
            ob, reward, terminated, truncated, info = env.step(action)

            if (_ == 1):
                print(ob)

            if terminated or truncated:
                env.reset()

        env.close()