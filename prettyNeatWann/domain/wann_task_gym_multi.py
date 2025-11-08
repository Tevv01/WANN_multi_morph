from typing import Optional, Tuple
import numpy as np
import random
from domain.wann_task_gym import WannGymTask
from neat_src import act, selectAct


class WannGymTaskMultiMorph(WannGymTask):
    """Evaluate WANNs across multiple morphologies.

    This extends WannGymTask by iterating over the morphologies available in
    the environment. It reuses the same interfaces as the base class so the
    optimizer code doesn't need to change.
    """
    def __init__(self, envClass, paramOnly=False, nReps=1):
        """Initialize task with environment class.
        
        Args:
            envClass: Environment class to instantiate (not instance)
            paramOnly: Only load parameters, don't create env
            nReps: Number of evaluation repetitions
        """
        # Store parameters
        self.envClass = envClass
        self.paramOnly = paramOnly
        self.nReps = nReps
        self.maxEpisodeLength = 1000  # Default episode length
        self.actSelect = np.array([1.0]) # Default activation selection

        if not paramOnly:
            # Create environment with a simple test body
            test_body = np.array([[1, 1], [1, 1]])
            self.env = envClass(bodies=[test_body], render_mode=None)

            # Get input/output sizes
            for _ in range(3):  # Try a few times
                try:
                    state = self.env.reset()[0]  # Get initial state
                    self.nInput = int(state.shape[-1])
                    self.nOutput = int(self.env.action_space.shape[0])
                    break
                except Exception as e:
                    print(f"Warning - environment init retry: {str(e)}")
                    continue
            else:
                print("Warning: Using default sizes after retries failed")
                self.nInput = 24  # Default sizes based on walker
                self.nOutput = 8

    def testInd(self, wVec, aVec, hyp=None, view=False, seed=-1, return_components=False):
        """Test a WANN (weights and activations) on all morphologies.

        Args:
          wVec - weight matrix (flattened or square) or vector accepted by act()
          aVec - activation vector
          hyp  - unused (kept for interface compatibility)
          view - render each rollout
          seed - random seed (single value)
          return_components - if True, return per-morphology scores array

        Returns:
          mean score across morphologies (or array if return_components=True)
        """
        # Keep RNG deterministic across morphologies if seed provided
        if seed is not None and seed >= 0:
            random.seed(seed)
            np.random.seed(seed)

        # Number of morphologies provided by the environment
        if not hasattr(self.env, 'get_num_morphologies'):
            raise RuntimeError('Environment does not expose multiple morphologies')
        n_morphs = self.env.get_num_morphologies()
        
        # Print morphology info on first evaluation
        if not hasattr(self, '_printed_info'):
            print(f"[Worker] Testing with {n_morphs} morphologies")
            self._printed_info = True

        scores = np.zeros(n_morphs)

        for m in range(n_morphs):
            total_score = 0.0
            for rep in range(self.nReps):
                # switch morphology and reset (capture returned observation)
                try:
                    state, info = self.env.reset(options={'morph_index': m})
                except TypeError:
                    # older env API
                    self.env.reset(options={'morph_index': m})
                    state = self.env.reset()

                # mimic GymTask.testInd loop but for a single rollout
                done = False
                totalReward = 0.0

            # Determine per-morph observation/action sizes
            try:
                local_nInput = int(state.shape[-1])
            except Exception:
                local_nInput = self.nInput
            try:
                local_nOutput = int(self.env.action_space.shape[0])
            except Exception:
                local_nOutput = self.nOutput

            # first action and step
            out = act(wVec, aVec, local_nInput, local_nOutput, state)
            action = selectAct(out, self.actSelect)
            try:
                state, reward, done, info, _ = self.env.step(action)
            except ValueError:
                state, reward, done, info = self.env.step(action)

            totalReward += reward

            # continue until done or max steps
            if self.maxEpisodeLength != 0:
                for t in range(self.maxEpisodeLength - 1):
                    out = act(wVec, aVec, local_nInput, local_nOutput, state)
                    action = selectAct(out, self.actSelect)
                    try:
                        state, reward, done, info, _ = self.env.step(action)
                    except ValueError:
                        state, reward, done, info = self.env.step(action)
                    totalReward += reward
                    if view:
                        try:
                            self.env.render()
                        except Exception:
                            pass
                    if done:
                        break

                total_score += totalReward
            
            # Average across repetitions
            scores[m] = total_score / self.nReps

        if return_components:
            return scores
        # Return mean score across morphologies
        return np.mean(scores)

    def getFitness(self, wVec: np.ndarray, aVec: np.ndarray, hyp: dict,
                  seed: int = -1, nRep: bool = False, nVals: int = 6,
                  view: bool = False, returnVals: bool = False) -> float:
        """Get fitness across multiple morphologies with weight distribution
        
        Args:
            Same as parent class
            
        Returns:
            float: Mean fitness across all morphologies and weight values
        """
        if nRep is False:
            nRep = hyp['alg_nReps']

        # Set weight values to test WANN with
        if (hyp['alg_wDist'] == "standard") and nVals == 6:
            wVals = np.array((-2, -1.0, -0.5, 0.5, 1.0, 2))
        else:
            wVals = np.linspace(-self.absWCap, self.absWCap, nVals)

        # Get reward from 'reps' rollouts across all morphologies
        reward = np.empty((nRep, nVals))
        for iRep in range(nRep):
            for iVal in range(nVals):
                wMat = self.setWeights(wVec, wVals[iVal])
                if seed == -1:
                    reward[iRep, iVal] = self.testInd(wMat, aVec, seed=seed, view=view)
                else:
                    reward[iRep, iVal] = self.testInd(wMat, aVec, seed=seed+iRep, view=view)

        if returnVals:
            return np.mean(reward, axis=0), wVals
        return np.mean(reward, axis=0)