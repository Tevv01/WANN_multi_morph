import random
import numpy as np
import sys
from domain.make_env import make_env
from domain.task_gym import GymTask
from neat_src import *
from domain.evogym_walker import SimpleWalkerEnvClass
from evogym.envs import EvoGymBase
from domain.make_env import generate_morphs


class WannGymTask(GymTask):
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1, morphs = None, conns = None): # , morphs = None, conns = None 
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness
    """

    GymTask.__init__(self, game, paramOnly, nReps)

    if morphs:
      self.morphologies = morphs
    if conns:
      self.connections = conns

    #self.morphologies, self.connections = generate_morphs(num_morphs=5, size=5)



# -- 'Weight Agnostic Network' evaluation -------------------------------- -- #
  def setWeights(self, wVec, wVal):
    """Set single shared weight of network
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      wVal    - (float)    - value to assign to all weights
  
    Returns:
      wMat    - (np_array) - weight matrix with single shared weight
                [N X N]
    """
    # Create connection matrix
    wVec[np.isnan(wVec)] = 0
    dim = int(np.sqrt(np.shape(wVec)[0]))    
    cMat = np.reshape(wVec,(dim,dim))
    cMat[cMat!=0] = 1.0

    # Assign value to all weights
    wMat = np.copy(cMat) * wVal 
    return wMat


  def getFitness(self, wVec, aVec, hyp, \
                    seed=-1,nRep=False,nVals=6,view=False,returnVals=False):
    """Get fitness of a single individual with distribution of weights
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
      hyp     - (dict)     - hyperparameters
        ['alg_wDist']        - weight distribution  [standard;fixed;linspace]
        ['alg_absWCap']      - absolute value of highest weight for linspace
  
    Optional:
      seed    - (int)      - starting random seed for trials
      nReps   - (int)      - number of trials to get average fitness
      nVals   - (int)      - number of weight values to test

  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = hyp['alg_nReps']

    # Set weight values to test WANN with
    if (hyp['alg_wDist'] == "standard") and nVals==6: # Double, constant, and half signal 
      wVals = np.array((-2,-1.0,-0.5,0.5,1.0,2))
    else:
      wVals = np.linspace(-self.absWCap, self.absWCap ,nVals)


    # Get reward from 'reps' rollouts -- test population on same seeds
    reward = np.empty((nRep,nVals))
    for iRep in range(nRep):

      try:
        if issubclass(type(self.env), EvoGymBase):
          morph_idx = random.randint(0, len(self.morphologies) - 1)
          print(f"Switching to morph_idx: {morph_idx}")
          self.env = SimpleWalkerEnvClass(body=self.morphologies[morph_idx], connections=self.connections[morph_idx], render_mode=None)
      except:
        pass

      for iVal in range(nVals):
        wMat = self.setWeights(wVec,wVals[iVal])
        if seed == -1:
          reward[iRep,iVal] = self.testInd(wMat, aVec, seed=seed,view=view)
        else:
          reward[iRep,iVal] = self.testInd(wMat, aVec, seed=seed+iRep,view=view)
          
    if returnVals is True:
      return np.mean(reward,axis=0), wVals
    return np.mean(reward,axis=0)
 

