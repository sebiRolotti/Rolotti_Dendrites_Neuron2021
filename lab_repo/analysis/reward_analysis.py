import numpy as np
import pandas as pd
import itertools as it
from collections import defaultdict
import warnings

import lab_repo.analysis.behavior_analysis as ba
from lab_repo.classes.classes import ExperimentGroup
from lab_repo.classes import exceptions as exc

def lick_to_reward_distance(expt_grp, rewardPositions=None):
    """Calculate the average lick to reward distance.

    Parameters
    ----------
    rewardPositions : {str, None, np.ndarray}
        If a string, assumed to be a condition label, and will use the
        reward positions used for each mouse during the condition.
        If 'None', uses the actual reward positions during the experiment.
        Otherwise pass in normalized reward positions.

    Returns
    -------
    pd.DataFrame

    """
    result = []

    if rewardPositions is None:
        rewards_by_expt = {
            expt: expt.rewardPositions(units='normalized')
            for expt in expt_grp}
    else:
        rewards_by_expt = defaultdict(lambda: np.array(rewardPositions))

    for expt in expt_grp:

        rewards = rewards_by_expt[expt]

        for trial in expt.findall('trial'):
            bd = trial.behaviorData(imageSync=False)
            position = ba.absolutePosition(
                trial, imageSync=False, sampling_interval='actual')

            if np.any(rewards >= 1.0):
                trial_rewards = rewards / bd['trackLength']
            else:
                trial_rewards = rewards

            licking = bd['licking'][:, 0]
            licking = licking[np.isfinite(licking)]
            licking = licking / bd['samplingInterval']
            licking = licking.astype('int')

            licking_positions = position[licking] % 1

            # meshgrid sets up the subtraction below
            # basically tile expands the arrays
            rewards_mesh, licking_mesh = np.meshgrid(
                trial_rewards, licking_positions)

            reward_distance = licking_mesh - rewards_mesh
            # All distances should be on [-0.5, 0.5)
            reward_distance[reward_distance >= 0.5] -= 1.0
            reward_distance[reward_distance < -0.5] += 1.0

            reward_distance = np.amin(np.abs(reward_distance), axis=1)

            assert len(licking_positions) == len(reward_distance)
            for lick, position in it.izip(
                    reward_distance, licking_positions):
                result.append(
                    {'expt': expt.trial_id, 'position': position, 'value': lick, 'session': expt.session})
    return pd.DataFrame(result, columns=['expt', 'position', 'value', 'session'])
