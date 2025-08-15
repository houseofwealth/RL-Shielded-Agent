import pytest
import numpy as np
from ipdb import set_trace

from meher.env import meherEnv
from meher.config import DEFAULT_CONFIG


@pytest.fixture
def blank_config():
    blank_config = DEFAULT_CONFIG['env_config']
    blank_config['preds_collide'] = False
    blank_config['collide_with_ground'] = False
    blank_config['collide_with_herd'] = False
    return blank_config   

def run_till_done(env, action_function):
    done = False
    num_steps = 0
    while not done:
        action = action_function(env)
        obs, rew, done, truncated, info = env.step(action)
        num_steps += 1
        if info['task_failed']:
            done = True
    return info, num_steps

def make_pred_collide(env):
        position = env.position
        action = -1 * np.ones(len(env.position))
        return action

def make_pred_collide_with_ground(env):
        num_steps = env.num_episode_steps
        num_obs = env.num_dims * env.num_preds
        action = np.zeros(num_obs)
        if num_steps < 3:
            action[(env.num_dims - 1):num_obs:env.num_dims] = 1
        else:
            action[(env.num_dims - 1):num_obs:env.num_dims] = -1
        return action


# Test if predators collide with one another
def test_no_collisions(blank_config):
    '''
    When predator collision is off, predators do not collide
    '''
    env = meherEnv(blank_config)
    env.reset()
    info, num_steps = run_till_done(env, make_pred_collide)
    assert info['task_failed'] == False, 'Predators should not have collided!'

def test_3_preds_collide(blank_config):
    ''' 
    Predator collision seeded to a point where 3 predators collide at the same step
    '''
    blank_config['preds_collide'] = True
    env = meherEnv(blank_config)
    env.reset()
    env.predators[0].position = np.asarray([-10, -10, 0])
    env.predators[1].position = np.asarray([-8, -10, 0])
    env.predators[2].position = np.asarray([-10, -8, 0])
    info, num_steps = run_till_done(env, make_pred_collide)
    print(env.position)
    assert np.all(env.position == -1), 'Predators should have collided!'

def test_pred_collide_with_base(blank_config):
    '''
    Test Predators collide with Herd
    '''
    blank_config['collide_with_herd'] = True
    env = meherEnv(blank_config)
    env.reset()
    info, num_steps = run_till_done(env, make_pred_collide)
    assert info['task_failed'] == True, 'Predators should have collided with base!'

def test_pred_collide_with_ground(blank_config):
    '''
    Test Predators collide with ground
    '''
    blank_config['collide_with_ground'] = True
    env = meherEnv(blank_config)
    env.reset()
    info, num_steps = run_till_done(env, make_pred_collide_with_ground)
    assert np.all(env.position == -1), 'Predators should have collided with ground!'

# TODO fake predator tests

def test_pred_catches_prey(blank_config):
    '''
    Test the Predator catches Pray by initializing them at the same location
    '''
    blank_config['num_preds'] = 1
    env = meherEnv(blank_config)
    env.reset()
    env.predators[0].position = [0,0,1]
    env.prey.position = [0,0,3]
    info, num_steps = run_till_done(env, make_pred_collide_with_ground)
    assert env.prey.is_live == False, 'Predators should have captured prey!'

    obs = env.get_observation()[0]
    assert env._compute_reward(obs['achieved_goal'], obs['desired_goal'], {'done': True}) == 1, 'Predator reward should be +1'

