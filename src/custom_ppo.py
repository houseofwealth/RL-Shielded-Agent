from stable_baselines3 import PPO
from gymnasium import spaces
from ipdb import set_trace
import time
import sys
from stable_baselines3.common.utils import safe_mean, obs_as_tensor
from stable_baselines3.common.buffers import DictRolloutBuffer
import numpy as np
import torch as th
from copy import deepcopy
from shields import ( NoShield, Z3Shield)
# from ppo_her.utils import get_dict_idxs
import pdb

def get_dict_idxs(dict_in, indices):
    new_dict = {}
    if indices.dtype == 'bool':
        indices = np.where(indices)[0]
    for key, value in dict_in.items():
        new_dict[key] = np.take(value, indices, axis=0)
    return new_dict

class CustomPPO(PPO):
    def __init__(
        self,
        policy,
        env,
        use_shield=False,
        num_shield_chances=100,
        shield_angle=20,
        shield_type=None,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs=None,
        stop_early=False,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.use_shield = use_shield
        self.shield_type = shield_type
        self.num_shield_chances = num_shield_chances
        self.shield_angle = shield_angle
        self.stop_early = stop_early
        self.reward_timestep = None

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1,
        tb_log_name="OnPolicyAlgorithm",
        reset_num_timesteps=True,
        progress_bar=False
        ):

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        original_buffer_size = self.rollout_buffer.buffer_size

        #SN: moved from inside get_action as it only needs to be done once per training
        if self.use_shield: 
            self.shield = Z3Shield(self.env.envs[0], self.policy, self.num_shield_chances, device=self.device)
            # self.shield = Z3Shield(self.env, self.policy, self.num_shield_chances, device=self.device)

        while self.num_timesteps < total_timesteps:
            self.rollout_buffer.buffer_size = original_buffer_size
            self.rollout_buffer.reset()

            print('\nstarting new rollout buffer after num_timesteps =', self.num_timesteps)      #SN:
            #SN: why is env being passed to collect_rollouts if its already a member var?
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

                self.shield_types = np.asarray(self.shield_types)
                self.logger.record("shield/used", 1 - np.mean(self.shield_types == 'none'))
                self.logger.record("shield/policy", np.mean(self.shield_types == 'policy'))
                self.logger.record("shield/random", np.mean(self.shield_types == 'random'))
                self.logger.record("shield/fail", np.mean(self.shield_types == 'fail'))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)
                print('time_elapsed (m) =', time_elapsed/60)      #SN:

            if self.stop_early:
                mean_episode_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                if mean_episode_reward == 1:
                    if self.reward_timestep is None:
                        self.reward_timestep = self.num_timesteps
                
                if self.reward_timestep is not None:
                    if self.num_timesteps > (1.3 * self.reward_timestep):
                        break
            self.train()


        callback.on_training_end()

        return self

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps,
    ):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        self.shield_types = []
        # this doesn't work due to reset() reviving dead preds: all_predators_dead = False                                          #SN:

        tot_rew = 0
        while n_steps < n_rollout_steps: #and not all_predators_dead
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            # if n_steps % 1000 == 0: print('rollout step #',n_steps)  #SN:
            use_shield = env.envs[0].use_shield
            #SN: why is this called actions and not action, its a single item
            actions, values, log_probs, shield_type = self.get_action(self._last_obs, self.device, use_shield)
            self.shield_types.append(shield_type)
            if shield_type == 'fail':
                for e in env.envs:
                    e.env.end_now = True
            #     # Return negative reward and exit
            #     assert self.env.num_envs == 1, 'We have not tested this with parallel envs, yet!'
            #     new_obs = self.env.reset()
            #     # new_obs = self._last_obs
            #     rewards = np.asarray([-1], dtype=np.float32)
            #     dones = np.asarray([True])
            #     infos = np.asarray([{}])
                # print('Fail!')    #SN: turned off b/c clogging up the debugging output 
            # else:
                #SN: StableBaselines.step() called before env.step() returns turns rew into list
            new_obs, rewards, dones, infos = env.step(actions)

            # if dones[0]: print('collect_rollbacks: rewards', rewards)
            # if dones[0]: 
            #     tot_rew += rewards
            #     mean_rew = tot_rew/n_steps
                # print('total reward for rollout', tot_rew, "mean reward", mean_rew)
                
            # print('stepped') SN:
            # if infos[0]['pred_is_dead'] and n_steps % 1000 == 0: 
            #     print('all predators dead')
            # if infos[0]['pred_is_dead']: breakpoint()
            # all_predators_dead = infos[0]['pred_is_dead']            

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def get_action(self, last_obs, device, use_shield):
        assert self.env.num_envs == 1, 'Not implemented for vec envs!'

        use_shield = self.use_shield
        # if self.use_shield:
        #     use_shield = use_shield
        # else:
        #     use_shield = False

        env = self.env.envs[0]
        policy = self.policy
        FENCE_LOC = 5

        # For partial shielding, see if we should use the shield
        if use_shield:
            use_shield = env.use_shield

        if use_shield:          #TBD: move this outside the loop for all the shield types not just Z3
            if self.shield_type == 'angle':
                shield = AngleShield(
                    env, policy, self.num_shield_chances, device=self.device, angle=self.shield_angle
                )
            elif self.shield_type == 'rectanguloid':
                shield = RectanguloidGeoFence(
                    env, policy, self.num_shield_chances, fence_loc=FENCE_LOC,
                    device=self.device
                )
            elif self.shield_type == 'can_reach':
                shield = CanReach(
                    env, policy, self.num_shield_chances, device=self.device)
            elif self.shield_type == 'chase':
                shield = ChaseShield(
                    env, policy, self.num_shield_chances, device=self.device)
            elif self.shield_type == 'z3_shield':
                #SN: moved so its outside loop: shield = Z3Shield(env, policy, self.num_shield_chances, device=self.device)
                shield = self.shield
            else:
                raise ValueError('Unknown shield type: ' + str(self.shield_type))

        else:
            shield = NoShield(env, policy, self.num_shield_chances)
            
        
        with th.no_grad():
            obs_tensor = obs_as_tensor(last_obs, device)
            actions, values, log_probs, shield_type = shield.get_action(obs_tensor)
            # print('customppo.get_action: actions, values', actions, values)

        return actions, values, log_probs, shield_type
