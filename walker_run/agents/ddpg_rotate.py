
import hydra
import copy
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core import DeterministicActor, DDPGCritic
import utils.utils as utils


class DDPGAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, linear_approx, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip,
                 clipped_noise, aug_ratio):

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.clipped_noise = clipped_noise
        self.stddev_clip = stddev_clip
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.aug_ratio = aug_ratio

        # models
        self.actor = DeterministicActor(obs_shape, action_shape[0],
                                        hidden_dim, linear_approx).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = DDPGCritic(obs_shape, action_shape[0],
                                 hidden_dim, linear_approx).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.actor_target.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        stddev = utils.schedule(self.stddev_schedule, step)
        action = self.actor(obs.float().unsqueeze(0))
        if eval_mode:
            action = action.cpu().numpy()[0]
        else:
            action = action.cpu().numpy()[0] + np.random.normal(0, stddev, size=self.action_dim)
            if step < self.num_expl_steps:
                action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        return action.astype(np.float32)

    def observe(self, obs, action):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).float().unsqueeze(0)

        q = self.critic(obs, action)

        return {
            'state': obs.cpu().numpy()[0],
            'value': q.cpu().numpy()[0]
        }

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            if self.clipped_noise:
                # Select action according to policy and add clipped noise
                stddev = utils.schedule(self.stddev_schedule, step)
                noise = (torch.randn_like(action) * stddev).clamp(-self.stddev_clip, self.stddev_clip)

                next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            else:
                next_action = self.actor_target(next_obs)

            # Compute the target Q value
            target_Q = self.critic_target(next_obs, next_action)
            target_Q = reward + discount * target_Q

        # Get current Q estimates
        current_Q = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q'] = current_Q.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        # Compute actor loss
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()

        return metrics
    
    
    def rotation_matirx(self, alphas):
        c, s = np.cos(alphas), np.sin(alphas)
        bz = len(c)
        R = torch.zeros((bz, 3, 3))
        R[:,0,0] = c
        R[:,0,1] = -s
        R[:,1,0] = s
        R[:,1,1] = c
        R[:,2,2] = 1
        
        return R.cuda()

    def roto_translate_vector_3d(self, vector_3d, R, trans):
        #R = self.rotation_matirx(alphas)
        transformed_vector_3d = torch.bmm(R, vector_3d)
        transformed_vector_3d = torch.permute(transformed_vector_3d, (0,2,1))
        #return torch.t(R@(torch.t(vector_3d))) + trans
        return transformed_vector_3d + trans
        
    def roto_translate(self, obs, action, next_obs, alphas):

        bz = len(obs)
        R = self.rotation_matirx(alphas)
        
        touch = obs[:,-1:]
        obs = obs[:,:-1]
        
        obs = obs.view(bz, -1, 3)
        obs = torch.permute(obs, (0,2,1))
        rotated_obs = self.roto_translate_vector_3d(obs, R, 0)
        rotated_obs = rotated_obs.reshape(bz, -1)
        rotated_obs = torch.cat((rotated_obs, touch), dim=1)
        
        
        next_touch = next_obs[:,-1:]
        next_obs = next_obs[:,:-1]
        next_obs = next_obs.view(bz, -1, 3)
        next_obs = torch.permute(next_obs, (0,2,1))
        rotated_next_obs = self.roto_translate_vector_3d(next_obs, R, 0)
        rotated_next_obs = rotated_next_obs.reshape(bz, -1)
        rotated_next_obs = torch.cat((rotated_next_obs, next_touch), dim=1)
        
        
        rotated_action = torch.clone(action)

        return rotated_obs, rotated_action, rotated_next_obs
        
    def get_aug_data(self, obs, action, next_obs):
        bz = len(obs)
        alphas = torch.rand(bz) * 360
        
        mask = torch.zeros(bz)
        if self.aug_ratio == 0:
            mask[:] = 0
        elif self.aug_ratio == 1:
            mask[:] = 1
        elif self.aug_ratio == 2: #50
            mask[:bz // 2] = 1
        elif self.aug_ratio == 3: #75
            mask[bz // 4:] = 1
        elif self.aug_ratio == 4: #25
            mask[:bz // 4] = 1
        alphas *= mask
        
        alphas = torch.deg2rad(alphas)
        rotated_obs, rotated_action, rotated_next_obs = self.roto_translate(obs, action, next_obs, alphas)
    
        return rotated_obs, rotated_action, rotated_next_obs

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _, eq_state, next_eq_state = utils.to_torch(batch, self.device)
        
        eq_state = eq_state.float()
        next_eq_state = next_eq_state.float()
#        obs = obs.float()
#        next_obs = next_obs.float()

        eq_state, action, next_eq_state = self.get_aug_data(torch.clone(eq_state), torch.clone(action), torch.clone(next_eq_state))
        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(eq_state, action, reward, discount, next_eq_state, step))

        # update actor (delayed)
        if step % self.update_every_steps == 0:
            metrics.update(self.update_actor(eq_state.detach(), step))

            # update target networks
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
            utils.soft_update_params(self.actor, self.actor_target, self.critic_target_tau)

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        self.actor.load_state_dict(
            torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f'{model_load_dir}/critic.pt', map_location=self.device)
        )
