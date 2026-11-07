"""Collect PPO trajectories on GPU and feed the original RLlib learner."""

import torch
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch


class GPURollout:
    def __init__(self,module,env,horizon,gamma=0.99,gae_lambda=0.95):
        self.module,self.env,self.horizon=module,env,int(horizon)
        self.gamma,self.gae_lambda=gamma,gae_lambda
        self.obs=env.observation()
        shape=(horizon,env.count)
        self.observations=torch.empty((*shape,*env.observation_space.shape),device=env.device)
        self.values=torch.empty(shape,device=env.device)
        self.rewards=torch.empty(shape,device=env.device)
        self.dones=torch.empty(shape,device=env.device,dtype=torch.bool)
        self.actions=torch.empty(shape,device=env.device,dtype=torch.long)
        self.logits=torch.empty((*shape,3),device=env.device)
        self.logps=torch.empty(shape,device=env.device)

    @torch.no_grad()
    def sample(self):
        events=[]
        # One synchronization per rollout, rather than one for every game step.
        possible_timeout=int(self.env.steps.max().item())+self.horizon >= self.env.max_steps
        for step in range(self.horizon):
            output=self.module.forward_train({'obs':self.obs})
            logits=output['action_dist_inputs']
            value=self.module.compute_values({'obs':self.obs},embeddings=output['embeddings'])
            distribution=torch.distributions.Categorical(logits=logits,validate_args=False)
            action=distribution.sample()
            self.observations[step].copy_(self.obs)
            self.values[step].copy_(value)
            self.actions[step].copy_(action)
            self.logits[step].copy_(logits)
            self.logps[step].copy_(distribution.log_prob(action))
            reward,terminated,truncated,info=self.env.step(action)
            if possible_timeout and bool(truncated.any().item()):
                # A time-limit cutoff bootstraps its final state, while collision,
                # starvation and winning have zero bootstrap value.
                terminal_obs=self.env.observation()[truncated]
                terminal_values=self.module.compute_values({'obs':terminal_obs})
                reward=reward.clone()
                reward[truncated]+=self.gamma*terminal_values
            self.rewards[step].copy_(reward)
            done=terminated | truncated
            self.dones[step].copy_(done)
            events.append(torch.stack((done.float(),info['apples'].float(),info['steps'].float(),
                                       info['return'],info['reason'].float()),dim=-1))
            self.env.reset(done)
            self.obs=self.env.observation()

        next_value=self.module.compute_values({'obs':self.obs})
        advantage=torch.zeros(self.env.count,device=self.env.device)
        advantages=torch.empty_like(self.values)
        for step in reversed(range(self.horizon)):
            alive=(~self.dones[step]).float()
            following=self.values[step+1] if step+1 < self.horizon else next_value
            delta=self.rewards[step]+self.gamma*following*alive-self.values[step]
            advantage=delta+self.gamma*self.gae_lambda*alive*advantage
            advantages[step]=advantage
        targets=advantages+self.values
        advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp(min=1e-4)
        count=self.horizon*self.env.count
        batch=SampleBatch({
            'obs':self.observations.reshape(count,*self.env.observation_space.shape),
            'actions':self.actions.reshape(count),
            'action_dist_inputs':self.logits.reshape(count,3),
            'action_logp':self.logps.reshape(count),
            'vf_preds':self.values.reshape(count),
            'advantages':advantages.reshape(count),
            'value_targets':targets.reshape(count),
        })
        recorded=torch.stack(events).reshape(-1,5)
        completed=recorded[recorded[:,0] > 0].cpu().numpy()
        return MultiAgentBatch({'default_policy':batch},env_steps=count),completed
