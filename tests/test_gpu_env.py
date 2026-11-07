import unittest

import numpy as np
import torch

from snake_v9.env import SnakeEnv
from snake_v9.gpu_env import GPUSnakeBatch


class BatchedRulesTests(unittest.TestCase):
    def copy_to_cpu(self,batch,index,cpu):
        ages=batch.ages[index].cpu().numpy()
        cells=np.flatnonzero(ages)
        cells=sorted(cells,key=lambda cell:-ages[cell])
        cpu.snake=[(int(cell)//cpu.width,int(cell)%cpu.width) for cell in cells]
        apple=int(batch.apple[index])
        cpu.apple=(apple//cpu.width,apple%cpu.width) if apple >= 0 else None
        cpu.direction=int(batch.direction[index])
        cpu.steps_since_apple=int(batch.hunger[index])
        cpu.episode_steps=int(batch.steps[index])
        cpu.apples=int(batch.apples[index])
        cpu._finished=False

    def test_batched_transitions_match_existing_rules(self):
        batch=GPUSnakeBatch(4,device='cpu',seed=123)
        cpu=[SnakeEnv() for _ in range(4)]
        for index,env in enumerate(cpu):
            self.copy_to_cpu(batch,index,env)
        rng=np.random.default_rng(100)
        for step in range(300):
            actions=rng.integers(0,3,size=4)
            reward,terminated,truncated,info=batch.step(torch.as_tensor(actions))
            obs=batch.observation().numpy()
            for index,env in enumerate(cpu):
                # Inject the same randomly placed apple before the CPU step;
                # placement after eating is independent but must remain empty.
                cpu_obs,cpu_reward,cpu_done,cpu_truncated,cpu_info=env.step(int(actions[index]))
                self.assertAlmostEqual(float(reward[index]),cpu_reward,places=6)
                self.assertEqual(bool(terminated[index]),cpu_done)
                self.assertEqual(bool(truncated[index]),cpu_truncated)
                self.assertEqual(int(info['apples'][index]),cpu_info['apples'])
                self.assertEqual(batch.REASONS[int(info['reason'][index])],cpu_info['reason'])
                # Synchronize only the independently randomized next apple.
                apple=int(batch.apple[index])
                env.apple=(apple//env.width,apple%env.width) if apple >= 0 else None
                np.testing.assert_allclose(obs[index],env._observation(),atol=1e-7,rtol=1e-6)
            batch.reset(terminated | truncated)
            for index,env in enumerate(cpu):
                if terminated[index] or truncated[index]:
                    self.copy_to_cpu(batch,index,env)
        for env in cpu:
            env.close()

    def test_starvation_and_time_limit_match(self):
        for timeout in (False,True):
            batch=GPUSnakeBatch(1,device='cpu')
            cpu=SnakeEnv()
            cpu.snake=[(4,4),(4,3),(4,2)]
            cpu.direction=1
            cpu.apple=(2,2)
            cpu._finished=False
            batch.ages.zero_()
            batch.ages[0,44]=3; batch.ages[0,43]=2; batch.ages[0,42]=1
            batch.head[0]=44; batch.direction[0]=1; batch.apple[0]=22
            if timeout:
                batch.steps[0]=batch.max_steps-1
                cpu.episode_steps=cpu.max_episode_steps-1
            else:
                batch.hunger[0]=batch.max_hunger-1
                cpu.steps_since_apple=cpu.max_steps_without_apple-1
            reward,done,truncated,_=batch.step(torch.tensor([0]))
            _,expected_reward,expected_done,expected_truncated,_=cpu.step(0)
            self.assertEqual(bool(done[0]),expected_done)
            self.assertEqual(bool(truncated[0]),expected_truncated)
            self.assertAlmostEqual(float(reward[0]),expected_reward,places=6)
            cpu.close()

    def test_full_board_is_win(self):
        batch=GPUSnakeBatch(1,{'width':5,'height':5},device='cpu')
        snake=[(1,2),(1,1),(2,1),(3,1),(3,2),(3,3),(2,3),(2,2)]
        batch.ages.zero_()
        for index,(row,col) in enumerate(snake):
            batch.ages[0,row*5+col]=len(snake)-index
        batch.head[0]=7; batch.direction[0]=1; batch.apple[0]=8; batch.length[0]=8
        reward,done,truncated,info=batch.step(torch.tensor([0]))
        self.assertTrue(bool(done[0]))
        self.assertFalse(bool(truncated[0]))
        self.assertTrue(bool(info['win'][0]))
        self.assertEqual(float(reward[0]),6)
        self.assertEqual(int(batch.apple[0]),-1)


if __name__ == '__main__':
    unittest.main()
