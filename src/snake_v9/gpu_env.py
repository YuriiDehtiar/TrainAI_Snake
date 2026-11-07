"""Batch Snake rules and observations on a single torch device."""

import torch

from .env import SnakeEnv


class GPUSnakeBatch:
    """Same v9 rules, with body age encoding instead of Python snake lists.

    Each occupied cell stores steps until it vacates: 1=tail, length=head.
    The CPU device is supported for differential tests against SnakeEnv.
    """

    REASONS = ("running", "wall", "body", "starvation", "win", "time_limit")

    def __init__(self, count, config=None, *, device="cuda", seed=42):
        reference = SnakeEnv(config)
        if reference.initial_length != 3:
            raise ValueError("The CUDA batch currently supports initial_length=3.")
        self.settings = reference.settings()
        self.width, self.height = reference.width, reference.height
        self.capacity = reference.capacity
        self.rewards = reference.rewards
        self.max_hunger = reference.max_steps_without_apple
        self.max_steps = reference.max_episode_steps
        self.observation_space = reference.observation_space
        self.action_space = reference.action_space
        reference.close()
        self.count = int(count)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.ids = torch.arange(self.count, device=self.device)
        self.cells = self.width * self.height
        self.positions = torch.arange(self.cells, device=self.device)
        rows, cols = self.positions // self.width, self.positions % self.width
        self.walls = (rows == 0) | (rows == self.height-1) | (cols == 0) | (cols == self.width-1)
        self.directions = torch.tensor(SnakeEnv.DIRECTIONS, device=self.device)
        self.turns = torch.tensor(SnakeEnv.TURNS, device=self.device)

        # Match the CPU reset distribution: uniform head, uniform first
        # neighbor, then uniform non-backtracking second neighbor.
        catalog, directions, weights = [], [], []
        for row in range(1, self.height-1):
            for col in range(1, self.width-1):
                head = (row, col)
                neighbors = [(row+dr, col+dc) for dr,dc in SnakeEnv.DIRECTIONS
                             if 1 <= row+dr < self.height-1 and 1 <= col+dc < self.width-1]
                for neck in neighbors:
                    tails = [(neck[0]+dr, neck[1]+dc) for dr,dc in SnakeEnv.DIRECTIONS
                             if 1 <= neck[0]+dr < self.height-1 and 1 <= neck[1]+dc < self.width-1
                             and (neck[0]+dr, neck[1]+dc) != head]
                    for tail in tails:
                        catalog.append([p[0]*self.width+p[1] for p in (head,neck,tail)])
                        directions.append(SnakeEnv.DIRECTIONS.index((head[0]-neck[0],head[1]-neck[1])))
                        weights.append(1/(self.capacity*len(neighbors)*len(tails)))
        self.catalog = torch.tensor(catalog, device=self.device)
        self.catalog_directions = torch.tensor(directions, device=self.device)
        self.catalog_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.ages = torch.zeros((self.count,self.cells),dtype=torch.long,device=self.device)
        self.head = torch.zeros(self.count,dtype=torch.long,device=self.device)
        self.direction = torch.zeros_like(self.head)
        self.length = torch.full_like(self.head,3)
        self.apple = torch.zeros_like(self.head)
        self.hunger = torch.zeros_like(self.head)
        self.steps = torch.zeros_like(self.head)
        self.apples = torch.zeros_like(self.head)
        self.returns = torch.zeros(self.count,device=self.device)
        self.reset(torch.ones(self.count,dtype=torch.bool,device=self.device))

    def _random_apple(self, ages):
        scores = torch.rand((self.count,self.cells),device=self.device,generator=self.generator)
        scores.masked_fill_((ages > 0) | self.walls[None], -1)
        return scores.argmax(dim=1)

    def reset(self, mask):
        chosen = torch.multinomial(self.catalog_weights,self.count,replacement=True,generator=self.generator)
        snake = self.catalog[chosen]
        spawn = torch.zeros_like(self.ages)
        spawn.scatter_(1,snake,torch.tensor([3,2,1],device=self.device).expand(self.count,3))
        self.ages = torch.where(mask[:,None],spawn,self.ages)
        self.head = torch.where(mask,snake[:,0],self.head)
        self.direction = torch.where(mask,self.catalog_directions[chosen],self.direction)
        self.length = torch.where(mask,3,self.length)
        self.hunger = torch.where(mask,0,self.hunger)
        self.steps = torch.where(mask,0,self.steps)
        self.apples = torch.where(mask,0,self.apples)
        self.returns = torch.where(mask,0.0,self.returns)
        self.apple = torch.where(mask,self._random_apple(self.ages),self.apple)

    def observation(self):
        occupied = self.ages > 0
        head = self.positions[None] == self.head[:,None]
        apple = self.positions[None] == self.apple[:,None]
        neck = self.ages == (self.length-1)[:,None]
        tail = self.ages == 1
        size = (self.count,self.cells)
        channels = [self.walls.expand(size),head,occupied & ~head,apple,neck,tail,
                    self.ages.float()/self.capacity,
                    (self.hunger.float()/self.max_hunger).clamp(max=1)[:,None].expand(size),
                    (self.steps.float()/self.max_steps).clamp(max=1)[:,None].expand(size)]
        channels.extend((self.direction == direction)[:,None].expand(size) for direction in range(4))
        return torch.stack(channels,dim=-1).reshape(self.count,self.height,self.width,13).float()

    def step(self, actions):
        self.direction = (self.direction+self.turns[actions]) % 4
        vector = self.directions[self.direction]
        row = self.head // self.width + vector[:,0]
        col = self.head % self.width + vector[:,1]
        proposed = row*self.width+col
        wall = (row < 1) | (row >= self.height-1) | (col < 1) | (col >= self.width-1)
        eating = proposed == self.apple
        age = self.ages[self.ids,proposed]
        body = (age > 0) & ~((age == 1) & ~eating)
        collision = wall | body
        moving = ~collision
        candidate = (self.ages-(~eating).long()[:,None]).clamp(min=0)
        grown_length = self.length+eating.long()
        candidate.scatter_(1,proposed[:,None],grown_length[:,None])
        self.ages = torch.where(moving[:,None],candidate,self.ages)
        self.head = torch.where(moving,proposed,self.head)
        self.length = torch.where(moving,grown_length,self.length)
        self.steps += 1
        self.hunger = torch.where(moving,torch.where(eating,0,self.hunger+1),self.hunger)
        self.apples += (moving & eating).long()
        win = moving & (self.length == self.capacity)
        starving = moving & ~win & (self.hunger >= self.max_hunger)
        terminated = collision | win | starving
        truncated = (self.steps >= self.max_steps) & ~terminated
        reward = torch.where(eating,self.rewards.apple,self.rewards.step)
        reward = torch.where(collision,self.rewards.collision,reward)
        reward = torch.where(starving,self.rewards.starvation,reward)
        reward = torch.where(win,self.rewards.apple+self.rewards.win_bonus,reward)
        new_apple = self._random_apple(self.ages)
        self.apple = torch.where(moving & eating,new_apple,self.apple)
        self.apple = torch.where(win,-1,self.apple)
        self.returns += reward
        done = terminated | truncated
        reason = torch.zeros_like(self.head)
        reason = torch.where(wall,1,reason)
        reason = torch.where(body & ~wall,2,reason)
        reason = torch.where(starving,3,reason)
        reason = torch.where(win,4,reason)
        reason = torch.where(truncated,5,reason)
        info = {"done":done,"apples":self.apples.clone(),"steps":self.steps.clone(),
                "return":self.returns.clone(),"reason":reason,"win":win}
        return reward,terminated,truncated,info

    def state_dict(self):
        return {"count":self.count,"generator":self.generator.get_state(),
                **{name:getattr(self,name) for name in
                   ("ages","head","direction","length","apple","hunger","steps","apples","returns")}}

    def load_state_dict(self,state):
        if state["count"] != self.count:
            return False
        for name in ("ages","head","direction","length","apple","hunger","steps","apples","returns"):
            setattr(self,name,state[name].to(self.device))
        self.generator.set_state(state["generator"].cpu())
        return True
