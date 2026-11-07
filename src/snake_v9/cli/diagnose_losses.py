"""Replay recorded losses and test alternate decisions without changing training."""

import argparse
from collections import Counter, deque
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from snake_v9.checkpoints import checkpoint_settings
from snake_v9.env import SnakeEnv
from snake_v9.evaluation import load_policy
from snake_v9.io import write_json

ACTION_NAMES = ('straight', 'left', 'right')


def snapshot(env):
    return {'snake': [list(p) for p in env.snake], 'apple': list(env.apple) if env.apple else None,
            'direction': env.direction, 'steps': env.episode_steps,
            'hunger': env.steps_since_apple, 'apples': env.apples,
            'rng': copy.deepcopy(env.np_random.bit_generator.state)}


def restore(settings, state):
    env = SnakeEnv(settings)
    env.snake = [tuple(p) for p in state['snake']]
    env.apple = tuple(state['apple']) if state['apple'] is not None else None
    env.direction = state['direction']
    env.episode_steps, env.steps_since_apple, env.apples = state['steps'], state['hunger'], state['apples']
    env.np_random.bit_generator.state = copy.deepcopy(state['rng'])
    return env


def legal_actions(env):
    legal = []
    for action in range(3):
        direction = (env.direction + env.TURNS[action]) % 4
        dr, dc = env.DIRECTIONS[direction]
        target = (env.snake[0][0]+dr, env.snake[0][1]+dc)
        occupied = env.snake if target == env.apple else env.snake[:-1]
        legal.append(env._inside(target) and target not in occupied)
    return legal


@torch.inference_mode()
def predict(module, obs):
    logits = module.forward_inference({'obs': torch.as_tensor(obs[None], dtype=torch.float32)})['action_dist_inputs'][0]
    return int(logits.argmax()), logits.tolist(), logits.softmax(-1).tolist()


@dataclass(frozen=True)
class SearchState:
    snake: tuple
    direction: int
    apple: object
    hunger: int
    steps: int
    rng: str

    @classmethod
    def from_snapshot(cls, state):
        return cls(tuple(tuple(p) for p in state['snake']), state['direction'],
                   tuple(state['apple']) if state['apple'] else None,
                   state['hunger'], state['steps'], json.dumps(state['rng'], sort_keys=True))


class SurvivalSearch:
    """Bounded exact search with moving tails and the saved apple RNG.

    A positive result proves only the stated survival horizon, not a winning game.
    Future apples use the saved RNG as a reproducible diagnostic counterfactual.
    """

    def __init__(self, settings, node_limit=20000):
        self.width, self.height = settings['width'], settings['height']
        self.capacity = (self.width-2)*(self.height-2)
        self.max_hunger, self.max_steps = settings['max_steps_without_apple'], settings['max_episode_steps']
        self.node_limit = node_limit
        self.nodes = 0
        self.cache = {}

    def advance(self, state, action):
        direction = (state.direction + SnakeEnv.TURNS[action]) % 4
        dr, dc = SnakeEnv.DIRECTIONS[direction]
        target = (state.snake[0][0]+dr, state.snake[0][1]+dc)
        eating = target == state.apple
        occupied = state.snake if eating else state.snake[:-1]
        if not (1 <= target[0] < self.height-1 and 1 <= target[1] < self.width-1) or target in occupied:
            return None, 'collision'
        snake = (target,) + (state.snake if eating else state.snake[:-1])
        if len(snake) == self.capacity:
            return None, 'win'
        hunger = 0 if eating else state.hunger+1
        steps = state.steps+1
        if hunger >= self.max_hunger or steps >= self.max_steps:
            return None, 'limit'
        apple, rng = state.apple, state.rng
        if eating:
            generator = np.random.default_rng()
            generator.bit_generator.state = json.loads(rng)
            occupied = set(snake)
            empty = [(r,c) for r in range(1,self.height-1) for c in range(1,self.width-1) if (r,c) not in occupied]
            apple = empty[int(generator.integers(len(empty)))]
            rng = json.dumps(generator.bit_generator.state, sort_keys=True)
        return SearchState(snake, direction, apple, hunger, steps, rng), None

    def survives(self, state, depth):
        if depth == 0:
            return True
        key = (state,depth)
        if key in self.cache:
            return self.cache[key]
        self.nodes += 1
        if self.nodes > self.node_limit:
            return None
        unknown = False
        for action in range(3):
            following, terminal = self.advance(state,action)
            if terminal == 'win':
                self.cache[key] = True
                return True
            if terminal:
                continue
            answer = self.survives(following,depth-1)
            if answer is True:
                self.cache[key] = True
                return True
            unknown |= answer is None
        answer = None if unknown else False
        if answer is not None:
            self.cache[key] = answer
        return answer

    def action_results(self, snapshot_, depth):
        self.nodes, self.cache = 0, {}
        state = SearchState.from_snapshot(snapshot_)
        results = []
        for action in range(3):
            following, terminal = self.advance(state,action)
            results.append(True if terminal == 'win' else False if terminal else self.survives(following,depth-1))
        return results


def counterfactual(module, settings, frame, forced_action):
    env = restore(settings,frame['state'])
    obs, reward, terminated, truncated, info = env.step(forced_action)
    decisions = 1
    while not (terminated or truncated):
        action, _, _ = predict(module, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        decisions += 1
    return {'forced_action':forced_action, 'forced_action_name':ACTION_NAMES[forced_action],
            'subsequent_policy':'unchanged deterministic argmax',
            'further_steps':decisions, 'final_info':info}


def render_cases(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    settings = checkpoint_settings(Path(summary['checkpoint']))
    width,height = settings['width'],settings['height']
    available = {}
    for path in sorted(directory.glob('loss_*.json')):
        case = json.loads(path.read_text(encoding='utf-8'))
        if case['critical_frame_index'] is not None and any(p['final_info']['won'] for p in case['counterfactuals']):
            available[case['seed']] = case
    selected = [seed for seed in (60054,60018,60084,60094) if seed in available]
    selected.extend(seed for seed in available if seed not in selected)
    if not selected:
        return
    selected = selected[:4]
    names = ('Прямо','Ліворуч','Праворуч')
    fig,axes = plt.subplots(2,2,figsize=(12,12))
    model = Path(summary['checkpoint']).name
    fig.suptitle(f'Причини поразок SnakeAI v9 · {model}',fontsize=17,y=0.99)
    fig.text(0.5,0.95,'Синя — голова · жовта — хвіст · червоне коло — яблуко',ha='center',fontsize=11)
    for axis in list(axes.flat)[len(selected):]:
        axis.set_visible(False)
    for axis,seed in zip(axes.flat,selected):
        case = available[seed]
        frame = case['frames'][case['critical_frame_index']]
        state = frame['state']
        title = 'Помилка перед зіткненням'
        if case['final_info']['reason'] == 'wall':
            title = 'Зіткнення зі стіною'
        elif case['final_info']['apples'] >= (width-2)*(height-2)-5:
            title = 'Поразка близько до перемоги'
        elif case['critical_step'] < case['final_info']['steps']:
            title = 'Небезпечне рішення перед пасткою'
        elif case['legal_actions_at_death']:
            title = 'Зіткнення попри безпечний вихід'
        board = np.ones((height,width,3))*0.97
        board[0,:,:] = board[-1,:,:] = 0.17
        board[:,0,:] = board[:,-1,:] = 0.17
        for index,(row,col) in enumerate(state['snake']):
            strength = 0.35+0.55*(1-index/len(state['snake']))
            board[row,col] = (0.35,1-strength*0.5,0.44)
        tail = state['snake'][-1]
        board[tail[0],tail[1]] = to_rgb('#f4bc34')
        head = state['snake'][0]
        board[head[0],head[1]] = to_rgb('#2563eb')
        axis.imshow(board)
        if state['apple']:
            axis.scatter(state['apple'][1],state['apple'][0],s=110,c='#ed334e',edgecolors='white',linewidths=0.7)
        axis.set_xticks(np.arange(-0.5,width,1),minor=True)
        axis.set_yticks(np.arange(-0.5,height,1),minor=True)
        axis.grid(which='minor',color='white',linewidth=0.55,alpha=0.5)
        axis.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
        alternate = next(probe['forced_action'] for probe in case['counterfactuals'] if probe['final_info']['won'])
        for action,color in ((frame['action'],'#dc2626'),(alternate,'#078b60')):
            direction = (state['direction']+SnakeEnv.TURNS[action])%4
            dr,dc = SnakeEnv.DIRECTIONS[direction]
            axis.annotate('',xy=(head[1]+0.92*dc,head[0]+0.92*dr),
                          xytext=(head[1]+0.12*dc,head[0]+0.12*dr),
                          arrowprops={'arrowstyle':'-|>','color':color,'lw':3,'mutation_scale':18})
            if not frame['legal'][action]:
                axis.plot(head[1]+dc,head[0]+dr,marker='x',markersize=13,markeredgewidth=2.6,color=color)
        axis.set_title(f"{title}\nПартія {seed} · хід {state['steps']+1} · яблук {state['apples']}",fontsize=12,pad=10)
        probabilities = '  |  '.join(f'{name}: {100*p:.2f}%' for name,p in zip(names,frame['probabilities']))
        axis.text(0.5,-0.07,probabilities,transform=axis.transAxes,ha='center',fontsize=9)
        axis.text(0.5,-0.13,f"Червона: {names[frame['action']]} · зелена: {names[alternate]} → перемога",
                  transform=axis.transAxes,ha='center',fontsize=10)
    fig.text(0.5,0.015,'Стрілки: вибір мережі та перевірений альтернативний хід. Далі працює та сама мережа.',
             ha='center',fontsize=10)
    fig.tight_layout(rect=(0,0.06,1,0.92),h_pad=5,w_pad=2)
    fig.savefig(directory/'examples.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--context',type=int,default=40)
    parser.add_argument('--lookahead',type=int,default=12)
    parser.add_argument('--mask-test',action='store_true')
    args = parser.parse_args()
    if min(args.context,args.lookahead) < 1:
        parser.error('Context and lookahead must be positive.')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    module = load_policy(args.checkpoint).to('cpu')
    settings = checkpoint_settings(args.checkpoint)
    evaluation = json.loads(args.evaluation.read_text(encoding='utf-8'))
    losses = [row for row in evaluation['rows'] if not row['won']]
    details = []
    search = SurvivalSearch(settings)
    for expected in losses:
        env = SnakeEnv(settings)
        obs, _ = env.reset(seed=expected['seed'])
        frames = deque(maxlen=args.context)
        while True:
            action, logits, probabilities = predict(module,obs)
            frame = {'state':snapshot(env),'action':action,'action_name':ACTION_NAMES[action],
                     'probabilities':probabilities,'logits':logits,'legal':legal_actions(env)}
            frames.append(frame)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert all(info[key] == expected[key] for key in ('steps','apples','reason','won')), (info,expected)
        frames = list(frames)
        for frame in frames:
            # The independent transition search must agree with the real environment.
            for action in range(3):
                clone = restore(settings,frame['state'])
                _, _, term, trunc, actual = clone.step(action)
                predicted, end = search.advance(SearchState.from_snapshot(frame['state']),action)
                assert (end is not None) == (term or trunc), (actual,end)
                if predicted is not None:
                    assert predicted == SearchState.from_snapshot(snapshot(clone))
        last = frames[-1]
        critical = None
        for frame_index in range(len(frames)-1,-1,-1):
            frame = frames[frame_index]
            outcomes = search.action_results(frame['state'],args.lookahead)
            frame['survives_horizon'] = outcomes
            frame['search_nodes'] = search.nodes
            if outcomes[frame['action']] is False and any(outcome is True for outcome in outcomes):
                critical = frame_index
                break
        alternatives = []
        if critical is not None:
            frame = frames[critical]
            for alternative in range(3):
                if alternative != frame['action'] and frame['survives_horizon'][alternative] is True:
                    alternatives.append(counterfactual(module,settings,frame,alternative))
        case = {'seed':expected['seed'],'final_info':info,
                'legal_actions_at_death':sum(last['legal']),
                'death_action_probability':last['probabilities'][last['action']],
                'critical_frame_index':critical,
                'critical_step':frames[critical]['state']['steps']+1 if critical is not None else None,
                'steps_from_critical_to_collision':info['steps']-frames[critical]['state']['steps'] if critical is not None else None,
                'counterfactuals':alternatives, 'frames':frames}
        details.append(case)
        write_json(args.output/f"loss_{expected['seed']}.json",case)
        print(json.dumps({key:value for key,value in case.items() if key != 'frames'}),flush=True)
    summary = {'checkpoint':str(args.checkpoint.resolve()),'baseline_episodes':evaluation['episodes'],
               'baseline_wins':sum(row['won'] for row in evaluation['rows']),
               'losses':len(details),'reasons':dict(Counter(case['final_info']['reason'] for case in details)),
               'no_legal_action_at_death':sum(case['legal_actions_at_death'] == 0 for case in details),
               'legal_alternative_at_death':sum(case['legal_actions_at_death'] > 0 for case in details),
               'critical_decision_found':sum(case['critical_frame_index'] is not None for case in details),
               'lookahead_depth':args.lookahead,
               'single_change_then_original_policy_wins':sum(any(probe['final_info']['won'] for probe in case['counterfactuals']) for case in details),
               'losses_after_50_apples':sum(case['final_info']['apples'] >= 50 for case in details),
               'cases':[{key:value for key,value in case.items() if key != 'frames'} for case in details]}
    write_json(args.output/'summary.json',summary)
    if args.mask_test:
        print('MASK TEST: same seeds; prevent only immediately colliding actions',flush=True)
        rows = []
        interventions = 0
        for index,expected in enumerate(evaluation['rows']):
            env = SnakeEnv(settings)
            obs,_ = env.reset(seed=expected['seed'])
            changes = 0
            while True:
                action,logits,_ = predict(module,obs)
                legal = legal_actions(env)
                if not legal[action] and any(legal):
                    action = max((a for a in range(3) if legal[a]),key=lambda a:logits[a])
                    changes += 1
                obs,_,terminated,truncated,info = env.step(action)
                if terminated or truncated:
                    break
            if not changes:
                assert all(info[key] == expected[key] for key in ('steps','apples','reason','won'))
            rows.append({**info,'seed':expected['seed'],'interventions':changes})
            interventions += changes
            if (index+1) % 20 == 0:
                print(f'MASK TEST: {index+1}/{len(evaluation["rows"])}',flush=True)
        summary['immediate_mask_test'] = {'episodes':len(rows),'wins':sum(row['won'] for row in rows),
                                          'apples_mean':statistics.mean(row['apples'] for row in rows),
                                          'reasons':dict(Counter(row['reason'] for row in rows)),
                                          'interventions':interventions,
                                          'episodes_with_interventions':sum(row['interventions'] > 0 for row in rows)}
        write_json(args.output/'mask_test.json',{'summary':summary['immediate_mask_test'],'rows':rows})
        write_json(args.output/'summary.json',summary)
    print('SUMMARY: '+json.dumps({key:value for key,value in summary.items() if key != 'cases'}),flush=True)
    render_cases(args.output)


if __name__ == '__main__':
    main()
