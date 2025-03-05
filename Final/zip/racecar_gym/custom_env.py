from .env import RaceEnv
import copy

class CustomEnv(RaceEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prev_info = dict()

    def reset(self, *args, **kwargs: dict):
        obs, info = super().reset(*args, **kwargs)

        self.prev_info = copy.deepcopy(info)
        self.prev_info['motor'] = 0

        return obs, info

    def step(self, actions):
        state, reward, done, truncated, info = super().step(actions)

        reward = 0

        if info['checkpoint'] != self.prev_info['checkpoint']:
            reward += 10

        reward += 1 * info['motor']

        if info['progress'] > self.prev_info['progress']:
            reward += 100 * (info['progress'] - self.prev_info['progress'])
        else:
            reward -= 1.0

        if info['wall_collision']:
            reward = -100
            done = True

        self.prev_info = info.copy()

        return state, reward, done, truncated, info