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
        self.prev_info['steering'] = 0

        return obs, info

    def step(self, actions):
        state, reward, done, truncated, info = super().step(actions)

        reward = 0
        truncated |= (info['time'] >= 100)

        if info['checkpoint'] != self.prev_info['checkpoint']:
            reward += 15

        # reward += 0.1 * motor_action # max: 0.4
        reward += 1 * info['motor']
        reward -= 0.1 * (abs(info['motor'] - self.prev_info['motor']) + abs(
            info['steering'] - self.prev_info['steering']))  # max: 0.2

        if info['progress'] > self.prev_info['progress']:  # move forward
            reward += 1000 * (info['progress'] - self.prev_info['progress'])  # max: 0.6
        elif info['progress'] == self.prev_info['progress']:  # not moving
            reward -= 0.1
        if info['wall_collision']:
            reward = -100
            done = True

        self.prev_info = info.copy()

        return state, reward, done, truncated, info