import gym
from gym.utils import seeding
import numpy as np

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


class MyPendulumEnv(gym.Env):
    def __init__(self, g=9.82,max_torque=5):
        self.max_speed = 20
        self.max_torque = max_torque
        self.dt = .1
        self.g = g
        self.m = 1.
        self.l = 1.
        self.b = 0.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # One initial state for simplicity
    def reset(self):
        self.state = np.array([0., 0.])
        self.last_u = None
        return self._get_obs()

    # Make the pendulum more realistic (add damping)
    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        b = self.b
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        dx = (u - b * thdot - m * g * l * np.sin(th) / 2.) / (m * l * l / 3.)
        newthdot = thdot + dx * dt
        newth = th + newthdot * dt
        # newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        s_c_sq = 0.5 * 0.5
        dcos = np.cos(newth) - np.cos(np.pi)
        dsin = np.sin(newth) - np.sin(np.pi)
        derr = dcos * dcos + dsin * dsin

        return self._get_obs(), np.exp(-0.5*derr/s_c_sq), False, {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            # self.img = rendering.Image(fname, 1., 1.)
            # self.imgtrans = rendering.Transform()
            # self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(angle_normalize(self.state[0] - np.pi / 2.))
        # if self.last_u:
        #     self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

