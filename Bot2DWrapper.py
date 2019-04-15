import numpy as np
import cv2
from Bot2D.SingleBotLaser2Dgrid import *
from Bot2D.GridMap import *
from Bot2D.MotionModel import *
from Bot2D.Viewer import *
import copy

class Bot2DEnv:
    def __init__(self, 
        # SensorSize, StartAngle, EndAngle, MaxDist, Velocity, Angular
        bot_param = [60, -30.0, 210.0, 120.0, 6.0, 10.0],
        # lo_occ, lo_free, lo_max, lo_min
        map_param = [-0.5, 0.5, 2.0, -2.0],
        # NormalVar, TangentVar, AngularVar
        motion_param = [0.2, 0.1, 0.1], # Simple Model
        #motion_param = [0.01, 0.01, 0.2, 0.2, 0.01, 0.01], # Velocity Model
        obs_size = 64,
        grid_size = 2.,
        map_path = 'Image/map.png'
    ):
        self.bot_param = bot_param
        self.map_param = map_param
        self.motion = SimpleMotionModel(motion_param[0], motion_param[1], motion_param[2])
        self.obs_size = obs_size
        self.grid_size = grid_size
        self.map_path = map_path
    
    def reset(self):
        self.env = SingleBotLaser2D([0,0,0], self.bot_param, self.map_path, self.motion)
        bot_pos = self.env.RandomPos()
        self.map = GridMap(self.map_param, gsize=self.grid_size)
        self.sensor_data = self.env.Sensor()
        SensorMapping(self.map, self.env.bot_pos, self.bot_param, self.sensor_data)
        fsize = self.obs_size
        self.obs = self.map.getObs(self.env.bot_pos,int(fsize/2),int(fsize/2)).reshape([fsize,fsize,1])
        return self.obs, np.array(self.sensor_data, dtype=np.float32)/self.bot_param[3]

    def step(self, action, discrete=True):
        action = action
        collision = self.env.BotAction(action, discrete)
        fsize = self.obs_size
        self.obs = self.map.getObs(self.env.bot_pos,int(fsize/2),int(fsize/2)).reshape([fsize,fsize,1])
        self.sensor_data = self.env.Sensor()
        info_gain = SensorMapping(self.map, self.env.bot_pos, self.bot_param, self.sensor_data)
        return [self.obs, np.array(self.sensor_data, dtype=np.float32)/self.bot_param[3]], info_gain, collision

    def render(self):
        # Initialize OpenCV Windows
        cv2.namedWindow('env', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('obs', cv2.WINDOW_AUTOSIZE)

        # Get Image
        img = DrawEnv(self.env.img_map)
        img = DrawPath(img, self.env.path)
        img = DrawBot(img, self.env.bot_pos, self.sensor_data, self.bot_param)
        mimg = AdaptiveGetMap(self.map)
        fsize = self.obs_size
        cv2.imshow('obs',self.obs)
        cv2.imshow('env',img)
        cv2.imshow('map',mimg)
        cv2.waitKey(1)

    def Image2Map(self, map_path):
        im = cv2.imread(map_path)
        m = np.asarray(im)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = m.astype(float) / 255.
        return m

    def RandInit(self, map_path):
        mm = self.Image2Map(map_path)
        x = mm.shape[1]
        y = mm.shape[0]
        bot_pos = np.array([np.random.randint(x), np.random.randint(y), np.random.randint(360)])

        while mm[bot_pos[1], bot_pos[0]] < 0.9:
            bot_pos = np.array([np.random.randint(x), np.random.randint(y), np.random.randint(360)])
        return bot_pos


DISCRETE = False
RENDER = True
if __name__ == '__main__':
    env = Bot2DEnv()
    for eps in range(10):
        print('[ Episode ' + str(eps)  + ' ]')
        state = env.reset()
        step = 0        
        while True:
            if RENDER:
                env.render()
            if DISCRETE:
                action = np.random.randint(0,4)
            else:
                action = np.random.rand(2) * 2 - 1
            
            state_next, reward, done = env.step(action, DISCRETE)

            print('Episode:', eps, '| Step:', step, '| Action:', action, '| Reward:', reward)
            step += 1
            if done:
                break