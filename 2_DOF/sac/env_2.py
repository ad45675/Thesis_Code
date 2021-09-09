import numpy as np
import math
import pyglet
import time
import config

class ArmEnv(object):

    viewer = None
    arm1_bound = [-np.pi/2, np.pi/2]
    arm2_bound = [-np.pi, np.pi]
    state_dim = 6  # end-effector ( u, v ), target ( u, v ), q1 q2
    action_dim = 2

    def __init__(self, mode):
        self.mode = mode

        self.on_goal = 0
        # goal
        self.goal = np.zeros((2, 1))  # ( u, v )
        self.width_arm = 66.5
        self.distance = 0
        self.inf = [0, 0, 0]  # EPISODE STEP distance

        self.m_arm = np.zeros((1, 2))
        self.b_arm = np.zeros((1, 2))
        self.b_arm_L = np.zeros((1, 2))
        self.b_arm_R = np.zeros((1, 2))
        self.b_arm_U = np.zeros((1, 2))
        self.b_arm_D = np.zeros((1, 2))
        self.vertices_arm1 = np.zeros((2, 4))
        self.vertices_arm2 = np.zeros((2, 4))

        # robot parameter
        self.L1 = 225
        self.L2 = 230
        self.q = np.zeros((2, 1))  # 2*1 [q1 q2]
        self.EEF_arm0_robot = np.zeros((3, 1))  # 3*1 ( Xr , Yr , Zr )
        self.EEF_arm1_robot = np.zeros((3, 1))  # 3*1 ( Xr , Yr , Zr )
        self.EEF_arm2_robot = np.zeros((3, 1))  # 3*1 ( Xr , Yr , Zr )
        self.EEF_arm0_camera = np.zeros((3, 1))  # 3*1 ( Xc , Yc , Zc )
        self.EEF_arm1_camera = np.zeros((3, 1))  # 3*1 ( Xc , Yc , Zc )
        self.EEF_arm2_camera = np.zeros((3, 1))  # 3*1 ( Xc , Yc , Zc )
        self.EEF_arm0_image = np.zeros((2, 1))  # 2*1 ( u, v )
        self.EEF_arm1_image = np.zeros((2, 1))  # 2*1 ( u, v )
        self.EEF_arm2_image = np.zeros((2, 1))  # 2*1 ( u, v )

        self.EEF_arm0_robot[2, 0] = 10

        # Intrisic matix
        self.matrix_camera = np.array([[1.5627044545649730 * math.pow(10, 3), 0, 2.8477025480870020 * math.pow(10, 2)],
                                       [0, 1.5637307281700662 * math.pow(10, 3), 2.8292536344105076 * math.pow(10, 2)],
                                       [0, 0, 1]])
        self.pinvmatrix_camera = np.zeros((3, 3))
        self.pinvmatrix_camera = np.linalg.inv(self.matrix_camera)

        # hand-eye Transformation matrix  -- > camera frame to robot base frame 6*6
        self.SbRc = np.zeros((3, 3))
        self.Sbtc = np.zeros((3, 3))
        self.bTc = np.zeros((6, 6))

        # translate robot base frame to camera frame 新校正的
        self.cRb = np.array([[-1.1856740449004755 * math.pow(10, -2), -9.9992970638236611 * math.pow(10, -1),
                              -3.0814879110195774 * math.pow(10, -33)],
                             [9.9992970638236622 * math.pow(10, -1), -1.1856740449004699 * math.pow(10, -2),
                              -3.0814879110195774 * math.pow(10, -33)],
                             [0, 0, 1]])  # 3*3

        self.ctb = np.array([[2.8491526330161156 * math.pow(10, 2)],
                             [-3.1426269556327304 * math.pow(10, 2)],
                             [1320]])  # 3*1

        self.bRc = np.linalg.inv(self.cRb)  # 3*3
        self.btc = -np.dot(self.bRc, self.ctb)  # 3*1

        #  skew matrix
        self.Sbtc[0, 0] = 0.0
        self.Sbtc[0, 1] = -1 * self.btc[2, 0]
        self.Sbtc[0, 2] = self.btc[1, 0]
        self.Sbtc[1, 0] = self.btc[2, 0]
        self.Sbtc[1, 1] = 0.0
        self.Sbtc[1, 2] = -1 * self.btc[0, 0]
        self.Sbtc[2, 0] = -1 * self.btc[1, 0]
        self.Sbtc[2, 1] = self.btc[0, 0]
        self.Sbtc[2, 2] = 0.0

        self.SbRc = np.dot(self.Sbtc, self.bRc)  # 3*3

        # spacial transform matrix
        self.bTc[0, 0:3] = self.bRc[0, 0:3]
        self.bTc[1, 0:3] = self.bRc[1, 0:3]
        self.bTc[2, 0:3] = self.bRc[2, 0:3]

        self.bTc[3, 3:6] = self.bRc[0, 0:3]
        self.bTc[4, 3:6] = self.bRc[1, 0:3]
        self.bTc[5, 3:6] = self.bRc[2, 0:3]

        self.bTc[0, 3:6] = self.SbRc[0, 0:3]
        self.bTc[1, 3:6] = self.SbRc[1, 0:3]
        self.bTc[2, 3:6] = self.SbRc[2, 0:3]

    def step(self, action, inf):

        # clip action
        action[0] = np.clip(action[0], *self.arm1_bound)
        action[1] = np.clip(action[1], *self.arm2_bound)

        self.q[0, 0] = action[0]
        self.q[1, 0] = action[1]

        self.Transformation()

        difference = [(self.goal[0, 0] - self.EEF_arm2_image[0, 0]), (self.goal[1, 0] - self.EEF_arm2_image[1, 0])]
        difference_normalize = [difference[0]/640, difference[1]/512]  # image size: 640*512

        distance = np.sqrt(pow(difference[0], 2) + pow(difference[1], 2))
        self.distance = distance
        distance_normalize = np.sqrt(pow(difference_normalize[0], 2) + pow(difference_normalize[1], 2))

        # reward
        r = -distance_normalize
        if distance <= config.Tolerance:
            r += 1
            done = True
        else:
            done = False

        # for draw
        self.inf = [inf[0], inf[1], distance]

        EEF_normalize = [self.EEF_arm2_image[0, 0]/640, self.EEF_arm2_image[1, 0]/512]
        goal_normalize = [self.goal[0, 0]/640, self.goal[1, 0]/512]
        q_normalize = [self.q[0, 0]/self.arm1_bound[1], self.q[1, 0]/self.arm2_bound[1]]

        # state
        s = np.concatenate((EEF_normalize, goal_normalize, q_normalize))
        return s, r, done

    def Transformation(self):

        # Forward kinematic
        # arm1
        x_arm1 = self.L1 * np.cos(self.q[0, 0])
        y_arm1 = self.L1 * np.sin(self.q[0, 0])
        # arm2
        x_arm2 = self.L1 * np.cos(self.q[0, 0]) + self.L2 * np.cos(self.q[0, 0] + self.q[1, 0])
        y_arm2 = self.L1 * np.sin(self.q[0, 0]) + self.L2 * np.sin(self.q[0, 0] + self.q[1, 0])
        z = 10  #
        self.EEF_arm1_robot[0, 0] = x_arm1
        self.EEF_arm1_robot[1, 0] = y_arm1
        self.EEF_arm1_robot[2, 0] = z

        self.EEF_arm2_robot[0, 0] = x_arm2
        self.EEF_arm2_robot[1, 0] = y_arm2
        self.EEF_arm2_robot[2, 0] = z

        # handeye Transform
        # original point
        self.EEF_arm0_camera, self.EEF_arm0_image = self.robotToimage(self.EEF_arm0_robot)
        # arm1
        self.EEF_arm1_camera, self.EEF_arm1_image = self.robotToimage(self.EEF_arm1_robot)
        # arm2
        self.EEF_arm2_camera, self.EEF_arm2_image = self.robotToimage(self.EEF_arm2_robot)

    def robotToimage(self, coodinate_robot):

        coodinate_camera = np.dot(self.cRb, coodinate_robot) + self.ctb
        coodinate_image_ = (1 / 1330) * np.dot(self.matrix_camera, coodinate_camera) # [u v z]
        coodinate_image = np.zeros((2, 1))
        coodinate_image[0, 0] = coodinate_image_[0, 0]
        coodinate_image[1, 0] = coodinate_image_[1, 0]

        return coodinate_camera, coodinate_image

    def arm_vertices(self):
        d = self.width_arm / 2  # 手臂寬度/2
        P_arm = np.array([[self.EEF_arm0_image[0, 0], self.EEF_arm0_image[1, 0]],
                          [self.EEF_arm1_image[0, 0], self.EEF_arm1_image[1, 0]],
                          [self.EEF_arm2_image[0, 0], self.EEF_arm2_image[1, 0]]])


        # L_arm1 --> m_arm1*x-y+b_arm1 = 0
        # L_arm2 --> m_arm2*x-y+b_arm2 = 0
        for i in range(np.shape(self.m_arm)[1]):
            self.m_arm[0, i] = (P_arm[i + 1, 1] - P_arm[i, 1]) / (P_arm[i + 1, 0] - P_arm[i, 0])
            self.b_arm[0, i] = P_arm[i + 1, 1] - self.m_arm[0, i] * P_arm[i + 1, 0]

        # L_arm1_L --> m_arm1*x-y+b_arm1_L = 0   L_arm1_R --> m_arm1*x-y+b_arm1_R = 0
        # L_arm2_L --> m_arm2*x-y+b_arm2_L = 0   L_arm2_R --> m_arm2*x-y+b_arm2_R = 0
        for i in range(np.shape(self.b_arm_L)[1]):
            # m^2 + (-1)^2
            base = d * math.sqrt(pow(self.m_arm[0, i], 2) + pow(-1, 2))
            if self.m_arm[0, i] >= 0:
                self.b_arm_L[0, i] = base + self.b_arm[0, i]
                self.b_arm_R[0, i] = -base + self.b_arm[0, i]
            else:
                self.b_arm_L[0, i] = -base + self.b_arm[0, i]
                self.b_arm_R[0, i] = base + self.b_arm[0, i]

        # L_arm1_U --> -(1/m_arm1)*x-y+b_arm1_U = 0   L_arm1_U --> -(1/m_arm1)*x-y+b_arm1_D = 0
        # L_arm2_D --> -(1/m_arm2)*x-y+b_arm2_U = 0   L_arm2_D --> -(1/m_arm2)*x-y+b_arm2_D = 0
        for i in range(np.shape(self.b_arm_U)[1]):
            self.b_arm_U[0, i] = (1 / self.m_arm[0, i]) * P_arm[i, 0] + P_arm[i, 1]
            self.b_arm_D[0, i] = (1 / self.m_arm[0, i]) * P_arm[i + 1, 0] + P_arm[i + 1, 1]


        for i in range(2):
            Line = np.array([[self.m_arm[0, i], -1, -self.b_arm_L[0, i]],
                             [(-1/self.m_arm[0, i]), -1, -self.b_arm_U[0, i]],
                             [self.m_arm[0, i], -1, -self.b_arm_R[0, i]],
                             [(-1/self.m_arm[0, i]), -1, -self.b_arm_D[0, i]],
                             [self.m_arm[0, i], -1, -self.b_arm_L[0, i]]])
            if i == 0:
                for j in range(4):
                    self.vertices_arm1[0, j], self.vertices_arm1[1, j] = self.simultaneous_equations_sol(Line[j], Line[j+1])
            else:
                for j in range(4):
                    self.vertices_arm2[0, j], self.vertices_arm2[1, j] = self.simultaneous_equations_sol(Line[j], Line[j+1])

    def simultaneous_equations_sol(self, L1, L2):
        # 寫出係數矩陣 A
        A = np.array([
            [L1[0], L1[1]],
            [L2[0], L2[1]]
        ])
        # 寫出常數矩陣 B
        B = np.array([L1[2], L2[2]]).reshape(2, 1)
        # 找出係數矩陣的反矩陣 A_inv
        A_inv = np.linalg.inv(A)
        # 將 A_inv 與 B 相乘，即可得到解答
        ans = A_inv.dot(B)
        return ans[0, 0], ans[1, 0]

    def reset(self):
        # random goal
        while True:
            u = np.random.uniform(0, 640)
            v = np.random.uniform(40, 512)
            goal = np.array([[u], [v], [1]])
            cP = np.dot(self.pinvmatrix_camera, 1330*goal)
            bP = np.dot(self.bRc, cP) + self.btc
            flag = self.robotarm_limit(bP)
            if flag:
                self.goal[0, 0] = goal[0, 0]
                self.goal[1, 0] = goal[1, 0]
                break

        q1 = np.random.uniform(self.arm1_bound[0], self.arm1_bound[1])
        q2 = np.random.uniform(self.arm2_bound[0], self.arm2_bound[1])
        self.q[0, 0] = q1
        self.q[1, 0] = q2
        self.Transformation()

        EEF_normalize = [self.EEF_arm2_image[0, 0] / 640, self.EEF_arm2_image[1, 0] / 512]
        goal_normalize = [self.goal[0, 0]/640, self.goal[1, 0]/512]
        q_normalize = [self.q[0, 0] / self.arm1_bound[1], self.q[1, 0] / self.arm2_bound[1]]

        # state
        s = np.concatenate((EEF_normalize, goal_normalize,q_normalize))
        return s

    def robotarm_limit(self, bP):

        x = bP[0, 0]
        y = bP[1, 0]
        test = (math.pow(x, 2) + math.pow(y, 2)-math.pow(self.L1, 2)-math.pow(self.L2, 2))/(2*self.L1*self.L2)
        if abs(test) > 1:  #表示機械手臂到達不了，重新產生u,v
            flag = 0
        else:
            flag = 1
        return flag

    def render(self):
        self.arm_vertices()
        if self.viewer is None:
            self.viewer = Viewer(self.EEF_arm1_image, self.EEF_arm2_image, self.vertices_arm1, self.vertices_arm2,
                                 self.goal, self.inf, self.mode)
        self.viewer.render(self.EEF_arm1_image, self.EEF_arm2_image, self.vertices_arm1, self.vertices_arm2, self.goal,
                           self.inf)

    def sample_action(self):
        return np.random.rand(2)    # two radians

    def reset_validation(self):
        # random goal
        while True:
            u = np.random.uniform(0, 640)
            v = np.random.uniform(40, 512)
            goal = np.array([[u], [v], [1]])
            cP = np.dot(self.pinvmatrix_camera, 1330*goal)
            bP = np.dot(self.bRc, cP) + self.btc
            flag = self.robotarm_limit(bP)
            if flag:
                self.goal[0, 0] = goal[0, 0]
                self.goal[1, 0] = goal[1, 0]
                break
        while True:
            q1 = np.random.uniform(self.arm1_bound[0], self.arm1_bound[1])
            q2 = np.random.uniform(self.arm2_bound[0], self.arm2_bound[1])
            self.q[0, 0] = q1
            self.q[1, 0] = q2
            self.Transformation()
            if self.EEF_arm2_image[0, 0] >= 0 and self.EEF_arm2_image[0, 0] <= 640:
                if self.EEF_arm2_image[1, 0] >= 0 and self.EEF_arm2_image[1, 0] <= 640:
                    break

        EEF_normalize = [self.EEF_arm2_image[0, 0] / 640, self.EEF_arm2_image[1, 0] / 512]
        goal_normalize = [self.goal[0, 0]/640, self.goal[1, 0]/512]
        q_normalize = [self.q[0, 0] / self.arm1_bound[1], self.q[1, 0] / self.arm2_bound[1]]

        # state
        s = np.concatenate((EEF_normalize, goal_normalize,q_normalize))
        return s

    def validation_reset(self, s):
        self.goal[0, 0] = s[2]*640
        self.goal[1, 0] = s[3]*512
        self.q[0, 0] = s[4]*self.arm1_bound[1]
        self.q[1, 0] = s[5]*self.arm2_bound[1]

        self.Transformation()

        EEF_normalize = [self.EEF_arm2_image[0, 0] / 640, self.EEF_arm2_image[1, 0] / 512]
        goal_normalize = [self.goal[0, 0] / 640, self.goal[1, 0] / 512]
        q_normalize = [self.q[0, 0] / self.arm1_bound[1], self.q[1, 0] / self.arm2_bound[1]]
        # state
        s = np.concatenate((EEF_normalize, goal_normalize, q_normalize))
        return s

class Viewer(pyglet.window.Window):
    window = pyglet.window.Window

    def __init__(self, EFF_arm1, EFF_arm2, arm1, arm2, goal, inf, mode):
        # self.origin_image_size_width = 640
        origin_image_size_width = 1000
        self.origin_image_size = 512
        self.offset = 108
        # self.offset = 208
        self.offset_image_size = self.origin_image_size + self.offset
        super(Viewer, self).__init__(width=origin_image_size_width, height=self.offset_image_size, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)

        # robot
        self.arm1_EFF_image = EFF_arm1
        self.arm2_EFF_image = EFF_arm2
        self.arm1_image = arm1
        self.arm2_image = arm2
        self.goal_image = goal

        self.iteration = inf[0]
        self.step = inf[1]
        self.distance_image = inf[2]
        self.mode = mode

        # add batch
        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        center = 616
        obstacle_width = 173
        obstacle_ = center + obstacle_width
        self.obstacle = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [430, self.offset_image_size,  # location
                     430, self.origin_image_size - 28,
                     obstacle_, self.origin_image_size - 28,
                     obstacle_, self.offset_image_size]),
            ('c3B', (86, 86, 86) * 4,))  # color

        # pyglet.gl.glLineWidth(50.)
        # arm1
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [200, 200,  # location
                     200, 250,
                     300, 250,
                     300, 200]),
            ('c3B', (211, 211, 211) * 4,))  # color

        # arm2
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 200,
                     200, 200,
                     200, 150]),
            ('c3B', (211, 211, 211) * 4,))  # color

        pyglet.gl.glEnable(pyglet.gl.GL_POINT_SMOOTH)
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
        pyglet.gl.glPointSize(20.)

        # EFF of arm1
        # self.joint = self.batch.add(
        #     1, pyglet.gl.GL_POINTS, None,
        #     ('v2f', (10, 10)),
        #     ('c3B', (100, 149, 237) * 1))

        # EFF of arm2
        self.EEF = self.batch.add(
            1, pyglet.gl.GL_POINTS, None,
            ('v2f', (30, 30)),
            ('c3B', (0, 255, 0) * 1))

        # goal
        self.goal = self.batch.add(
            1, pyglet.gl.GL_POINTS, None,
            ('v2f', (50, 50)),
            ('c3B', (238, 99, 99)))

        if self.mode == 'Train':
            self.Label_mode = pyglet.text.Label(text="Mode : Train",font_size=14, x=5, y=65,color=(0, 0, 255, 150))
            self.Label_iteration = pyglet.text.Label(text="iteration : 1",font_size=14, x=5, y=45, color=(0, 0, 255, 150))
            self.Label_step = pyglet.text.Label(text="step : 1", font_size=14, x=5, y=25, color=(0, 0, 255, 150))
            self.Label_distance = pyglet.text.Label(text="distance : 10", font_size=14, x=5, y=5, color=(0, 0, 255, 150))
        else:
            self.Label_mode = pyglet.text.Label(text="Mode : Eval", font_size=14, x=5, y=25, color=(0, 0, 255, 150))
            self.Label_distance = pyglet.text.Label(text="distance : 10", font_size=14, x=5, y=5, color=(0, 0, 255, 150))

    def render(self, EFF_arm1, EFF_arm2, arm1, arm2, goal, inf):
        # robot
        self.arm1_EFF_image = EFF_arm1
        self.arm2_EFF_image = EFF_arm2
        self.arm1_image = arm1
        self.arm2_image = arm2
        self.goal_image = goal
        self.iteration = inf[0]
        self.step = inf[1]
        self.distance_image = inf[2]
        self.update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
        time.sleep(0.1)

    def on_draw(self):

        self.clear()
        self.batch.draw()

        self.Label_mode.draw()
        self.Label_distance.draw()

        if self.mode == 'Train':
            self.Label_iteration.draw()
            self.Label_step.draw()

    def update(self):

        if self.mode == 'Train':
            self.Label_iteration.text = 'iteration : ' + np.str(self.iteration)
            self.Label_step.text = 'step : ' + np.str(self.step)
            self.Label_distance.text = 'distance : ' + np.str(self.distance_image)
        else:
            self.Label_distance.text = 'distance : ' + np.str(self.distance_image)


        # re-map coodinate
        # coodinate_arm1_EFF = self.re_coodinate(self.arm1_EFF_image)
        coodinate_arm2_EFF = self.re_coodinate(self.arm2_EFF_image)
        coodinate_arm1 = self.re_coodinate(self.arm1_image)
        coodinate_arm2 = self.re_coodinate(self.arm2_image)
        coodinate_goal = self.re_coodinate(self.goal_image)

        self.arm1.vertices = np.concatenate((coodinate_arm1[0], coodinate_arm1[1], coodinate_arm1[2], coodinate_arm1[3]))
        self.arm2.vertices = np.concatenate((coodinate_arm2[0], coodinate_arm2[1], coodinate_arm2[2], coodinate_arm2[3]))

        self.EEF.vertices = np.concatenate((coodinate_arm2_EFF ))
        self.goal.vertices = np.concatenate((coodinate_goal))

    def re_coodinate(self, original_coodinate_image):

        offset_coodinate_image = []
        for i in range(np.size(original_coodinate_image, 1)):
            u = original_coodinate_image[0, i]
            v = original_coodinate_image[1, i]
            v_ = self.origin_image_size - v
            offset_coodinate_image.append([u, v_])
        return offset_coodinate_image

    # convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):
        goal = {'x': 0, 'y': 0}
        goal['x'] = x
        goal['y'] = y
        self.goal_image[0, 0] = goal['x']
        self.goal_image[1, 0] = self.origin_image_size - goal['y']

if __name__ == '__main__':
    env = ArmEnv('Eval')
    env.reset()
    print(env.arm1_bound[0])
    while True:
        env.render()
        env.step(env.sample_action(), [0, 0])
        time.sleep(0.5)