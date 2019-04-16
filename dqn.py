import tensorflow as tf
import numpy as np
import models

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        feature_size,
        sensor_size,
        learning_rate = 1e-4,
        reward_decay = 0.92,
        e_greedy = 0.9,
        replace_target_iter = 300,
        memory_size = 500,
        batch_size = 64,
        e_greedy_increment = None,
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.feature_size = feature_size
        self.sensor_size = sensor_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.2 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0
        self.init_memory()
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Qnet_target')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Qnet_eval')
 
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # inputs
        self.s1 = tf.placeholder(tf.float32, [None, self.feature_size[0], self.feature_size[1], self.feature_size[2]], name="s1")
        self.s2 = tf.placeholder(tf.float32, [None, self.sensor_size], name="s2")
        self.s1_ = tf.placeholder(tf.float32, [None, self.feature_size[0], self.feature_size[1], self.feature_size[2]], name="s_")
        self.s2_ = tf.placeholder(tf.float32, [None, self.sensor_size], name="s2_")
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.end = tf.placeholder(tf.float32, [None, ], name='end')
        
        # ------------------ build Q Network ------------------
        # Evaluate Net
        self.q_eval = models.QNetwork(self.s1, self.s2, self.n_actions, 'Qnet_eval', reuse=False)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )

        # Target Net
        self.q_next = models.QNetwork(self.s1_, self.s2_, self.n_actions, 'Qnet_target', reuse=False)
        
        with tf.variable_scope("q_target"):
            q_target = self.r + self.end * self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_') # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        # Loss & Train
        with tf.variable_scope('loss'):
            self.td_error = tf.sqrt(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            self.loss = tf.reduce_mean(self.td_error)

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    
    def init_memory(self):
        self.memory = {
            "s1": np.zeros((self.memory_size, self.feature_size[0], self.feature_size[1], self.feature_size[2])),
            "s2": np.zeros((self.memory_size, self.sensor_size)),
            "a": np.zeros((self.memory_size)),
            "r": np.zeros((self.memory_size)),
            "s1_": np.zeros((self.memory_size, self.feature_size[0], self.feature_size[1], self.feature_size[2])),
            "s2_": np.zeros((self.memory_size, self.sensor_size)),
            "end": np.zeros((self.memory_size)),
            "error": np.zeros((self.memory_size)),
        }

    def store_transition(self, s1, s2, a, r, s1_, s2_, end):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        if self.memory_counter <= self.memory_size:
            index = self.memory_counter % self.memory_size
        else:
            index = self.memory_counter % self.memory_size
            #index = np.argmin(self.memory["error"])
        self.memory["s1"][index] = s1
        self.memory["s2"][index] = s2
        self.memory["a"][index] = a
        self.memory["r"][index] = r
        self.memory["s1_"][index] = s1_
        self.memory["s2_"][index] = s2_
        self.memory["end"][index] = end
        self.memory_counter += 1
    
    def choose_action(self, s1, s2):
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s1: np.expand_dims(s1, 0), self.s2: np.expand_dims(s2, 0)})
        if np.random.uniform() < self.epsilon:
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print("Q Value:", actions_value, "Choose Action:", action, "Epsilon:", self.epsilon, "\n")
        return action
    
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\n[Update network param]\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            prob = self.memory["error"].copy()
            prob = prob / np.sum(prob)
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, p=prob)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        _, cost, tderr = self.sess.run(
            [self._train_op, self.loss, self.td_error],
            feed_dict={
                self.s1: self.memory["s1"][sample_index], 
                self.s2: self.memory["s2"][sample_index],
                self.a: self.memory["a"][sample_index],
                self.r: self.memory["r"][sample_index],
                self.s1_: self.memory["s1_"][sample_index],
                self.s2_: self.memory["s2_"][sample_index],
                self.end: self.memory["end"][sample_index],
            })
        self.memory["error"][sample_index] = tderr

        if not hasattr(self, 'iter'):
            self.iter = 0
        else:
            self.iter += 1

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return cost