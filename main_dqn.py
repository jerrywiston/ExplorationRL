import Bot2DWrapper
import numpy as np
import dqn
import matplotlib.pyplot as plt
import json
#%%
seq_size = 3
memory_size = 1000
env = Bot2DWrapper.Bot2DEnv(obs_size=64, 
                            grid_size=3, 
                            map_path="Image/map7.png")

RL = dqn.DeepQNetwork(n_actions=3,
                  feature_size=[64, 64, seq_size],
                  sensor_size=60,
                  learning_rate=2e-4, 
                  reward_decay = 0.95,
                  e_greedy=0.98,
                  replace_target_iter=100, 
                  memory_size=memory_size,
                  e_greedy_increment=0.0001,)
#%%
if __name__ == '__main__':
    total_step = 0
    state_m_rec = np.zeros([64,64,seq_size], np.float32)
    reward_rec= []
    for eps in range(250):
        print('[ Episode ' + str(eps)  + ' ]')
        state_m, state_s = env.reset()
        step = 0
        
        # One Episode
        eps_reward = []
        loss = 0.
        while True:
            env.render()
            if step >= seq_size and total_step > memory_size:
                action = RL.choose_action(state_m, state_s)
            else:
                action = np.random.randint(0,3)

            # Get next state
            [state_m_next, state_s_next], reward_, collision = env.step(action)
            reward = reward_ - 10 # Baseline
            state_m_next = np.concatenate([state_m_rec[:,:,1:seq_size], state_m_next], axis=2)
            
            done = 1.
            if collision:
                reward = -80
                done = 0.

            if step > seq_size:
                RL.store_transition(state_m, state_s, action, reward, state_m_next, state_s_next, done)
                eps_reward.append(reward_)
                if total_step > memory_size:
                    loss = RL.learn()

            print('Episode:', eps, '| Step:',step, '| Reward:', reward, '| Loss:', loss)
            state_m = state_m_next.copy()
            step += 1
            total_step += 1
            if done == 0 and step >= seq_size:
                reward_rec.append(eps_reward)
                break
            if step >= 1000:
                break
    
    f = open("rec.json", "w")
    json.dump(reward_rec, f)

