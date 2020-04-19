from __future__ import division
import argparse

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from phoenix_dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

INPUT_SHAPE = (128,)

class AtariProcessor(Processor):
    def __init__(self):
        self.time_limit = 0
        self.alive = 1.0
        self.nb_enemy = 8
        self.bossHit = 0
        self.level_limit = 30
        self.level = -1
        self.enemyX = np.array([0,0,0,0,0,0,0,0])
        self.record = 0
        self.levelup = True
        
    def process_observation(self, observation):
        assert observation.shape == INPUT_SHAPE

        self.nb_enemy = observation[78]
        self.level = observation[74] + 6*observation[124]
        
        nb_fire = 0
        if observation[62]==128:
            nb_fire=1
        elif observation[62]==192:
            nb_fire=2
        elif observation[62]==224:
            nb_fire=3
        elif observation[62]==240:
            nb_fire=4

        self_fire = int(observation[89]!=193)
        ob = np.array([255*int(observation[15]>0)-128, observation[94]-128, 255*self_fire-128, self_fire*observation[93]-128, self_fire*observation[48]-128, 255*int(observation[49]>0)-128, 63*observation[70]-128, 17*(observation[13]%16)-128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -128, -128, -128, -128, 127, 127, 127, 127, 127, 127, 127, 127])
        for i in range(nb_fire):
            ob[48+i] = 127
            ob[52+i] = observation[88-i]-128
            ob[56+i] = observation[61-i]-128

        if self.levelup:
            self.enemyX[0] = observation[99]
            self.enemyX[1] = observation[100]
            self.enemyX[2] = observation[101]
            self.enemyX[3] = observation[102]
            self.enemyX[4] = observation[103]
            self.enemyX[5] = observation[104]
            self.enemyX[6] = observation[105]
            self.enemyX[7] = observation[106]
        
        if observation[74]==0:
            temp3 = int(observation[65]==1)
            temp4 = int(observation[66]==1)
            temp1 = int(observation[27]<68)
            temp2 = 1-temp1
            ob[8] = 255*temp1 - 128
            ob[16] = observation[99]*temp1 + 255*temp2 - 128
            ob[24] = observation[41]*temp1 + 255*temp2 - 128
            ob[32] = temp1*(observation[99] - self.enemyX[0])
            ob[40] = temp1 * 2*temp3*(2*temp4-1)
            temp1 = int(observation[34]<68)
            temp2 = 1-temp1
            ob[12] = 255*temp1 - 128
            ob[20] = observation[103]*temp1 + 255*temp2 - 128
            ob[28] = observation[41]*temp1 + 255*temp2 - 128
            ob[36] = temp1*(observation[103] - self.enemyX[4])
            ob[44] = temp1 * 2*temp3*(2*temp4-1)
            
            temp3 = int(observation[65]==2)
            temp4 = int(observation[66]==2)
            temp1 = int(observation[28]<68)
            temp2 = 1-temp1
            ob[9] = 255*temp1 - 128
            ob[17] = observation[100]*temp1 + 255*temp2 - 128
            ob[25] = observation[42]*temp1 + 255*temp2 - 128
            ob[33] = temp1*(observation[100] - self.enemyX[1])
            ob[41] = temp1 * 2*temp3*(2*temp4-1)
            temp1 = int(observation[35]<68)
            temp2 = 1-temp1
            ob[13] = 255*temp1 - 128
            ob[21] = observation[104]*temp1 + 255*temp2 - 128
            ob[29] = observation[42]*temp1 + 255*temp2 - 128
            ob[37] = temp1*(observation[104] - self.enemyX[5])
            ob[45] = temp1 * 2*temp3*(2*temp4-1)
            
            temp3 = int(observation[65]==4)
            temp4 = int(observation[66]==4)
            temp1 = int(observation[29]<68)
            temp2 = 1-temp1
            ob[10] = 255*temp1 - 128
            ob[18] = observation[101]*temp1 + 255*temp2 - 128
            ob[26] = observation[43]*temp1 + 255*temp2 - 128
            ob[34] = temp1*(observation[101] - self.enemyX[2])
            ob[42] = temp1 * 2*temp3*(2*temp4-1)
            temp1 = int(observation[36]<68)
            temp2 = 1-temp1
            ob[14] = 255*temp1 - 128
            ob[22] = observation[105]*temp1 + 255*temp2 - 128
            ob[30] = observation[43]*temp1 + 255*temp2 - 128
            ob[38] = temp1*(observation[105] - self.enemyX[6])
            ob[46] = temp1 * 2*temp3*(2*temp4-1)

            temp3 = int(observation[65]==8)
            temp4 = int(observation[66]==8)
            temp1 = int(observation[30]<68)
            temp2 = 1-temp1
            ob[11] = 255*temp1 - 128
            ob[19] = observation[102]*temp1 + 255*temp2 - 128
            ob[27] = observation[44]*temp1 + 255*temp2 - 128
            ob[35] = temp1*(observation[102] - self.enemyX[3])
            ob[43] = temp1 * 2*temp3*(2*temp4-1)
            temp1 = int(observation[37]<68)
            temp2 = 1-temp1
            ob[15] = 255*temp1 - 128
            ob[23] = observation[106]*temp1 + 255*temp2 - 128
            ob[31] = observation[44]*temp1 + 255*temp2 - 128
            ob[39] = temp1*(observation[106] - self.enemyX[7])
            ob[47] = temp1 * 2*temp3*(2*temp4-1)
        elif observation[74]==1:
            temp3 = int(observation[65]==1)
            temp4 = int(observation[66]==1)
            temp1 = int(observation[27]<100)
            temp2 = 1-temp1
            ob[8] = 255*temp1 - 128
            ob[16] = observation[99]*temp1 + 255*temp2 - 128
            ob[24] = observation[41]*temp1 + 255*temp2 - 128
            ob[32] = temp1*(observation[99] - self.enemyX[0])
            ob[40] = temp1 * 2*temp3*(2*temp4-1)
            temp1 = int(observation[34]<100)
            temp2 = 1-temp1
            ob[12] = 255*temp1 - 128
            ob[20] = observation[103]*temp1 + 255*temp2 - 128
            ob[28] = observation[41]*temp1 + 255*temp2 - 128
            ob[36] = temp1*(observation[103] - self.enemyX[4])
            ob[44] = temp1 * 2*temp3*(2*temp4-1)

            temp3 = int(observation[65]==2)
            temp4 = int(observation[66]==2)
            temp1 = int(observation[28]<100)
            temp2 = 1-temp1
            ob[9] = 255*temp1 - 128
            ob[17] = observation[100]*temp1 + 255*temp2 - 128
            ob[25] = observation[42]*temp1 + 255*temp2 - 128
            ob[33] = temp1*(observation[100] - self.enemyX[1])
            ob[41] = temp1 * 2*temp3*(2*temp4-1)
            temp1 = int(observation[35]<100)
            temp2 = 1-temp1
            ob[13] = 255*temp1 - 128
            ob[21] = observation[104]*temp1 + 255*temp2 - 128
            ob[29] = observation[42]*temp1 + 255*temp2 - 128
            ob[37] = temp1*(observation[104] - self.enemyX[5])
            ob[45] = temp1 * 2*temp3*(2*temp4-1)
            
            temp3 = int(observation[65]==4)
            temp4 = int(observation[66]==4)
            temp1 = int(observation[29]<100)
            temp2 = 1-temp1
            ob[10] = 255*temp1 - 128
            ob[18] = observation[101]*temp1 + 255*temp2 - 128
            ob[26] = observation[43]*temp1 + 255*temp2 - 128
            ob[34] = temp1*(observation[101] - self.enemyX[2])
            ob[42] = temp1 * 2*temp3*(2*temp4-1)
            temp1 = int(observation[36]<100)
            temp2 = 1-temp1
            ob[14] = 255*temp1 - 128
            ob[22] = observation[105]*temp1 + 255*temp2 - 128
            ob[30] = observation[43]*temp1 + 255*temp2 - 128
            ob[38] = temp1*(observation[105] - self.enemyX[6])
            ob[46] = temp1 * 2*temp3*(2*temp4-1)

            temp3 = int(observation[65]==8)
            temp4 = int(observation[66]==8)
            temp1 = int(observation[30]<100)
            temp2 = 1-temp1
            ob[11] = 255*temp1 - 128
            ob[19] = observation[102]*temp1 + 255*temp2 - 128
            ob[27] = observation[44]*temp1 + 255*temp2 - 128
            ob[35] = temp1*(observation[102] - self.enemyX[3])
            ob[43] = temp1 * 2*temp3*(2*temp4-1)
            temp1 = int(observation[37]<100)
            temp2 = 1-temp1
            ob[15] = 255*temp1 - 128
            ob[23] = observation[106]*temp1 + 255*temp2 - 128
            ob[31] = observation[44]*temp1 + 255*temp2 - 128
            ob[39] = temp1*(observation[106] - self.enemyX[7])
            ob[47] = temp1 * 2*temp3*(2*temp4-1)
        elif observation[74]==2:
            vy = 2 - 4*int(observation[64]==64)
            temp1 = int(observation[27]<8)
            temp2 = 1-temp1
            ob[8] = 255*temp1 - 128
            ob[16] = observation[99]*temp1 + 255*temp2 - 128
            ob[24] = observation[41]*temp1 + 255*temp2 - 128
            ob[32] = temp1*(observation[99] - self.enemyX[0])
            ob[40] = temp1*vy
            temp1 = int(observation[28]<8)
            temp2 = 1-temp1
            ob[9] = 255*temp1 - 128
            ob[17] = observation[100]*temp1 + 255*temp2 - 128
            ob[25] = observation[42]*temp1 + 255*temp2 - 128
            ob[33] = temp1*(observation[100] - self.enemyX[1])
            ob[41] = temp1*vy
            temp1 = int(observation[29]<8)
            temp2 = 1-temp1
            ob[10] = 255*temp1 - 128
            ob[18] = observation[101]*temp1 + 255*temp2 - 128
            ob[26] = observation[43]*temp1 + 255*temp2 - 128
            ob[34] = temp1*(observation[101] - self.enemyX[2])
            ob[42] = temp1*vy
            temp1 = int(observation[30]<8)
            temp2 = 1-temp1
            ob[11] = 255*temp1 - 128
            ob[19] = observation[102]*temp1 + 255*temp2 - 128
            ob[27] = observation[44]*temp1 + 255*temp2 - 128
            ob[35] = temp1*(observation[102] - self.enemyX[3])
            ob[43] = temp1*vy
            temp1 = int(observation[31]<8)
            temp2 = 1-temp1
            ob[12] = 255*temp1 - 128
            ob[20] = observation[103]*temp1 + 255*temp2 - 128
            ob[28] = observation[45]*temp1 + 255*temp2 - 128
            ob[36] = temp1*(observation[103] - self.enemyX[4])
            ob[44] = temp1*vy
            temp1 = int(observation[32]<8)
            temp2 = 1-temp1
            ob[13] = 255*temp1 - 128
            ob[21] = observation[104]*temp1 + 255*temp2 - 128
            ob[29] = observation[46]*temp1 + 255*temp2 - 128
            ob[37] = temp1*(observation[104] - self.enemyX[5])
            ob[45] = temp1*vy
            temp1 = int(observation[33]<8)
            temp2 = 1-temp1
            ob[14] = 255*temp1 - 128
            ob[22] = observation[105]*temp1 + 255*temp2 - 128
            ob[30] = observation[47]*temp1 + 255*temp2 - 128
            ob[38] = temp1*(observation[105] - self.enemyX[6])
            ob[46] = temp1*vy

            ob = np.delete(ob, [15, 23, 31, 39, 47])
        elif observation[74]==3:
            vy = 2 - 4*int(observation[64]==64)
            temp1 = int(observation[27]<40)
            temp2 = 1-temp1
            ob[8] = 255*temp1 - 128
            ob[16] = observation[99]*temp1 + 255*temp2 - 128
            ob[24] = observation[41]*temp1 + 255*temp2 - 128
            ob[32] = temp1*(observation[99] - self.enemyX[0])
            ob[40] = temp1*vy
            temp1 = int(observation[28]<40)
            temp2 = 1-temp1
            ob[9] = 255*temp1 - 128
            ob[17] = observation[100]*temp1 + 255*temp2 - 128
            ob[25] = observation[42]*temp1 + 255*temp2 - 128
            ob[33] = temp1*(observation[100] - self.enemyX[1])
            ob[41] = temp1*vy
            temp1 = int(observation[29]<40)
            temp2 = 1-temp1
            ob[10] = 255*temp1 - 128
            ob[18] = observation[101]*temp1 + 255*temp2 - 128
            ob[26] = observation[43]*temp1 + 255*temp2 - 128
            ob[34] = temp1*(observation[101] - self.enemyX[2])
            ob[42] = temp1*vy
            temp1 = int(observation[30]<40)
            temp2 = 1-temp1
            ob[11] = 255*temp1 - 128
            ob[19] = observation[102]*temp1 + 255*temp2 - 128
            ob[27] = observation[44]*temp1 + 255*temp2 - 128
            ob[35] = temp1*(observation[102] - self.enemyX[3])
            ob[43] = temp1*vy
            temp1 = int(observation[31]<40)
            temp2 = 1-temp1
            ob[12] = 255*temp1 - 128
            ob[20] = observation[103]*temp1 + 255*temp2 - 128
            ob[28] = observation[45]*temp1 + 255*temp2 - 128
            ob[36] = temp1*(observation[103] - self.enemyX[4])
            ob[44] = temp1*vy
            temp1 = int(observation[32]<40)
            temp2 = 1-temp1
            ob[13] = 255*temp1 - 128
            ob[21] = observation[104]*temp1 + 255*temp2 - 128
            ob[29] = observation[46]*temp1 + 255*temp2 - 128
            ob[37] = temp1*(observation[104] - self.enemyX[5])
            ob[45] = temp1*vy
            temp1 = int(observation[33]<40)
            temp2 = 1-temp1
            ob[14] = 255*temp1 - 128
            ob[22] = observation[105]*temp1 + 255*temp2 - 128
            ob[30] = observation[47]*temp1 + 255*temp2 - 128
            ob[38] = temp1*(observation[105] - self.enemyX[6])
            ob[46] = temp1*vy

            ob = np.delete(ob, [15, 23, 31, 39, 47])
        else:
            ob[8] = observation[1] - 128
            ob[9] = observation[3] - 128
            ob[10] = observation[4] - 128
            ob[11] = observation[5] - 128
            ob[12] = observation[6] - 128
            ob[13] = observation[7] - 128
            ob[14] = observation[12] - 128
            for i in range(20):
                ob[15+i] = observation[27+i] - 128
            ob[35] = observation[51] - 128
            ob[36] = observation[90] - 128
            ob[37] = observation[91] - 128
            ob[38] = observation[95] - 128
            ob[39] = observation[98] - 128
            ob[40] = observation[110] - 128
            ob[41] = observation[122] - 128
            ob[42] = observation[126] - 128
            ob[43] = observation[127] - 128
            ob = np.delete(ob, [44, 45, 46, 47])

        self.enemyX = ob[16:24]
        
        return ob.astype('int8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        
        return processed_batch

    def process_step(self, observation, reward, done, info):
        die = float(observation[15]>0)
        processed_reward = float(observation[78]<self.nb_enemy) + float(observation[74]==4) * float(observation[97]>self.bossHit) * float(observation[48]<189) * (0.2 + 0.02*float(observation[48] - 87 + 6*int((136 - observation[95])/8))) - 2.0 * die * self.alive
        currentlevel = observation[74] + 6*observation[124]
        self.levelup = (self.level != currentlevel)
        
        self.time_limit = int(processed_reward==0) * (self.time_limit+1)
        self.alive = 1.0 - die
        self.bossHit = observation[97]

        done = done or (currentlevel>self.level_limit)
        
        processed_observation = self.process_observation(observation)

        return processed_observation, processed_reward, done, info

processor = AtariProcessor()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='Phoenix-ramDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name).unwrapped
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the similar model that was described by Mnih et al. (2015).
model_number = 15
WINDOW_LENGTH = 1
input_shape = [(WINDOW_LENGTH,) + (60,), (WINDOW_LENGTH,) + (55,), (WINDOW_LENGTH,) + (56,), (WINDOW_LENGTH,) + (60,), (WINDOW_LENGTH,) + (55,), (WINDOW_LENGTH,) + (56,), (WINDOW_LENGTH,) + (60,), (WINDOW_LENGTH,) + (55,), (WINDOW_LENGTH,) + (56,), (WINDOW_LENGTH,) + (60,), (WINDOW_LENGTH,) + (55,), (WINDOW_LENGTH,) + (56,), (WINDOW_LENGTH,) + (60,), (WINDOW_LENGTH,) + (55,), (WINDOW_LENGTH,) + (56,)]

models = []
for i in range(model_number):
    models.append(Sequential())

if K.image_dim_ordering() == 'tf':
    # (n, channels)
    for i in range(model_number):
        models[i].add(Permute((2, 1), input_shape=input_shape[i]))
elif K.image_dim_ordering() == 'th':
    # (channels, n)
    for i in range(model_number):
        models[i].add(Permute((1, 2), input_shape=input_shape[i]))
else:
    raise RuntimeError('Unknown image_dim_ordering.')

for i in range(model_number):
    models[i].add(Flatten())
    models[i].add(Dense(128))
    models[i].add(Activation('relu'))
    models[i].add(Dense(128))
    models[i].add(Activation('relu'))
    models[i].add(Dense(128))
    models[i].add(Activation('relu'))
    models[i].add(Dense(128))
    models[i].add(Activation('relu'))
    models[i].add(Dense(nb_actions))
    models[i].add(Activation('linear'))
    print(models[i].summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memories = []
for i in range(model_number):
    memories.append(SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH))

# Select a policy.
policy = BoltzmannQPolicy()

dqn = DQNAgent(models=models, nb_actions=nb_actions, policy=policy, memories=memories,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=1, delta_clip=1. ,enable_dueling_network=True, dueling_type='avg')
dqn.compile(Adam(lr=.00025), metrics=['mae'])


if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]

    dqn.fit(env, callbacks=callbacks, nb_steps=5000000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
    
    
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
