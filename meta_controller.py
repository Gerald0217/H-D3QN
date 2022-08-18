"""
@author: Chuanxin Yu Modified
H-D3QN (Hierarchical reinforcement learning) Multi-objective point trajectory optimization of cellular-connected UAV
2022年8月17日11:54:56
"""
import numpy as np
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from keras.layers import Input, Dense,Lambda, Flatten
from keras.models import Model
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
import keras.backend as K
# import radio_environment as rad_env #the actual radio environment

X_MAX=2000.0 
Y_MAX=2000.0 #The area region in meters
MAX_VALS=np.array([[X_MAX,Y_MAX]])

DESTINATION=np.array([[1400,1600]],dtype="float32")#UAV flying destination in meter
DIST_TOLERANCE=30#considered as reach destination if UAV reaches the vicinity within DIST_TOLERANCE meters
TARGET_A=np.array([[800,1200]],dtype="float32")
GOAL = [TARGET_A, DESTINATION]

DISCOUNT = 1
REPLAY_MEMORY_SIZE = 100_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 5_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MAX_STEP=200 #maximum number of time steps per episode
MODEL_NAME = '512_256_128_128'
MIN_REWARD = -1000  # For model save
nSTEP=30 #parameter for multi-step learning

# Environment settings
EPISODES = 5000#Number of training episodes


# Exploration settings
epsilon =0.5  # not a constant, going to be decayed
EPSILON_DECAY = 0.998
MIN_EPSILON = 0

episode_all=np.arange(EPISODES)
epsilon_all=epsilon*EPSILON_DECAY**episode_all
epsilon_all=np.maximum(epsilon_all,MIN_EPSILON)

plt.figure()
plt.plot(episode_all,epsilon_all,'b',linewidth=2)
plt.grid(True,which='major',axis='both')
plt.show()

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


delta_t=0.5 #time step length in seconds

#penalty measured in terms of the time required to reach destination
REACH_DES_REWARD=200
MOVE_PENALTY = 1
NON_COVER_PENALTY = 40
OUT_BOUND_PENALTY = 10000


x=np.linspace(0,X_MAX,200)
y=np.linspace(0,Y_MAX,200)


OBSERVATION_SPACE_VALUES=(2,)#2-dimensional UAV flight, x-y coordinate of UAV
ACTIONS=np.array([[0, 1],
             [1,0],
             [0,-1],
             [-1,0]],dtype=int)#the possible actions (UAV flying directions)   
ACTION_SPACE_SIZE = ACTIONS.shape[0]
   
MAX_SPEED=20 #maximum UAV speed in m/s
STEP_DISPLACEMENT=MAX_SPEED*delta_t #The displacement per time step

# loc_x=np.random.uniform(50,1950,(1,1))
# loc_y=np.random.uniform(50,1950,(1,1))
# loc=np.concatenate((loc_x,loc_y),axis=1)
# M = np.concatenate((DESTINATION, loc),axis=0)
M=np.random.uniform(50,Y_MAX-50,(2,2))
  
# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        pass


class UAVEnv:
    
    def reset(self):
        self.episode_step = 0
        loc_x=np.random.uniform(50,X_MAX-50,(1,1))
        loc_y=np.random.uniform(50,Y_MAX-50,(1,1))
        s0=np.concatenate((loc_x,loc_y),axis=1)
        return s0


    def random_generate_states(self,num_states):
        loc = np.random.uniform(50,Y_MAX-50,(num_states,2,2))
        return loc
            
    def random_generate_states_lower(self,num_states):
        loc=np.random.uniform(50,X_MAX-50,(num_states,2))
        
        return loc

   #for each location visited by the UAV, it will have the J signal measurements from
   #each of the M cellular BSs
   #based on these M*J measurements, calculate the empirical outage probability
    def get_empirical_outage(self, location):
        #location given in meters
        #convert the location to kilometer
        loc_km=np.zeros(shape=(1,3))
        loc_km[0,:2]=location/1000
        loc_km[0,2]=0.1#UAV height in km
        Pout=rad_env.getPointMiniOutage(loc_km)
        return Pout[0]
    
    
    def step(self, current_state, action_idx, goal_index, step, cur_traj):
        step = step + 1

        print("step is :{}".format(step))              
        next_state=current_state+STEP_DISPLACEMENT*ACTIONS[action_idx]
        outbound=False
        out_bound_check1=next_state<0
        out_bound_check2=next_state[0,0]>X_MAX
        out_bound_check3=next_state[0,1]>Y_MAX
        if out_bound_check1.any() or out_bound_check2.any() or out_bound_check3.any():
            outbound=True
            print('Happen outbound-x-x-x--x-x-x-x-x-x-x-Happen outbound--x-x-x-x--xx--x-Happen outbound-x-x-x-x-x--x-x-x-x--x!!!!!!!!')
            next_state[next_state<0]=0
            next_state[0,0]=np.minimum(X_MAX,next_state[0,0])
            next_state[0,1]=np.minimum(Y_MAX,next_state[0,1])
           
        if LA.norm(next_state-GOAL[goal_index])<=DIST_TOLERANCE:
            terminal=True
            next_state = GOAL[goal_index]
            
            print('ARRIVE TARGET->->->->->->->->->->->->->ARRIVE TARGET>->->->->->->->->-ARRIVE TARGET>->->->->->->>->->->->!!!!!!!!')
        else:
            terminal=False
    
        if terminal or outbound:
            reward=-MOVE_PENALTY
        else: 
            Pout=self.get_empirical_outage(next_state)
            reward=-MOVE_PENALTY-NON_COVER_PENALTY*Pout
                          
        done = False
                               
        if terminal or step >= MAX_STEP or outbound:
            done = True
                           
        return next_state, reward, terminal, outbound, step, done

env = UAVEnv()


# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


# Agent class
class DQNAgent:
    def __init__(self):
        # Main model
        self.model = self.create_model(dueling=True)
        
        self.initilize_model()

        # Target network
        self.target_model = self.create_model(dueling=True)
        self.target_model.set_weights(self.model.get_weights())
       
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self, dueling):

        # inp = Input(shape=OBSERVATION_SPACE_VALUES)
        inp = Input(shape=M.shape)
        outp=Dense(128,activation='relu')(inp)
        outp=Dense(256,activation='relu')(outp)
        outp=Dense(128,activation='relu')(outp)
        outp=Dense(128,activation='relu')(outp)
        outp = Flatten()(outp)
        
        if(dueling):
            # Have the network estimate the Advantage function as an intermediate layer
            outp=Dense(ACTION_SPACE_SIZE+1, activation='linear')(outp)
            outp=Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(ACTION_SPACE_SIZE,))(outp)
        else:
            outp=Dense(ACTION_SPACE_SIZE,activation='linear')(outp)



        model=Model(inp,outp)
        model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_absolute_error', 'mean_squared_error'])
        model.summary()
        return model


       
    def normalize_data(self,input_data):
        return input_data/MAX_VALS

       
    def initilize_model(self):       
        #initialize the DQN so that the Q values of each (state,action) pair
        #equal to: -MOVE_PENALTY*distance/STEP_DISPLACEMENT,
        #where distance is the distance between the next state and the destination
        #this will encourage shortest path flying initially when there is no information on the coverage map
               
        xx,yy=np.meshgrid(x,y,indexing='ij')
        
        plt.figure(0)
        plt.plot(DESTINATION[0,0],DESTINATION[0,1],'r>',markersize=15)
        plt.show()
                
        num_states=50000

        
        #下面这个是数据一
        xy_loc_target = env.random_generate_states_lower(num_states) #这个数据的标签
        y_mat_target = np.zeros((1,2,2))
        for i in xy_loc_target:
            i = i.reshape(1,2)
            s = np.concatenate((i, TARGET_A), axis=0)
            s = s.reshape(1,2,2)
            c = np.concatenate((y_mat_target, s),axis=0)
            y_mat_target= c

        c = c[1:num_states+1,:] #这个数据是训练(其中的一部分)

        Actions_aug=np.zeros((1,2,ACTION_SPACE_SIZE),dtype=int)
        for i in range(Actions_aug.shape[2]):
            Actions_aug[:,:,i]=ACTIONS[i]
            
        Actions_aug=np.tile(Actions_aug,(num_states,1,1))
        xy_loc_aug=np.zeros((num_states,2,1))
        xy_loc_aug[:,:,0]=xy_loc_target
        xy_loc_aug=np.repeat(xy_loc_aug,ACTION_SPACE_SIZE,axis=2)
        xy_loc_next_state=xy_loc_aug+STEP_DISPLACEMENT*Actions_aug
        
        xy_loc_next_state[xy_loc_next_state<0]=0
        xy_loc_next_state[:,0,:]=np.minimum(X_MAX,xy_loc_next_state[:,0,:])
        xy_loc_next_state[:,1,:]=np.minimum(Y_MAX,xy_loc_next_state[:,1,:])
        
        end_loc_reshaped=np.zeros((1,2,1))
        end_loc_reshaped[0,:,0]=TARGET_A
        distance_to_destination=LA.norm(xy_loc_next_state-end_loc_reshaped,axis=1)#compute the distance to destination            
        Q_init_target=-distance_to_destination/STEP_DISPLACEMENT*MOVE_PENALTY


        #下面这个是数据二
        xy_loc_destination = env.random_generate_states_lower(num_states) #这个数据的标签
        y_mat_destination = np.zeros((1,2,2))
        for i in xy_loc_destination:
            i = i.reshape(1,2)
            s = np.concatenate((i, DESTINATION), axis=0)
            s = s.reshape(1,2,2)
            g = np.concatenate((y_mat_destination, s),axis=0)
            y_mat_destination= g

        g = g[1:num_states+1,:] #这个数据是训练(其中的一部分)

        Actions_aug_destination=np.zeros((1,2,ACTION_SPACE_SIZE),dtype=int)
        for i in range(Actions_aug_destination.shape[2]):
            Actions_aug_destination[:,:,i]=ACTIONS[i]
            
        Actions_aug_destination=np.tile(Actions_aug_destination,(num_states,1,1))
        xy_loc_aug_destination=np.zeros((num_states,2,1))
        xy_loc_aug_destination[:,:,0]=xy_loc_destination
        xy_loc_aug_destination=np.repeat(xy_loc_aug_destination,ACTION_SPACE_SIZE,axis=2)
        xy_loc_next_state_destination=xy_loc_aug_destination+STEP_DISPLACEMENT*Actions_aug_destination
        
        xy_loc_next_state_destination[xy_loc_next_state_destination<0]=0
        xy_loc_next_state_destination[:,0,:]=np.minimum(X_MAX,xy_loc_next_state_destination[:,0,:])
        xy_loc_next_state_destination[:,1,:]=np.minimum(Y_MAX,xy_loc_next_state_destination[:,1,:])
        
        end_loc_reshaped_destination=np.zeros((1,2,1))
        end_loc_reshaped_destination[0,:,0]=DESTINATION
        distance_to_destination_destination=LA.norm(xy_loc_next_state_destination-end_loc_reshaped_destination,axis=1)#compute the distance to destination            
        Q_init_destination=-distance_to_destination_destination/STEP_DISPLACEMENT*MOVE_PENALTY




        
        
        # Actions_aug=np.zeros((1,2,8),dtype=int)
        # for i in range(8):
        #     if i<4:
        #         Actions_aug[:,:,i] = ACTIONS[i]
        #     else:
        #         Actions_aug[:,:,i] = ACTIONS[i-4]

        # Actions_aug[:,1,:] = 0

        # Actions_aug[:,0,4] = 1
        # Actions_aug[:,0,5] = 0
        # Actions_aug[:,0,6] = -1
        # Actions_aug[:,0,7] = 0
                    
        # Actions_aug=np.tile(Actions_aug,(xy_loc.shape[0],1,1))
        # xy_loc_aug=np.zeros((xy_loc.shape[0],2,2))
        # xy_loc_aug=xy_loc
        # xy_loc_aug=np.repeat(xy_loc_aug,4,axis=2)
        # xy_loc_next_state=xy_loc_aug+STEP_DISPLACEMENT*Actions_aug

        # xy_loc_next_state[xy_loc_next_state<0]=0
        # xy_loc_next_state[:,0,:]=np.minimum(X_MAX,xy_loc_next_state[:,0,:])
        # xy_loc_next_state[:,1,:]=np.minimum(Y_MAX,xy_loc_next_state[:,1,:])
        
        # end_loc_reshaped=np.zeros((1,2,1))
        # end_loc_reshaped[0,:,0]=DESTINATION
        # distance_to_destination=LA.norm(xy_loc_next_state-end_loc_reshaped,axis=1)#compute the distance to destination            
        # Q_init=-distance_to_destination/STEP_DISPLACEMENT*MOVE_PENALTY

        Q_init = np.concatenate((Q_init_target, Q_init_destination),axis=0)
        xy_loc = np.concatenate((c, g),axis=0)

        # Q_init = Q_init_destination
        # xy_loc = g




        
        train_data=xy_loc[:int(num_states*0.8),:]
        train_label=Q_init[:int(num_states*0.8),:]
             
        test_data=xy_loc[int(num_states*0.8):,:]
        test_label=Q_init[int(num_states*0.8):,:]
        
       
        history=self.model.fit(self.normalize_data(train_data),train_label,epochs=20,validation_split=0.2,verbose=1)
                    
        history_dict = history.history
        history_dict.keys()
                                                                
        mse = history_dict['mean_squared_error']
        val_mse = history_dict['val_mean_squared_error']
        mae = history_dict['mean_absolute_error']
        val_mae=history_dict['val_mean_absolute_error']
        
     
        epochs = range(1, len(mse) + 1)
        
        plt.figure()   
        
        plt.plot(epochs, mse, 'bo', label='Training MSE')
        plt.plot(epochs, val_mse, 'r', label='Validation MSE')
        plt.title('Training and validation MSE')
#        plt.ylim(0,100)
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        
        plt.show()
        
        
        plt.figure()   # clear figure
        
        plt.plot(epochs, mae, 'bo', label='Training MAE')
        plt.plot(epochs, val_mae, 'r', label='Validation MAE')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
    #    plt.ylim(0,15)
        plt.legend()
        
        plt.show()
             
        result=self.model.evaluate(self.normalize_data(test_data),test_label)
        print(result)
                        
             

    #Add data to replay memory for n-step return
    #(St, At, R_nstep, S_{t+n}, terminal, outbound, done)
    #where R_nstep=R_{t+1}+gamma*R_{t+2}+gamma^2*R_{t+3}....+gamma^(nSTEP-1)*R_{t+n}
    def update_replay_memory_nStepLearning(self,slide_window,nSTEP,endEpisode):
        #update only after n steps
        if len(slide_window)<nSTEP:
            return

#        slide_window contains the list in the following order:
#        (current_state,action_idx,reward,next_state,terminal,outbound,done)        
        rewards_nsteps= [transition[2] for transition in slide_window]
        discount_nsteps=DISCOUNT**np.arange(nSTEP)
        R_nstep=sum(rewards_nsteps*discount_nsteps)
                                             
        St=slide_window[0][0]
        At=slide_window[0][1]
        
        St_plus_n=slide_window[-1][3]
        terminal=slide_window[-1][4]
        outbound=slide_window[-1][5]
        goal_index = slide_window[-1][6]
        done=slide_window[-1][7]
        
        """ Store experience in memory buffer
        """         
        self.replay_memory.append((St,At,R_nstep,St_plus_n,terminal,outbound,goal_index,done))
        
             
        if endEpisode:#Truncated n-step return for the last few steps at the end of the episode 
            for i in range(1,nSTEP):
                rewards_i=rewards_nsteps[i:]
                discount_i=DISCOUNT**np.arange(nSTEP-i)
                R_i=sum(rewards_i*discount_i)
                
                St_i=slide_window[i][0]
                At_i=slide_window[i][1]
                                
                self.replay_memory.append((St_i,At_i,R_i,St_plus_n,terminal,outbound,goal_index,done))
            
        
    def sample_batch_from_replay_memory(self):
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_state_batch = np.zeros((MINIBATCH_SIZE, OBSERVATION_SPACE_VALUES[0]))
        next_state_batch = np.zeros((MINIBATCH_SIZE, OBSERVATION_SPACE_VALUES[0]))
        
        actions_idx, rewards, terminal, outbound, goal_index,done= [], [], [],[],[],[]

        y_mat_current = np.zeros((1,2,2))   
        y_mat_next = np.zeros((1,2,2))      
        for idx, val in enumerate(minibatch):

            current_state_batch[idx] = val[0]
            current_observation = np.concatenate((val[0], GOAL[val[6]]), axis=0)
            current_observation = current_observation.reshape(1,2,2)
            current_observation_batch = np.concatenate((y_mat_current, current_observation), axis=0)
            y_mat_current = current_observation_batch

            actions_idx.append(val[1])
            rewards.append(val[2])

            next_state_batch[idx] = val[3]
            next_observation = np.concatenate((val[3], GOAL[val[6]]), axis=0)
            next_observation = next_observation.reshape(1,2,2)
            next_observation_batch = np.concatenate((y_mat_next, next_observation), axis=0)
            y_mat_next = next_observation_batch

            terminal.append(val[4])
            outbound.append(val[5])
            goal_index.append(val[6])
            done.append(val[7])
        current_observation_batch = current_observation_batch[1:MINIBATCH_SIZE+1,:]
        next_observation_batch = next_observation_batch[1:MINIBATCH_SIZE+1,:]

            
        return current_state_batch, actions_idx, rewards, next_state_batch, terminal, outbound, goal_index, current_observation_batch,next_observation_batch, done

       
    
    def deepDoubleQlearn(self,episode_done):
        # Start training only if certain number of samples is already saved                 
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
                    
        current_state_batch, actions_idx, rewards, next_state_batch, terminal,outbound,goal_index, current_observation_batch, next_observation_batch, done = self.sample_batch_from_replay_memory()
                           
        current_Q_values=self.model.predict(self.normalize_data(current_observation_batch))
       
        next_Q_values_currentNetwork=self.model.predict(self.normalize_data(next_observation_batch))  # use the current network to evaluate action
        next_actions=np.argmax(next_Q_values_currentNetwork,axis=1)        
          
        next_Q_values = self.target_model.predict(self.normalize_data(next_observation_batch))  # use the target network to evaluate value
        

        Y=current_Q_values
        
        for i in range(MINIBATCH_SIZE):

            if terminal[i]:
                target = rewards[i]+REACH_DES_REWARD
            elif outbound[i]:
                target=rewards[i]-OUT_BOUND_PENALTY
            else:
#                target = rewards[i] + DISCOUNT**nSTEP*np.minimum(next_Q_values[i,next_actions[i]],-1)
                target = rewards[i] + DISCOUNT**nSTEP*next_Q_values[i,next_actions[i]]
                
            Y[i,actions_idx[i]]=target
                                  
        self.model.fit(self.normalize_data(current_observation_batch), Y, batch_size=MINIBATCH_SIZE,verbose=0, shuffle=False, callbacks=[self.tensorboard] if episode_done else None)
        
        # Update target network counter every episode
        if episode_done:
            self.target_update_counter += 1            
            # If counter reaches set value, update target network with weights of main network
            if self.target_update_counter >= UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
                             
   
    def choose_action(self,current_state,cur_traj,goal_index,epsilon):                       
        next_possible_states=current_state+STEP_DISPLACEMENT*ACTIONS       
        
        next_possible_states[next_possible_states<0]=0
        next_possible_states[:,0]=np.minimum(next_possible_states[:,0],X_MAX)
        next_possible_states[:,1]=np.minimum(next_possible_states[:,1],Y_MAX)        
        
        next_possible_states=next_possible_states.tolist()
        
        no_repetition=[]
        
        cur_traj=cur_traj[-10:] #no return to the previous few locations
        
        for state in next_possible_states:
            no_repetition.append(state not in cur_traj)
             
        
        actions_idx_all=np.arange(ACTION_SPACE_SIZE)
        actions_idx_valid=actions_idx_all[no_repetition]
           
        if np.random.rand()<=epsilon or len(actions_idx_valid)==0:#Exploration
            action_idx=np.random.randint(0,ACTION_SPACE_SIZE) 
            return action_idx
        else:
            current_observation = np.concatenate((current_state,GOAL[goal_index]), axis=0)       
            current_observation = current_observation.reshape(1,2,2)
            Q_value=self.model.predict(self.normalize_data(current_observation))
            Q_value=Q_value[0]            
            action_idx_maxVal=np.argmax(Q_value)
            if action_idx_maxVal in actions_idx_valid:
                action_idx=action_idx_maxVal
            else:
                action_idx=random.sample(actions_idx_valid.tolist(),1)
                action_idx=action_idx[0]                                                         
            return action_idx
        
AgentController = DQNAgent()
current_state=np.array([[900,1000]],dtype="float32")
goal_state=np.array([[700,200]],dtype="float32")
current_observation = np.concatenate((current_state, goal_state), axis=0)       
current_observation = current_observation.reshape(1,2,2)
Q_value=AgentController.model.predict(AgentController.normalize_data(current_observation))
print(Q_value)

print("next q table")
next_observation = np.random.uniform(50,Y_MAX-50,(3,2,2))
Q_value_next=AgentController.model.predict(AgentController.normalize_data(next_observation))
print(Q_value_next)