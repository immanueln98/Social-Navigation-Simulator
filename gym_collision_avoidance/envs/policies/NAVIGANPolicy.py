# #!/usr/bin/env python

# from __future__ import print_function
# import os

# import math
# import numpy as np
# import sys
# ##sys.path.append('/home/sam/proj/robot_software/social_navigation/src/navigan/scripts/sgan')

# import torch
# from collections import deque, defaultdict
# from threading import Lock

# from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
# from gym_collision_avoidance.envs import Config
# from gym_collision_avoidance.envs.util import *

# from gym_collision_avoidance.envs.policies.NAVIGAN.scripts.sgan.models import TrajectoryGenerator, TrajectoryIntention
# from gym_collision_avoidance.envs.policies.NAVIGAN.scripts.sgan.various_length_models import LateAttentionFullGenerator
# from gym_collision_avoidance.envs.policies.NAVIGAN.scripts.sgan.utils import relative_to_abs

# from attrdict import AttrDict






# VERBOSE = True
# RADIUS = 0.5
# VALIDSCAN = 20.0 #20.0

# '''
# a=naviGAN()
# Loading Attention Generator...

# >>> poses = [[1,1],[3,1]]
# >>> history = [[[4,4],[3,1]],[[3,3],[3,1]],[[2,2],[3,1]]]
# >>> a.predict(0,poses,history)

# '''

# class NAVIGANPolicy(InternalPolicy):
#     def __init__(self):

#         InternalPolicy.__init__(self, str="NAVIGAN")

#         CHECKPOINT = os.path.dirname(os.path.realpath(__file__)) + '/NAVIGAN/models/benchmark_zara1_with_model.pt'

#         self.obs_rate = 0.4#0.4
#         self.goal_reached = False

#         self.dt = Config.DT

#         #self.exe_rate = 0.15
#         self.OBS_LEN = 4

#         self.predictGoal = True
        
#         if VERBOSE:
#             print('Loading Attention Generator...')
#         self.intention_generator = self.get_attention_generator(torch.load(CHECKPOINT))
#         if VERBOSE:
#             print('Done.')

#         self.max_linear_speed = 0.4 #pref_speed

#         self.is_init = False

#     def init(self,agents):
#         state_dim = 2
#         self.pos_agents = np.empty((self.n_agents, state_dim))
#         self.vel_agents = np.empty((self.n_agents, state_dim))
#         self.goal_agents = np.empty((self.n_agents, state_dim))
#         self.pref_vel_agents = np.empty((self.n_agents, state_dim))
#         self.pref_speed_agents = np.empty((self.n_agents))
        
#         self.total_agents_num = [None]*self.n_agents

#         self.agents_history = np.empty((self.n_agents, 999999, 2 )) #99999 is the empty length for history, 2 is the size for [  x, y]
 
#         self.is_init = True

#     def find_next_action(self, obs, agents, i ):

#         if not self.is_init:   #Execute one time per init (complete simulation iteration)
#             self.n_agents = len(agents)
#             self.init(agents)
#         #current_time = self.world_info().sim_time

#         agent_index = i
            
#         timestep_history_interval = 4  #dataset 0.4s per data, so only record 1 time every 4 data
#         for a in range(self.n_agents):
#             # Copy current agent positions, goal and preferred speeds into np arrays
#             self.pos_agents[a,:] = agents[a].pos_global_frame[:]
#             self.vel_agents[a,:] = agents[a].vel_global_frame[:]
#             self.goal_agents[a,:] = agents[a].goal_global_frame[:]
#             self.pref_speed_agents[a] = agents[a].pref_speed

#             timestep_now = agents[a].step_num

#             self.agents_history[a, timestep_now ] = [ agents[a].pos_global_frame[0] , agents[a].pos_global_frame[1] ]

#         ###############################Input bridge#################
#         #agents position at this moment
#         agent_poses = self.pos_agents
#         #goal of this agent
#         #robot_goal = self.goal_agents[agent_index]

#         #create intermediate waypoint for the agent, if the goal is too far from the agent.
#         displacement = self.goal_agents[agent_index] - self.pos_agents[agent_index]
#         dist_to_goal =  np.linalg.norm( displacement )

#         goal_step_threshold = 0.5
#         if dist_to_goal > goal_step_threshold:
#             #normalized to 1 in magntitude length
#             dist_next_waypoint = displacement /np.linalg.norm( displacement ,ord=1)
#             #walk 1M towards the goal direction
#             robot_goal = self.pos_agents[agent_index] + dist_next_waypoint* goal_step_threshold #0.5M
#         else:
#             #it is within 1m, use goal direction, no need for intermediate waypoint
#             robot_goal = self.goal_agents[agent_index]


        
#         #Since agent_history is arranged in [ oldest, old , new, latest ]
#         #While the section below require [ latest, new, old ,oldest ]
            
        
#         #need to flip along the timestamp axis
#         #First trim off the empty part of array, and then flip it along timestep axis, so that the latest timestep is at position 0
#         agent_poses_history=np.flip(self.agents_history[:,:timestep_now,:],1)

#         #agents_history is in the shape of [ number of agents, history_size, size of xy tuple]
        
#         #but navigan will require [history_size,  number of agent,  size of xy tuple]
#         #therefore use np.transpose(in, (1, 0, 2)) to swap first two axis to fit the required format
        
#         agent_poses_history = np.transpose(agent_poses_history, (1, 0, 2))

#         #agent_poses_history = agent_poses_history[::4] #only 1 per 4 (0.1) timesteossss

#         #############################Input bridge end#################

   
#         #Get past position history, only need newest 3 records in this case  (3 past and 1 current position = 4 poses => obs_trajectory)
#         past_poses_history = list(agent_poses_history)[: (self.OBS_LEN-1) ] #get the newest 3 record
#         #Get current position 
#         now_poses = [agent_poses]
#         #Combine current position + past 3 record => 4 in total => obs_traj
#         poses_history = now_poses + past_poses_history
#         #[ now , new, old , oldest ]
        
#         #if poses_history is less than 4, clone more of the oldest data and append that to the end
#         if len(poses_history)< (self.OBS_LEN):               
#             poses_history.extend(  [ poses_history[-1]] * ( self.OBS_LEN - len(poses_history) ) )

#         #[ now , new, old , oldest ]

#         #reverse the order of time to match the style of navigan....
#         #becomes [oldest , old, new ,now]
#         poses_history = np.flip(poses_history.copy(),0) #0 is stupid!!! #no need flip?! Seems navigan [now, old , oldest], with largest timestep as first element because it is latest
        
      
#         rearranged_history = [np.nan] * self.OBS_LEN


#         for i in range(self.OBS_LEN):
#             #process the position of all agents in Time[i], re-arrange such that  poses = self , other_0, other_1, 2,3,4...
#             past_all_agents_poses = poses_history[-1]     
#             past_robot_pose = past_all_agents_poses[agent_index]
# ##            print("past_robot_pose")
# ##            print(past_robot_pose)

           
#             past_other_agent_poses = past_all_agents_poses.copy()
#             past_other_agent_poses = list(past_other_agent_poses)
#             past_other_agent_poses.pop(agent_index)   #remove agent's position, leaving only other's position

#             ####exaggeration...  exaggerate peds agents, by comparing to robot
#             #################
#             for j in range(len(past_other_agent_poses)):
#                 #BIG mistake! the position tuple for agent_index is removed before referenced!
#                 #displacement = np.array( past_robot_pose[0] ) - np.array( past_other_agent_poses[j] )

#                 #Fixed
# ##                print("np.array( past_robot_pose )")
# ##                print(np.array( past_robot_pose ))
# ##                print("np.array( past_other_agent_poses[j] )")
# ##                print(np.array( past_other_agent_poses[j] ))
                
#                 displacement = (np.array( past_robot_pose ) - np.array( past_other_agent_poses[j] )+ 0.00001)
                
#                 #should not be using newest , should be relative to oldest position
#                 #newest_xxxx_xxx are not in use currently, consider remove it
                
#                 #displacement = np.array( newest_robot_pose[0] ) - np.array( newest_other_agent_poses[j] )   
#                 norm_disp = (np.sqrt(displacement[0]**2+displacement[1]**2)+0.00001)

#                 if i == (  self.OBS_LEN -1 ):

#                     if norm_disp > 2:
#                         displacement *= 0.5/norm_disp   #2
#                     elif norm_disp > 1:
#                         displacement *= 0.5/norm_disp   #0.9
#                     elif norm_disp > 0.5:
#                         displacement *= 0.4/norm_disp    #0.7
#                     else:
#                         #displacement /= 2
#                         displacement = 0
#                         pass
#                     pass

#                 else:
#                     displacement = 0

# ##                displacement = 0
          
#             ##############
#                 past_other_agent_poses[j] = list( np.array( past_other_agent_poses[j] ) + displacement)

#             #################
#             #Output array of all poses in time series,  each element in array contain all agent poses, [0] element in rearranged_history is the oldest 
#             rearranged_history[i] = np.concatenate(([past_robot_pose], past_other_agent_poses))
# ##            print("combined=>")
# ##            print(rearranged_history[i])

            


            
# #############################################################        
# ##[INFO] [1587987020.665589]: TTTTTTTTTTTTTTTT
# ##[INFO] [1587987020.666708]: [[1.0564772892265575, -2.9439042539070246], [1.1145057294792293, -5.99606735577477], [1.1153218711699553, 2.640532728710876e-06]]
# ##obs_traj => [[[ 1.13049939e+00 -2.86988351e+00]
# ##  [ 1.39572241e+00 -5.99669114e+00]
# ##  [ 1.39613869e+00 -2.94107679e-06]]
# ##
# ## [[ 1.11235920e+00 -2.88802337e+00]
# ##  [ 1.32198718e+00 -5.99649834e+00]
# ##  [ 1.32250828e+00 -1.21340563e-06]]
# ##
# ## [[ 1.07441882e+00 -2.92596304e+00]
# ##  [ 1.22978925e+00 -5.99629044e+00]
# ##  [ 1.23044144e+00  6.46133517e-07]]
# ##
# ## [[ 1.05647729e+00 -2.94390425e+00]
# ##  [ 1.11450573e+00 -5.99606736e+00]
# ##  [ 1.11532187e+00  2.64053273e-06]]]
# ##shape (4, 3, 2)
# ##
# ########################################################
      
#         obs_traj = np.array(rearranged_history)

# ##        try:
# ##            if np.isnan(np.sum(obs_traj)):
# ##                print("nan")
# ##                return [0,0]
# ##        except TypeError:
# ##            print("TypeError")
# ##            return [0,0]


        

# ##        print("shape",np.array(obs_traj).shape)
            
# ##            obs_traj = [robot_pos]+peds_array
# ##            obs_traj = np.array(obs_traj).transpose((1,0,2))

#         self.goalData = robot_goal
#         goal = np.array([self.goalData[0], self.goalData[1]])

# ##        rospy.loginfo("[goal]")
# ##        rospy.loginfo(goal)
        
#         goal = torch.from_numpy(goal).type(torch.float)

#         obs_traj = torch.from_numpy(obs_traj).type(torch.float)
#         obs_traj_rel = obs_traj - obs_traj[0,:,:]

# ##        rospy.loginfo("[obs_traj]")
# ##        rospy.loginfo(obs_traj)
# ##
# ##        rospy.loginfo("[obs_traj_rel]")
# ##        rospy.loginfo(obs_traj_rel)

#         seq_start_end = torch.from_numpy(np.array([[0,obs_traj.shape[1]]]))

#         goals_rel = goal - obs_traj[0,0,:]
#         goals_rel = goals_rel.repeat(1,obs_traj.shape[1],1)


#         # move everything to GPU
#         obs_traj = obs_traj.cuda()
#         obs_traj_rel = obs_traj_rel.cuda()
#         seq_start_end = seq_start_end.cuda()
#         goals_rel = goals_rel.cuda()

#         pred_traj_fake = self.feedforward(obs_traj, obs_traj_rel, seq_start_end, goals_rel)
#         # print(pred_traj_fake.size())
#         ptf = pred_traj_fake[:,0,:].cpu().numpy()  #[12,2]

#         # pick the 3rd position as robot predicted step
#         # self.next_waypoint = Point(ptf[2,0],ptf[2,1], 0)
#         self.next_waypoint = [ptf[3,0],ptf[3,1], 0]   #change to 1st [psotopm as robot prediction step

        

#         #reach the goal check  need to be tuned
#         #rospy.loginfo("waypoint is x: {}, y:{}".format(self.next_waypoint.x, self.next_waypoint.y))

#         #####!!!! Connect agent index

#         #dont worry, agents already included all agents
#         #dont worry, this agent can already see all others, and is only using the policy for itself and calc its own speed

#         ####!!!! Internal policy that use external dynamics
#         ####!!!!! Set state to set position of this agent and directly use external dynamics to bypass setting up velocity actions.

#         #Directly update agent's position
#         position_x = np.clip( ( (self.next_waypoint[0] - agents[agent_index].pos_global_frame[0])/5) , -0.2, 0.2) + agents[agent_index].pos_global_frame[0]
#         position_y = np.clip( ( (self.next_waypoint[1] - agents[agent_index].pos_global_frame[1])/5) , -0.2, 0.2) + agents[agent_index].pos_global_frame[1]

# ##        if   (self.next_waypoint[0] > agents[agent_index].pos_global_frame[0]):   #0.2 original
# ##            position_x = agents[agent_index].pos_global_frame[0] + 0.1
# ##        elif (self.next_waypoint[0] < agents[agent_index].pos_global_frame[0]):
# ##            position_x = agents[agent_index].pos_global_frame[0] - 0.1
# ##        else:
# ##            position_x = agents[agent_index].pos_global_frame[0]
# ##
# ##        if   (self.next_waypoint[1] > agents[agent_index].pos_global_frame[1]):
# ##            position_y = agents[agent_index].pos_global_frame[1] + 0.1
# ##        elif (self.next_waypoint[1] < agents[agent_index].pos_global_frame[1]):
# ##            position_y = agents[agent_index].pos_global_frame[1] - 0.1
# ##        else:
# ##            position_y = agents[agent_index].pos_global_frame[1]

#         #agents[agent_index].set_state( position_x , position_y )
#         agents[agent_index].set_state( self.next_waypoint[0] , self.next_waypoint[1] )

#         resultant_speed_global_frame         = agents[agent_index].speed_global_frame
#         resultant_delta_heading_global_frame = agents[agent_index].delta_heading_global_frame

#         #Although documentation and code comment mentioned that action is consisted with  [heading delta, speed]
#         #But in reality, the format of action is [speed, heading_delta]
#         action = [ resultant_speed_global_frame , resultant_delta_heading_global_frame ]

#         if agent_index ==0:
#             print("Agent "+str(agent_index)+" obs_traj =>",obs_traj[-3:, agent_index])

#             print("time_remaining_to_reach_goal")
#             print(agents[agent_index].time_remaining_to_reach_goal)

#             print("ran_out_of_time")
#             print(agents[agent_index].ran_out_of_time)

#             print("self.next_waypoint")
#             print([ptf[2,0],ptf[2,1]])

#             print("final")
#             print([position_x,position_y])

#             print("action")
#             print(action)
#         return action


#         # late attention model by full state (actually used)
#     def get_attention_generator(self, checkpoint, best=False):
#         args = AttrDict(checkpoint['args'])
#         generator = LateAttentionFullGenerator(
#             goal_dim=(2,),
#             obs_len=args.obs_len,
#             pred_len=args.pred_len,
#             embedding_dim=args.embedding_dim,
#             encoder_h_dim=args.encoder_h_dim_g,
#             decoder_h_dim=args.decoder_h_dim_g,
#             mlp_dim=args.mlp_dim,
#             num_layers=args.num_layers,
#             noise_dim=args.noise_dim,
#             noise_type=args.noise_type,
#             noise_mix_type=args.noise_mix_type,
#             pooling_type=args.pooling_type,
#             pool_every_timestep=args.pool_every_timestep,
#             dropout=args.dropout,
#             bottleneck_dim=args.bottleneck_dim,
#             neighborhood_size=args.neighborhood_size,
#             grid_size=args.grid_size,
#             batch_norm=args.batch_norm,
#             spatial_dim=2)
#         if best:
#             generator.load_state_dict(checkpoint['g_waypointbest_state'])
#         else:
#             generator.load_state_dict(checkpoint['g_state'])
#         generator.cuda()
#         generator.train()
#         return generator



#     def reachGoalCheck(self, robotPosition, goalPosition, _r=RADIUS):
#         if (robotPosition.x-goalPosition[0])**2 + (robotPosition.y-goalPosition[1])**2 < _r**2:
#             return True
#         else:
#             return False



#     def feedforward(self, obs_traj, obs_traj_rel, seq_start_end, goals_rel):
#         """
#         obs_traj: torch.Tensor([4, num_agents, 2])
#         obs_traj_rel: torch.Tensor([4, num_agents, 2])
#         seq_start_end: torch.Tensor([batch_size, 2]) #robot+#pedstrains
#         goals_rel: torch.Tensor([1, num_agents, 2])
#         """

#         with torch.no_grad():
#             pred_traj_fake_rel, _ = self.intention_generator(obs_traj, obs_traj_rel, seq_start_end, goal_input=goals_rel)
#             pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])
#         return pred_traj_fake


import os
import math
import numpy as np
import torch
from collections import deque, defaultdict
from threading import Lock

from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import *

from gym_collision_avoidance.envs.policies.NAVIGAN.scripts.sgan.models import TrajectoryGenerator, TrajectoryIntention
from gym_collision_avoidance.envs.policies.NAVIGAN.scripts.sgan.various_length_models import LateAttentionFullGenerator
from gym_collision_avoidance.envs.policies.NAVIGAN.scripts.sgan.utils import relative_to_abs

from attrdict import AttrDict

from itertools import compress

VERBOSE = True
RADIUS = 0.5
VALIDSCAN = 20.0

class NAVIGANPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="NAVIGAN")

        CHECKPOINT = os.path.dirname(os.path.realpath(__file__)) + '/NAVIGAN/models/benchmark_zara1_with_model.pt'

        self.obs_rate = 0.4
        self.goal_reached = False

        self.dt = Config.DT
        self.OBS_LEN = 4

        self.predictGoal = True

        if VERBOSE:
            print('Loading Attention Generator...')
        self.intention_generator = self.get_attention_generator(torch.load(CHECKPOINT))
        if VERBOSE:
            print('Done.')

        self.max_linear_speed = 0.4  # pref_speed

        self.is_init = False

    def init(self, agents):
        state_dim = 2
        self.pos_agents = []
        self.vel_agents = []
        self.goal_agents = []
        self.pref_vel_agents = []
        self.pref_speed_agents = []

        self.total_agents_num = []
        self.agents_history = []

        self.is_init = True

    def find_next_action(self, obs, agents, target_agent_index, full_agent_list, active_agent_mask):

        # Get indices of active agents
        active_indices = [i for i, active in enumerate(active_agent_mask) if active]
        # Map target_agent_index to its index in active_agents
        agent_index_in_active = active_indices.index(target_agent_index)
        # Get the list of active agents
        active_agents = [full_agent_list[i] for i in active_indices]
        self.n_agents = len(active_agents)

        if not self.is_init:
            self.init(active_agents)

        # Ensure data structures are sized correctly for active agents
        if len(self.pos_agents) < self.n_agents:
            # Extend the lists to match the number of active agents
            for _ in range(self.n_agents - len(self.pos_agents)):
                self.pos_agents.append(None)
                self.vel_agents.append(None)
                self.goal_agents.append(None)
                self.pref_vel_agents.append(None)
                self.pref_speed_agents.append(None)
                self.total_agents_num.append(None)
                self.agents_history.append([])

        timestep_now = active_agents[agent_index_in_active].step_num

        # Update agent data
        for i in range(self.n_agents):
            # Update positions, velocities, goals, and preferred speeds
            self.pos_agents[i] = active_agents[i].pos_global_frame[:]
            self.vel_agents[i] = active_agents[i].vel_global_frame[:]
            self.goal_agents[i] = active_agents[i].goal_global_frame[:]
            self.pref_speed_agents[i] = active_agents[i].pref_speed

            # Update position history
            if len(self.agents_history[i]) <= timestep_now:
                self.agents_history[i].extend([None] * (timestep_now - len(self.agents_history[i]) + 1))
            self.agents_history[i][timestep_now] = [
                active_agents[i].pos_global_frame[0],
                active_agents[i].pos_global_frame[1],
            ]

        # Prepare the observation trajectory
        agent_poses = np.array(self.pos_agents)  # Now contains positions of active agents only

        # Process agent histories
        agent_histories = []
        max_history_len = 0
        for i in range(self.n_agents):
            history = self.agents_history[i]
            if history is None:
                history = []
            # Fill missing history with the last known position
            if len(history) < timestep_now + 1:
                last_known_pos = history[-1] if history else [0.0, 0.0]
                history.extend([last_known_pos] * (timestep_now + 1 - len(history)))
            # Collect positions up to current timestep
            positions = history[:timestep_now + 1]
            agent_histories.append(positions)
            if len(positions) > max_history_len:
                max_history_len = len(positions)

        # Ensure all histories are the same length
        for i in range(self.n_agents):
            positions = agent_histories[i]
            if len(positions) < max_history_len:
                last_known_pos = positions[-1]
                padding = [last_known_pos] * (max_history_len - len(positions))
                agent_histories[i].extend(padding)

        # Convert histories to numpy array
        agent_histories = np.array(agent_histories)  # Shape: [n_agents, history_len, 2]

        # Transpose to match expected format: [history_len, n_agents, 2]
        agent_histories = np.transpose(agent_histories, (1, 0, 2))

        # Prepare the observation trajectory with the required observation length
        obs_traj_len = self.OBS_LEN
        if agent_histories.shape[0] < obs_traj_len:
            # Not enough history, pad with the earliest positions
            padding_len = obs_traj_len - agent_histories.shape[0]
            padding = np.tile(agent_histories[0:1, :, :], (padding_len, 1, 1))
            obs_traj = np.concatenate((padding, agent_histories), axis=0)
        else:
            obs_traj = agent_histories[-obs_traj_len:, :, :]

        # Proceed with the rest of the method as in the original code
        # Process goal
        displacement = self.goal_agents[agent_index_in_active] - self.pos_agents[agent_index_in_active]
        dist_to_goal = np.linalg.norm(displacement)
        goal_step_threshold = 0.5
        if dist_to_goal > goal_step_threshold:
            dist_next_waypoint = displacement / (np.linalg.norm(displacement, ord=1) + 1e-6)
            robot_goal = self.pos_agents[agent_index_in_active] + dist_next_waypoint * goal_step_threshold
        else:
            robot_goal = self.goal_agents[agent_index_in_active]

        # Prepare input for the model
        obs_traj = torch.from_numpy(obs_traj).type(torch.float)
        obs_traj_rel = obs_traj - obs_traj[0, :, :]

        goal = torch.from_numpy(robot_goal).type(torch.float)
        goals_rel = goal - obs_traj[0, agent_index_in_active, :]
        goals_rel = goals_rel.repeat(1, self.n_agents, 1)

        seq_start_end = torch.from_numpy(np.array([[0, obs_traj.shape[1]]]))

        # Move everything to GPU
        obs_traj = obs_traj.cuda()
        obs_traj_rel = obs_traj_rel.cuda()
        seq_start_end = seq_start_end.cuda()
        goals_rel = goals_rel.cuda()

        # Ensure tensors are contiguous
        obs_traj = obs_traj.contiguous()
        obs_traj_rel = obs_traj_rel.contiguous()

        pred_traj_fake = self.feedforward(obs_traj, obs_traj_rel, seq_start_end, goals_rel)
        ptf = pred_traj_fake[:, agent_index_in_active, :].cpu().numpy()

        # Select the next waypoint
        self.next_waypoint = [ptf[3, 0], ptf[3, 1], 0]

        # Update the agent's state
        agents[target_agent_index].set_state(self.next_waypoint[0], self.next_waypoint[1])

        resultant_speed_global_frame = agents[target_agent_index].speed_global_frame
        resultant_delta_heading_global_frame = agents[target_agent_index].delta_heading_global_frame

        action = [resultant_speed_global_frame, resultant_delta_heading_global_frame]

        return action


    def get_attention_generator(self, checkpoint, best=False):
        args = AttrDict(checkpoint['args'])
        generator = LateAttentionFullGenerator(
            goal_dim=(2,),
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm,
            spatial_dim=2)
        if best:
            generator.load_state_dict(checkpoint['g_waypointbest_state'])
        else:
            generator.load_state_dict(checkpoint['g_state'])
        generator.cuda()
        generator.train()
        return generator

    def reachGoalCheck(self, robotPosition, goalPosition, _r=RADIUS):
        if (robotPosition.x - goalPosition[0]) ** 2 + (robotPosition.y - goalPosition[1]) ** 2 < _r ** 2:
            return True
        else:
            return False

    def feedforward(self, obs_traj, obs_traj_rel, seq_start_end, goals_rel):
        with torch.no_grad():
            # Ensure tensors are contiguous
            obs_traj = obs_traj.contiguous()
            obs_traj_rel = obs_traj_rel.contiguous()
            pred_traj_fake_rel, _ = self.intention_generator(obs_traj, obs_traj_rel, seq_start_end,
                                                             goal_input=goals_rel)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])
        return pred_traj_fake
