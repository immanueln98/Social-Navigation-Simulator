import argparse
import os
import torch

from attrdict import AttrDict

import numpy as np

from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import *

from gym_collision_avoidance.envs.policies.Social_nav.model_loaders import get_combined_generator
from gym_collision_avoidance.envs.policies.Social_nav.utils import relative_to_abs, abs_to_relative


from gym_collision_avoidance.envs.policies.SOCIALGAN.socialgan.data.loader import data_loader, custom_data_loader
# from gym_collision_avoidance.envs.policies.SOCIALGAN.socialgan.models import TrajectoryGenerator
# from gym_collision_avoidance.envs.policies.SOCIALGAN.socialgan.losses import displacement_error, final_displacement_error
# from gym_collision_avoidance.envs.policies.SOCIALGAN.socialgan.utils import relative_to_abs, get_dset_path

# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='models/sgan-models')
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

class SocialNavPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="SOCIALNAV")
        self.dt = Config.DT
        self.obs_seq_len=8
        self.pred_seq_len=12
       
        self.is_init = False
        


        # #print("load 1")
        self.checkpoint = torch.load("/home/immanuel/Social-Navigation-Simulator/gym_collision_avoidance/envs/policies/Social_nav/SocialNavPoolNetIntentZ1.pt")
        # #print("load 2")        
        # self.generator = self.get_generator(self.checkpoint)
        # #print("load 3")
        # self._args = AttrDict(self.checkpoint['args'])

        self.generator = get_combined_generator(self.checkpoint)
        self.OBS_LEN = self.generator.goal.obs_len
        self.pred_seq_len = self.generator.goal.pred_len

    def init(self,agents):
        # Initialise agents far away to avoid influencing Husky planned path
        state_dim = 2
        self.pos_agents = np.empty((self.n_agents, state_dim))
        self.vel_agents = np.empty((self.n_agents, state_dim))
        self.goal_agents = np.empty((self.n_agents, state_dim))
        self.pref_vel_agents = np.empty((self.n_agents, state_dim))
        self.pref_speed_agents = np.empty((self.n_agents))
        
        self.total_agents_num = [None]*self.n_agents


        self.agents_history = np.empty((self.n_agents, 999999, 2 )) #99999 is the empty length for history, 2 is the size for [  x, y]
 
        self.is_init = True

    def find_next_action(self, obs, agents, i):
        if not self.is_init:   #Execute one time per init (complete simulation iteration)
            self.n_agents = len(agents)
            self.init(agents)
        #current_time = self.world_info().sim_time

        agent_index = i
        print("obs: ",obs)

        timestep_history_interval = 4  #dataset 0.4s per data, so only record 1 time every 4 data
        for a in range(self.n_agents):
            # Copy current agent positions, goal and preferred speeds into np arrays
            self.pos_agents[a,:] = agents[a].pos_global_frame[:]
            self.vel_agents[a,:] = agents[a].vel_global_frame[:]
            self.goal_agents[a,:] = agents[a].goal_global_frame[:]
            self.pref_speed_agents[a] = agents[a].pref_speed

            timestep_now = agents[a].step_num

            self.agents_history[a, timestep_now ] = [ agents[a].pos_global_frame[0] , agents[a].pos_global_frame[1] ]

        #agents position at this moment
        agent_poses = self.pos_agents
        #goal of this agent
        #robot_goal = self.goal_agents[agent_index]
        
        #create intermediate waypoint for the agent, if the goal is too far from the agent.
        displacement = self.goal_agents[agent_index] - self.pos_agents[agent_index]
        dist_to_goal =  np.linalg.norm( displacement )

        # goal_step_threshold = 0.2
        # if dist_to_goal > goal_step_threshold:
        #     #normalized to 1 in magntitude length
        #     dist_next_waypoint = displacement /np.linalg.norm( displacement ,ord=1)
        #     #walk 1M towards the goal direction
        #     robot_goal = self.pos_agents[agent_index] + dist_next_waypoint* goal_step_threshold #0.5M
        # else:
        #     #it is within 1m, use goal direction, no need for intermediate waypoint
        #     robot_goal = self.goal_agents[agent_index]
        robot_goal = self.goal_agents[agent_index]
        print("robot_goal: ",robot_goal)
        print("agents goal: ", agents[agent_index].goal_global_frame)
        print("agent velocity: ",agents[agent_index].vel_global_frame)
        print("agent collided: ", agents[agent_index].in_collision)
        #agents_history is in the shape of [ number of agents, history_size, size of xy tuple]
        
        #but navigan will require [history_size,  number of agent,  size of xy tuple]
        #therefore use np.transpose(in, (1, 0, 2)) to swap first two axis to fit the required format
        
        agent_poses_history = np.transpose(self.agents_history, (1, 0, 2))
        #Get past position history, only need newest 3 records in this case  (3 past and 1 current position = 4 poses => obs_trajectory)
        #past_poses_history = list(agent_poses_history)[: (self.OBS_LEN-1) ] #get the newest 3 record
        past_poses_history = list(agent_poses_history)[:]
        #Get current position 
        now_poses = [agent_poses]
        #Combine current position + past 3 record => 4 in total => obs_traj
        poses_history = now_poses + past_poses_history
        #[ now , new, old , oldest ]
        #[ now , new, old , oldest ]

        #reverse the order of time to match the style of navigan....
        #becomes [oldest , old, new ,now]
        poses_history = np.flip(poses_history.copy(),0)
        rearranged_history = [np.nan] * self.OBS_LEN

        print("OBS_LEN: ",self.OBS_LEN)
        for i in range(self.OBS_LEN):
            #process the position of all agents in Time[i], re-arrange such that  poses = self , other_0, other_1, 2,3,4...
            past_all_agents_poses = poses_history[-1]     
            past_robot_pose = past_all_agents_poses[agent_index]
            print("past_robot_pose")
            print(past_robot_pose)

           
            past_other_agent_poses = past_all_agents_poses.copy()
            past_other_agent_poses = list(past_other_agent_poses)
            past_other_agent_poses.pop(agent_index)   #remove agent's position, leaving only other's position

            ####exaggeration...  exaggerate peds agents, by comparing to robot
            #################
            for j in range(len(past_other_agent_poses)):
                #BIG mistake! the position tuple for agent_index is removed before referenced!
                #displacement = np.array( past_robot_pose[0] ) - np.array( past_other_agent_poses[j] )

                #Fixed
##                print("np.array( past_robot_pose )")
##                print(np.array( past_robot_pose ))
##                print("np.array( past_other_agent_poses[j] )")
##                print(np.array( past_other_agent_poses[j] ))
                
                displacement = (np.array( past_robot_pose ) - np.array( past_other_agent_poses[j] )+ 0.00001)
                
                #should not be using newest , should be relative to oldest position
                #newest_xxxx_xxx are not in use currently, consider remove it
                
                #displacement = np.array( newest_robot_pose[0] ) - np.array( newest_other_agent_poses[j] )   
                norm_disp = (np.sqrt(displacement[0]**2+displacement[1]**2)+0.00001)

                if i == (  self.OBS_LEN -1 ):

                    if norm_disp > 2:
                        displacement *= 0.5/norm_disp   #2
                    elif norm_disp > 1:
                        displacement *= 0.5/norm_disp   #0.9
                    elif norm_disp > 0.5:
                        displacement *= 0.4/norm_disp    #0.7
                    else:
                        #displacement /= 2
                        displacement = 0
                        pass
                    pass

                else:
                    displacement = 0

##                displacement = 0
          
            ##############
                past_other_agent_poses[j] = list( np.array( past_other_agent_poses[j] + displacement ) )

            #################
            #Output array of all poses in time series,  each element in array contain all agent poses, [0] element in rearranged_history is the oldest 
            rearranged_history[i] = np.concatenate(([past_robot_pose], past_other_agent_poses))

        obs_traj = np.array(rearranged_history)
        self.goalData = robot_goal
        goal = np.array([self.goalData[0], self.goalData[1]])
        print("goal data: ", self.goalData)

        goal = torch.from_numpy(goal).type(torch.float)
        #goal = goal.view(-1)
        obs_traj = torch.from_numpy(obs_traj).type(torch.float)
        # Extract the robot's trajectory over time
        robot_traj = obs_traj[:, 0:1, :]  # Shape: (T, 1, C)
        
        obs_traj_rel = obs_traj - robot_traj
    
        with torch.no_grad():
            seq_start_end = torch.tensor([0, obs_traj.shape[1]]).unsqueeze(0)

        #seq_start_end = torch.from_numpy(np.array([[0,obs_traj.shape[1]]]))

            # goals_rel = goal - obs_traj[-1,0,:]
            # print("goals_rel", goals_rel)
            goal_state = torch.zeros((1, 1, 2))
            goal_state[0, 0, 0] = goal[0] - obs_traj[-1, 0, 0].item()
            goal_state[0, 0, 1] = goal[1] - obs_traj[-1, 0, 1].item()
            print("goal_state", goal_state)
            
            # move everything to GPU
            # obs_traj = obs_traj.cuda()
            # obs_traj_rel = obs_traj_rel.cuda()
            # seq_start_end = seq_start_end.cuda()
            # goals_rel = goals_rel.cuda()
            #obs_traj_rel = abs_to_relative(obs_traj)
            #obs_traj_rel = obs_traj - obs_traj[0,:,:]
            pred_traj_fake = self.generator(obs_traj, obs_traj_rel, seq_start_end, goal_state)

            start_pos = obs_traj[-1]
            #ptf = relative_to_abs(pred_traj_fake, start_pos=start_pos)
            ptf = relative_to_abs(pred_traj_fake, start_pos)
            ptf = ptf[:,0,:].cpu().numpy()
        

        mean_waypoint = np.mean(ptf, axis=0)
        # self.next_waypoint = [mean_waypoint[0],mean_waypoint[1]]   #change to 1st [psotopm as robot prediction step
        self.next_waypoint = [ptf[3,0],ptf[3,1]]  
        # print("waypoint: ",self.next_waypoint)
        # print("agent index:",agent_index)
        # print("agent position:",agents[agent_index].pos_global_frame)
        
        # position_x = np.clip( ( (self.next_waypoint[0] - agents[agent_index].pos_global_frame[0])/5) , -0.2, 0.2) + agents[agent_index].pos_global_frame[0]
        # position_y = np.clip( ( (self.next_waypoint[1] - agents[agent_index].pos_global_frame[1])/5) , -0.2, 0.2) + agents[agent_index].pos_global_frame[1]

        # print("position: ",position_x, position_y)
        # # agents[agent_index].set_state( position_x , position_y )
        # agents[agent_index].set_state( self.next_waypoint[0] , self.next_waypoint[1] )

        # resultant_speed_global_frame         = agents[agent_index].speed_global_frame
        # resultant_delta_heading_global_frame = agents[agent_index].delta_heading_global_frame

        # #Although documentation and code comment mentioned that action is consisted with  [heading delta, speed]
        # #But in reality, the format of action is [speed, heading_delta]
        # action = [ resultant_speed_global_frame , resultant_delta_heading_global_frame ]

        
        # print("Agent "+str(agent_index)+" obs_traj =>",obs_traj[-3:, agent_index])

        # print("time_remaining_to_reach_goal")
        # print(agents[agent_index].time_remaining_to_reach_goal)

        # print("ran_out_of_time")
        # print(agents[agent_index].ran_out_of_time)

        # print("self.next_waypoint")
        # print([ptf[5,0],ptf[5,1]])

        # print("final")
        # print([position_x,position_y])

        # print("action")
        # print(action)
        # return action

        goal_direction = self.next_waypoint - agents[agent_index].pos_global_frame
        self.dist_to_goal = math.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / agents[agent_index].dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg

        ref_prll_angle_global_frame = np.arctan2(ref_prll[1],
                                                 ref_prll[0])
        heading_ego_frame = wrap( agents[agent_index].heading_global_frame -
                                      ref_prll_angle_global_frame)

    

        vel_global_frame = (( goal_direction)/4) / agents[agent_index].dt_nominal

        speed_global_frame = np.linalg.norm(vel_global_frame) 
        print("calc speed")
        print(speed_global_frame)
        #if speed_global_frame > agents[agent_index].pref_speed: speed_global_frame = agents[agent_index].pref_speed

        if speed_global_frame > 1.5: speed_global_frame = 1.5
        if speed_global_frame < 0.5: speed_global_frame = 0.5

        #But in reality, the format of action is [speed, heading_delta]

        action = np.array([speed_global_frame, -heading_ego_frame])
        print("action")
        print(action)
       
        return action

