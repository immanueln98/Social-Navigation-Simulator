import pathlib
import time
import sys
import torch

from model_loaders import get_combined_generator
from utils import relative_to_abs, abs_to_relative

import numpy as np

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseWithCovarianceStamped

_DEVICE_ = torch.device("cpu")


class Navigator:

    def __init__(self, model_path, agents=9, rate=1):
        self.last_times = np.ones(agents) * time.perf_counter()
        self.goal = [0, 0, 0]
        self.abs_goal = [0, 0]
        if pathlib.Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=_DEVICE_)
        else:
            print('Invalid model path.')
            sys.exit(0)
        self.generator = get_combined_generator(checkpoint)
        self.obs_len = self.generator.goal.obs_len
        self.pred_len = self.generator.goal.pred_len
        self.agents = agents
        # TODO consider best solution to init agent values
        # Initialise agents far away to avoid influencing Husky planned path
        self.obs_traj = torch.ones((self.obs_len, 1 + agents, 2), device=_DEVICE_) * 100
        self.obs_traj[::, 0] = 0
        self.last_callback = time.perf_counter()
        self.odom_last_callback = time.perf_counter()
        self.rate_value = rate
        self.odom = None

    def sleep(self):
        self.rate.sleep()

    def goal_step(self, predictions, step_horizon=11):
        """predictions: absolute coordinate predictions from sgan network of shape(12, no.agents, 2)
            Row 0 is predictions for husky.
            step_horizon: index of predictions to send as goal to move_base_mapless_demo controller"""

        # pose = Pose()
        w = 1
        # Inver x value to deal with network bias issue. To be fixed
        x = predictions[step_horizon, 0, 0]
        y = predictions[step_horizon, 0, 1]

        # self.publisher.publish(pose)
        
    def object_update(self, object_track, object_id):
        #update oldest agent data when above agent limit
        if object_id > 9:
            object_id = object_id%9
            
        object_track = torch.from_numpy(object_track)
        self.obs_traj[:-1, 1 + object_id] = self.obs_traj.clone()[1:, 1 + object_id]
        self.obs_traj[-1, 1 + object_id, 0] = object_track[-1,0]
        self.obs_traj[-1, 1 + object_id, 1] = object_track[-1,1]
    	

    def callback(self, tracked_pts):
        """update_obs_traj when a new list msg of tracked pts are received from the object tracker
        NOTE: 1 is added to agent index because obs_traj row zero is for husky and is updated via wheel odom"""
        if time.perf_counter() - self.last_times[int(tracked_pts.z)] < 0.4:
            return
        # Slide selected agent's points fwd a timestep
        self.obs_traj[:-1, 1 + int(tracked_pts.z)] = self.obs_traj.clone()[1:, 1 + int(tracked_pts.z)]
        # # From agent 0 (Husky) to agent 9 update
        # # x and y coordinate from object tracker
        # # point.z value contains agent id no.
        # TODO flipping x and -y for now because of wrong axes from tracker
        self.obs_traj[-1, 1 + int(tracked_pts.z), 0] = tracked_pts.x
        # + self.obs_traj[0, 0, 0]
        # + self.obs_traj[-1, 0, 0].item()
        self.obs_traj[-1, 1 + int(tracked_pts.z), 1] = tracked_pts.y
        self.last_times[int(tracked_pts.z)] = time.perf_counter()
        self.last_callback = time.perf_counter()
        self.callback_status = True

    def goal_update(self, goal):
        """update_obs_traj when a new list msg of tracked pts are received from the object tracker"""
        # limit update rate to self.rate_value
        self.abs_goal = [goal[0], goal[1]]
        self.goal = goal
        self.goal[0] -= self.obs_traj[-1, 0, 0].item()
        self.goal[1] -= self.obs_traj[-1, 0, 1].item()
        self.goal_status = True

    def amcl_callback(self, pose):
        """Update obs_traj when a new pose from AMCL is received."""
        # Limit update rate to 2.5 Hz (or any desired rate)
        if 1 / (time.perf_counter() - self.odom_last_callback) > 2.5:
            return
        
        # Store the pose data from AMCL
        #self.odom = amcl_pose.pose
        pose_= torch.from_numpy(pose)

        # Update the observed trajectory with the new pose x and y
        self.obs_traj[:-1, 0] = self.obs_traj.clone()[1:, 0]

        # Using the AMCL pose, 
        self.obs_traj[-1, 0, 0] = pose_[0]
        self.obs_traj[-1, 0, 1] = pose_[1]

        # Indicate that the callback has been successfully processed
        self.odom_callback_status = True

        # Update the last callback time to control the rate
        self.odom_last_callback = time.perf_counter()

    def seek_live_goal(self, agent_id=0, x=40, y=10):
        with torch.no_grad():
            seq_start_end = torch.tensor([0, self.obs_traj.shape[1]], device=_DEVICE_).unsqueeze(0)

            goal_state = torch.zeros((1, 1, 2), device=_DEVICE_)
            goal_state[0, agent_id, 0] = self.goal[0] - self.obs_traj[-1, 0, 0].item()
            goal_state[0, agent_id, 1] = self.goal[1] - self.obs_traj[-1, 0, 1].item()
            goal_state[0, agent_id, 1] = -goal_state[0, agent_id, 1]

            # TODO Resolve pred imbalance for now flip x - axis
            obs_traj_rel = abs_to_relative(self.obs_traj)
            pred_traj_fake_rel = self.generator(self.obs_traj, obs_traj_rel, seq_start_end, goal_state, goal_aggro=0.5)

            start_pos = self.obs_traj[-1]
            ptfa = relative_to_abs(pred_traj_fake_rel, start_pos=start_pos)
            # Make separate variable to plot. Flipped along x-axis to look visually correct.
            pred_traj_to_plot = ptfa.clone()
            pred_traj_to_plot[::, ::, 0] = -pred_traj_to_plot[::, ::, 0]

            obs_traj_to_plot = self.obs_traj.clone()
            obs_traj_to_plot[::, ::, 0] = -obs_traj_to_plot[::, ::, 0]

            #self.plotter.xlim = [obs_traj_to_plot[-1, 0, 0] + -5, obs_traj_to_plot[-1, 0, 0] + 5]
            #self.plotter.ylim = [obs_traj_to_plot[-1, 0, 1] + -5, obs_traj_to_plot[-1, 0, 1] + 5]


            numpy_out=ptfa.cpu().numpy()

        return numpy_out
