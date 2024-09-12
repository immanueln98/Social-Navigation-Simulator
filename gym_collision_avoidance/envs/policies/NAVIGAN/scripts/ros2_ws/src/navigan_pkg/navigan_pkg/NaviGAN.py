import sys
sys.path.append('/home/immanuel/Social-Navigation-Simulator/gym_collision_avoidance/envs/policies/NAVIGAN/scripts/sgan')
import rclpy
from rclpy.node import Node
import math
import numpy as np
import torch
from collections import deque, defaultdict
from threading import Lock
from people_msgs.msg import People

from sgan.models import TrajectoryGenerator, TrajectoryIntention
from sgan.various_length_models import LateAttentionFullGenerator
from sgan.utils import relative_to_abs

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Pose, Twist, PointStamped, Point
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Joy


VERBOSE = True
RADIUS = 0.5
VALIDSCAN = 20.0

def get_attention_generator(checkpoint, best=False):
    args = checkpoint['args']
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
        spatial_dim=2
    )
    if best:
        generator.load_state_dict(checkpoint['g_waypointbest_state'])
    else:
        generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

class NaviGanNode(Node):
    def __init__(self, checkpoint):
        super().__init__('navigan_node')

        # Publishers
        self.waypoint_pub = self.create_publisher(PointStamped, '/way_point', 5)
        self.path_pub = self.create_publisher(Path, '/navigan_path', 1)

        # Subscribers
        self.tracked_pt_sub = self.create_subscription(PointCloud2, '/tracked_points', self.tracked_pt_callback, 2)
        self.goal_sub = self.create_subscription(PointStamped, '/navigan_goal', self.goal_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/a200)0656/amcl_pose', self.odom_callback, 1)
        #self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 1)

        self.obs_rate = 0.4
        self.goal_reached = False
        self.OBS_LEN = 4
        self.mutex = Lock()

        # Variables for state handling
        self.state_queue = deque(maxlen=10000)
        self.goal_init = False
        self.scan_init = False
        self.odom_init = False
        self.predict_goal = True
        self.next_waypoint = Point()

        # Load the intention generator
        if VERBOSE:
            self.get_logger().info('Loading Attention Generator...')
        self.intention_generator = get_attention_generator(checkpoint)
        if VERBOSE:
            self.get_logger().info('Done loading.')

    def people_callback(self, msg):
        if not msg.people:
            self.get_logger().info("No people detected.")
            return

        # Mutex Lock to safely update shared state (people and robot pose data)
        with self.state_lock:
            # Create a dictionary to store people's data (ID as key, position, and velocity)
            people_data = defaultdict(lambda: {'position': None, 'velocity': None})

            for person in msg.people:
                person_id = int(person.name)  # Convert name (string) to ID (int)
                
                # Store person's position and velocity in the people_data dict
                people_data[person_id]['position'] = person.position
                people_data[person_id]['velocity'] = person.velocity

                # Log the person's position for debugging
                self.get_logger().info(f"Person ID: {person_id}, Position: ({person.position.x}, {person.position.y}, {person.position.z})")

            # Store robot's current pose (assuming robot_pose is updated elsewhere)
            robot_position = self.robot_pose.position

            # Append the collected state (people data + robot pose) to the state_queue
            self.state_queue.append({
                'people_data': people_data,  # Store people data (positions and velocities)
                'robot_position': robot_position,  # Store robot's current position
                'timestamp': self.get_clock().now()  # Timestamp of the state
            })

    # def tracked_pt_callback(self, scan_in):
    #     if not self.odom_init or not self.goal_init:
    #         return
        
    #     scan_time = scan_in.header.stamp
    #     if not self.scan_init:
    #         self.scan_init_time = scan_time
    #         self.scan_init = True
    #         self.get_logger().info('Scan initialized')

    #     # Mutex Acquire
    #     with self.mutex:
    #         active_peds_id = set()
    #         peds_pos_t = defaultdict(lambda: None)
    #         min_distance = 9999

    #         for p in pc2.read_points(scan_in, field_names=("x", "y", "z", "h", "s", "v"), skip_nans=True):
    #             active_peds_id.add(p[3])
    #             peds_pos_t[p[3]] = np.array([p[0], p[1]])
    #             dist = math.sqrt((self.odom_data.position.x - p[0])**2 + (self.odom_data.position.y - p[1])**2)
    #             if dist < min_distance:
    #                 min_distance = dist

    #         self.state_queue.append({
    #             'active_peds_id': active_peds_id,
    #             'peds_pos_t': peds_pos_t,
    #             'time_stamp': scan_time,
    #             'robot_pos_t': np.array([self.odom_data.position.x, self.odom_data.position.y]),
    #             'robot_rot_t': self.odom_data.orientation
    #         })

    #     # Prediction logic
    #     if min_distance <= VALIDSCAN:
    #         if scan_time - self.scan_init_time > rclpy.duration.Duration(seconds=5*self.obs_rate):
    #             self.get_logger().info('Predicting...')
    #             self.predict(scan_time)
    #     else:
    #         self.get_logger().info('Using goalpoint')
    #         self.next_waypoint = self.goal_data

    #     tmp_point = PointStamped()
    #     tmp_point.header.frame_id = "map"
    #     tmp_point.header.stamp = scan_time
    #     tmp_point.point = self.next_waypoint
    #     self.waypoint_pub.publish(tmp_point)

    def predict(self, current_time):
        if self.goal_reached:
            return
        # Acquire mutex and collect information
        with self.mutex:
            state = self.state_queue[-1] #most recent state
            active_peds_id = state['active_peds_id']
            peds_pos_t = defaultdict(deque)
            husky_pos_t = deque()
            for id in active_peds_id:
                peds_pos_t[id].append(state['peds_pos_t'][id])
            
            husky_pos_t.append(state['robot_pos_t'])

            # looper = len(self.state_queue) - 1 #load prev step
            # for i in range(1, self.OBS_LEN):
            #     while current_time - self.state_queue[looper]['time_stamp'] > rclpy.duration.Duration(seconds=self.obs_rate*i):
            #         looper -= 1
            #     state = self.state_queue[looper]
            #     for id in active_peds_id:
            #         peds_pos_t[id].appendleft(state['peds_pos_t'][id])
            #     husky_pos_t.appendleft(state['robot_pos_t'])
            # #Mutex Released
            # Precompute the indices for the required observation window
            timestamps = [state['time_stamp'] for state in self.state_queue]
            target_times = [current_time - i * self.obs_rate for i in range(self.OBS_LEN)]

            # Find indices closest to the target_times
            indices = [min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - t)) for t in target_times]

            # Now collect the data using the precomputed indices
            for idx in indices:
                state = self.state_queue[idx]
                for id in active_peds_id:
                    peds_pos_t[id].appendleft(state['peds_pos_t'][id])
                husky_pos_t.appendleft(state['robot_pos_t'])
        
        husky_pos_t = np.array(husky_pos_t)
        peds_arr = []
        for id in active_peds_id:
            temp = np.array(peds_pos_t[id])

            # Exaggerate Peds' distance to robot
            displacement = husky_pos_t[-1,:] - temp[-1,:]

            norm_disp = np.sqrt(displacement[0]**2+displacement[1]**2)

            if norm_disp > 2:
                displacement *= 1.75/norm_disp
            elif norm_disp > 1:
                displacement *= 0.75/norm_disp
            elif norm_disp > 0.5:
                displacement *= 0.40/norm_disp
            else:
                displacement /= 2

            temp += displacement.reshape((1,2))
            displacement = husky_pos_t[-1,:] - temp[-1,:]
            temp = np.pad(temp, ((self.OBS_LEN-len(temp),0), (0,0)), 'edge')
            peds_arr.append(temp)

        try:
            obs_traj = [husky_pos_t]+peds_arr
            obs_traj = np.array(obs_traj).transpose((1,0,2))
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()

        goal = np.array([self.goalData.x, self.goalData.y])
        goal = torch.from_numpy(goal).type(torch.float)

        obs_traj = torch.from_numpy(obs_traj).type(torch.float)
        obs_traj_rel = obs_traj - obs_traj[0,:,:]

        seq_start_end = torch.from_numpy(np.array([[0,obs_traj.shape[1]]]))

        goals_rel = goal - obs_traj[0,0,:]
        goals_rel = goals_rel.repeat(1,obs_traj.shape[1],1)       
        
        # move everything to GPU
        obs_traj = obs_traj.cuda()
        obs_traj_rel = obs_traj_rel.cuda()
        seq_start_end = seq_start_end.cuda()
        goals_rel = goals_rel.cuda()

        pred_traj_fake = self.feedforward(obs_traj, obs_traj_rel, seq_start_end, goals_rel)
        # print(pred_traj_fake.size())
        ptf = pred_traj_fake[:,0,:].cpu().numpy()  #[12,2]

        # pick the 3rd position as robot predicted step
        # self.next_waypoint = Point(ptf[2,0],ptf[2,1], 0)
        self.next_waypoint = Point(ptf[2,0],ptf[2,1], 0)
        #reach the goal check  need to be tuned
        #rospy.loginfo("waypoint is x: {}, y:{}".format(self.next_waypoint.x, self.next_waypoint.y))
        '''
        Implement goToPose simple commander api here

        while navigation is not done:
            do self.reachGoalCheck here (might not work)

        Alternatively, do goal check when at pose
        if true, set goal reached to true
        '''

        if self.reachGoalCheck(self.next_waypoint, self.goalData, 0.3) is not True:
            dist = math.sqrt((self.odom_data.position.x-self.goalData.x)**2 + (self.odom_data.position.y-self.goalData.y)**2)
            length =  math.sqrt((self.next_waypoint.x-self.odom_data.position.x)**2 + (self.next_waypoint.y-self.odom_data.position.y)**2)
            self.next_waypoint.x = dist / length * (self.next_waypoint.x - self.odom_data.position.x) + self.odom_data.position.x
            self.next_waypoint.y = dist / length * (self.next_waypoint.y - self.odom_data.position.y) + self.odom_data.position.y
        else:
            self.goal_reached = True

    def goal_callback(self, goal_in):
        if not self.goal_init:
            self.goal_init = True
            self.get_logger().info('Goal initialized')
        self.goal_data = goal_in.point

    def odom_callback(self, odom_in):
        if not self.odom_init:
            self.odom_init = True
            self.get_logger().info('Odometry initialized')
        self.odom_data = odom_in.pose.pose
    def reachGoalCheck(self, robotPosition, goalPosition, _r=RADIUS):
        if (robotPosition.x-goalPosition.x)**2 + (robotPosition.y-goalPosition.y)**2 < _r**2:
            return True
        else:
            return False
    
    def feedforward(self, obs_traj, obs_traj_rel, seq_start_end, goals_rel):
        """
        obs_traj: torch.Tensor([4, num_agents, 2])
        obs_traj_rel: torch.Tensor([4, num_agents, 2])
        seq_start_end: torch.Tensor([batch_size, 2]) #robot+#pedstrains
        goals_rel: torch.Tensor([1, num_agents, 2])
        """

        with torch.no_grad():
            pred_traj_fake_rel, _ = self.intention_generator(obs_traj, obs_traj_rel, seq_start_end, goal_input=goals_rel)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])
        return pred_traj_fake


def main(args=None):
    rclpy.init(args=args)
    
    CHECKPOINT = '/home/immanuel/Social-Navigation-Simulator/gym_collision_avoidance/envs/policies/NAVIGAN/models/benchmark_zara1_with_model.pt'
    node = NaviGanNode(torch.load(CHECKPOINT))

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

