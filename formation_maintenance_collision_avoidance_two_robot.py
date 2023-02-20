import time

import numpy as np
import random
import cvxopt
from matplotlib import pyplot as plt
from sensor_msgs.msg import LaserScan
from matplotlib import cm
import quadprog
import math
from std_msgs.msg import Header
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import rospy
from geometry_msgs.msg import Twist, Point, TransformStamped,Pose2D
#from nav_msgs.msg import Path
from math import pow, atan2, sqrt,atan
from tf2_msgs.msg import TFMessage
#y_i: scalar target
import threading

"""xx_array = np.random.uniform(-10.,10.,1000)
xy_array = np.random.uniform(-10.,10.,1000)
xx_array_test = np.random.uniform(-20.,20.,2000)
xy_array_test = np.random.uniform(-20.,20.,2000)"""
y_array = []
y_array_2 = []
sigma_f = 1
sigma_y = 0.01
r = 10.1
l = 0.06
lidar_range = 4
max_num_of_pts =1000
min_distance = 0.003
ignore_area_size = 15
#querry_point = np.array([1,2])
data_point_array = []
data_point_array_2 = []
speed_limit = 0.075
multiplier=1
delta_y= 0
delta_h_gp=0.25
epsilon = 0.05
robot_distance = 0.5

L = np.diag([0.1, 0.1])
Lminus2 = np.linalg.inv(np.linalg.inv(L))
K_matrix =[]
#calculate kernel (smallest_k)
def K_matrix_inv_cal_init():
    global K_matrix
    K_matrix = [[1.01]]
    K_matrix_numpy = np.array(K_matrix)
    K_matrix_inv = np.linalg.inv(K_matrix_numpy)
    return K_matrix_inv

def K_matrix_inv_cal():
    global data_point_array, K_matrix
    last_point_added = data_point_array[-1]
    new_k_array = []
    for i in range(len(data_point_array)-1):

        diff = data_point_array[i]-last_point_added
        added_k_regard_to_new_point = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff))/2)
        K_matrix[i].append(added_k_regard_to_new_point)
        if len(K_matrix[i])>max_num_of_pts:
            K_matrix[i].pop(0)
        
    for j in range(len(data_point_array)):
        diff =last_point_added- data_point_array[j]
        k_new_point = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff)) / 2)+int((len(data_point_array)-1)==j)*sigma_y
        new_k_array.append(k_new_point)
    K_matrix.append(new_k_array)
    if len(K_matrix)>max_num_of_pts:
        K_matrix.pop(0)
    
    K_matrix_numpy = np.array(K_matrix)
    #prev_cov = 0
    K_matrix_inv = np.linalg.inv(K_matrix_numpy)
    return K_matrix_inv


def K_matrix_inv_cal_init_2():
    global K_matrix_2
    K_matrix_2 = [[1.01]]
    K_matrix_2_numpy = np.array(K_matrix)
    K_matrix_2_inv = np.linalg.inv(K_matrix_2_numpy)
    return K_matrix_2_inv

def K_matrix_inv_cal_2():
    global data_point_array_2, K_matrix_2
    last_point_added = data_point_array_2[-1]
    new_k_array = []
    for i in range(len(data_point_array_2)-1):

        diff = data_point_array_2[i]-last_point_added
        added_k_regard_to_new_point = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff))/2)
        K_matrix_2[i].append(added_k_regard_to_new_point)
        if len(K_matrix_2[i])>max_num_of_pts:
            K_matrix_2[i].pop(0)
        
    for j in range(len(data_point_array_2)):
        diff =last_point_added- data_point_array_2[j]
        k_new_point = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff)) / 2)+int((len(data_point_array_2)-1)==j)*sigma_y
        new_k_array.append(k_new_point)
    K_matrix_2.append(new_k_array)
    if len(K_matrix_2)>max_num_of_pts:
        K_matrix_2.pop(0)
    
    K_matrix_numpy_2 = np.array(K_matrix_2)
    #prev_cov = 0
    K_matrix_inv_2 = np.linalg.inv(K_matrix_numpy_2)
    return K_matrix_inv_2
#u_array = []
hgp_array = []
mu_sig_array =[]
mu_array = []
sig_array =[]
collected_point_x = []
collected_point_y = []
#goal_list = np.array([[6,1],[-10,-8],[-1,1],[-10,0]])
current_goal = np.array([-2.,0.])
#robot_angle = np.pi/6
goal_idx = 0
x_querry_test = np.array([[0,0]])
collected_data = []


hgp_array_2 = []
mu_sig_array_2 =[]
mu_array_2 = []
sig_array_2 =[]
collected_point_x_2 = []
collected_point_y_2 = []
#goal_list = np.array([[6,1],[-10,-8],[-1,1],[-10,0]])
current_goal_2 = np.array([-2.,0.])
#robot_angle = np.pi/6
goal_idx_2 = 0
x_querry_test_2 = np.array([[0,0]])
collected_data_2 = []

def quadprog_solve_qp(P, q, G, h, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
#initiate, get first point and begin calculation
initiated_array = False
initiated_loc = False
initiated_y = False


initiated_array_2 = False
initiated_loc_2 = False
initiated_y_2 = False

current_loc = np.array([0.,0.])
h_gp_array_2 = []
step_array =[]
c = 0
yaw = 0
latest_y = 0
y_array_all = []
robot_distance_array = []

def update_scan(data):
    global y_array,y_array_output, latest_y, initiated_y
    distance_array = data.ranges
    distance_array_list_pre = list(distance_array)
    idx = round((-yaw+math.atan2(current_loc_2[1]-current_loc[1],current_loc_2[0]-current_loc[0]))/math.pi*180)
    #print("yaw",yaw)
    #print("vector angle",math.atan2(current_loc[1]-current_loc_2[1],current_loc[0]-current_loc_2[0]))
    #print("facing ",idx)
    idx = idx%360
    print("idx1",idx)
    start_idx = idx - ignore_area_size
    end_idx = idx + ignore_area_size
    if end_idx == 0:
        del distance_array_list_pre[start_idx:]
        del distance_array_list_pre[0]
    elif end_idx* start_idx >=0:
        del distance_array_list_pre[start_idx:end_idx+1]
    else:
        del distance_array_list_pre[start_idx:]
        del distance_array_list_pre[0:end_idx]

    distance_array_list = []
    distance_array_list = [x for x in distance_array_list_pre if x > 0]
    min_distance = min(distance_array_list)
    latest_y = min(min_distance,lidar_range)-delta_y
    print("y1 ",latest_y)
    #y_array.append(min_distance)
    #y_array_output = np.array(y_array)

    initiated_y = True

def update_tf(data): #Callback function which 0.25is called when a new message of type Pose is received by the subscriber.
    global roll, pitch, yaw
    global pose_tf
    global latest_y,initiated_y,latest_y,c,current_loc,initiated_loc

    pose_tf = data
    x_coord = pose_tf.x
    y_coord = pose_tf.y
    current_loc = np.array([float(x_coord),float(y_coord)])
#    print('pose_tf: ',pose_tf)
    yaw = pose_tf.theta
    initiated_loc = True


#function for 2nd robot (yes it is repetitive but only a temporary until I rewrite all to class)
current_loc_2 = np.array([0.,0.])
h_gp_array_2_2 = []
step_array_2 =[]
c_2 = 0
yaw_2 = 0
latest_y_2 = 0
y_array_all_2 = []
goal_reach = False
def update_scan_2(data):
    global y_array_2,y_array_output_2, latest_y_2, initiated_y_2,yaw_2, current_loc,current_loc_2
    distance_array = data.ranges
    distance_array_list_pre = list(distance_array)
    vector_angle = math.atan2(current_loc[1]-current_loc_2[1],current_loc[0]-current_loc_2[0])
    print("vector ", vector_angle)
    idx = round((-yaw_2+vector_angle)/math.pi*180)
    idx = idx%360
    start_idx = idx - ignore_area_size
    end_idx = idx + ignore_area_size
    print(idx)
    if end_idx == 0:
        del distance_array_list_pre[start_idx:]
        del distance_array_list_pre[0]
    elif end_idx* start_idx >=0:
        del distance_array_list_pre[start_idx:end_idx+1]
    else:
        del distance_array_list_pre[start_idx:]
        del distance_array_list_pre[0:end_idx]

    distance_array_list = []
    distance_array_list = [x for x in distance_array_list_pre if x > 0]
    min_distance = min(distance_array_list)
    latest_y_2 = min(min_distance,lidar_range)-delta_y
    print("y2 ",latest_y_2)
    #y_array.append(min_distance)
    #y_array_output = np.array(y_array)

    initiated_y_2 = True


def update_tf_2(data): #Callback function which is called when a new message of type Pose is received by the subscriber.
    global roll, pitch, yaw_2
    global pose_tf
    global current_loc_2,initiated_loc_2

    pose_tf = data
    x_coord = pose_tf.x
    y_coord = pose_tf.y
    current_loc_2 = np.array([float(x_coord),float(y_coord)])
#    print('pose_tf: ',pose_tf)
    yaw_2 = pose_tf.theta
    initiated_loc_2 = True

distance_between_robot = 0.5
def move_robot():
    global yaw, goal_list,goal_idx, K_matrix_inv, y_array_output,data_point_array, velocity_publisher,initiated_array ,latest_y,initiated_y,c,current_loc,step_array,y_array, Lminus2,collected_point_x,collected_point_y
    vel_msg = Twist()
    global yaw_2, goal_list_2,goal_idx_2, K_matrix_inv_2, y_array_output_2,data_point_array_2, velocity_publisher_2,initiated_array_2 ,latest_y_2,initiated_y_2,c_2,current_loc_2,step_array_2,y_array_2, Lminus2,collected_point_x_2,collected_point_y_2
    global speed_limit,ignore_area_size,distance_between_robot,epsilon
    n =1
    goal1_reached = False
    goal2_reached = False
    robot_distance = np.linalg.norm(current_loc-current_loc_2)
    current_goal[1]+=robot_distance
    while not rospy.is_shutdown() :
        distance_to_goal = np.linalg.norm(current_goal-current_loc)
        distance_to_goal_2 = np.linalg.norm(current_goal_2-current_loc_2)
        distance_between_robot = np.linalg.norm(current_loc-current_loc_2)
        ignore_area_size = round(atan(0.04/distance_between_robot))+1
        if distance_to_goal_2<=0.1:
            goal2_reached = True
        if distance_to_goal<=0.1:
            goal1_reached = True
        if goal1_reached and goal2_reached:
            break
        c+=1
        try:
            c = 0

            if initiated_array == False:
                data_point_array.append(current_loc)
                K_matrix_inv = K_matrix_inv_cal_init()
                y_array.append(latest_y)
                y_array_output = np.array(y_array)
                
                u_initial = current_goal - current_loc
                l_matrix = np.matmul(np.array([[1, 0], [0, 1 / l]]), np.array(
                        [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]))
                u_rec_ahead = np.matmul(l_matrix, u_initial.T)

                vel_msg.linear.x = u_rec_ahead[0]*0.5
                vel_msg.linear.y = 0
                vel_msg.angular.z = u_rec_ahead[1]
                velocity_publisher.publish(vel_msg)
                # update distance
                distance_vec = current_goal - current_loc
                distance_len = np.linalg.norm(distance_vec)
                
                initiated_array = True

            elif not goal1_reached:
                k_querry_array = list()
                # print("data_point_arr", data_point_array)
                # print("current_loc", current_loc)
                #print(len(data_point_array))
                delta_kernel_array_sim = []

                for data_point in data_point_array:
                    diff = current_loc - data_point
                    k = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff)) / 2)
                    k_querry_array.append(k)
                    kernel = np.matmul(-diff * k, Lminus2)
                    delta_kernel_array_sim.append(kernel)

                k_querry_array = np.array(k_querry_array)
                delta_kernel_array_sim = np.array(delta_kernel_array_sim)

                # average mu
                # print("datapointarra: ", data_point_array)
                # print("k_qerry: ",k_querry_array)

                # print("y_array: ",y_array_output)
                #print(current_goal)
                mu = np.matmul(np.matmul(k_querry_array, K_matrix_inv), y_array_output)

                # covariance
                sig_querry = 1 - np.matmul(np.matmul(k_querry_array, K_matrix_inv), np.transpose(k_querry_array))
                # print(x_querry, "mu: ", mu, ";sigma: ", sig_querry)
                # _looks like h_gp is wrong a bit, it does not fall under 0.060 even when collided
                h_gp = mu - sig_querry -delta_h_gp
                # u_array.append(goal-x_querry)
                # calculate delta_hgp_over_x_regard_to_x_querry

                """for data_point in data_point_array:
                    diff = data_point - current_loc
                    k = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff)) / 2)
                    kernel = np.matmul(diff * k, Lminus2)
                    delta_kernel_array_sim.append(kernel)
                delta_kernel_array_sim = np.array(delta_kernel_array_sim)"""
                try:
                # delta_hgp_over_x_regard_to_x_querry is d_hgp/d_x regarding x_query
                    delta_hgp_over_x_regard_to_x_querry = np.matmul(np.matmul(y_array, K_matrix_inv),delta_kernel_array_sim) + 2 * np.matmul(np.matmul(k_querry_array, K_matrix_inv), delta_kernel_array_sim)
                    delta_hgp_over_x_regard_to_x_querry = np.array(delta_hgp_over_x_regard_to_x_querry)
                    u_nom_xy = current_goal - current_loc
                    #print("u_nom",u_nom_xy)
                    #print("yaw", yaw)
                    norm = np.hypot(u_nom_xy[0], u_nom_xy[1])
                    if norm > speed_limit: u_nom_xy = speed_limit* u_nom_xy / norm
                    M = np.eye(2)
                    P = np.dot(M.T, M)
                    q = np.dot(M.T, -u_nom_xy)
                    # print(q)
                    # transform delta_hgp_over_x_regard_to_x_querry to good form to suit the calculation
                    #delta_hgp_over_x_regard_to_x_querry_good_form = np.array([delta_hgp_over_x_regard_to_x_querry])
                    # print(delta_hgp_over_x_regard_to_x_querry_good_form)
                    #G = -delta_hgp_over_x_regard_to_x_querry_good_form*6
                    #h = np.array([h_gp])
                    h_fmu = -(distance_between_robot**2) + (robot_distance+epsilon)**2
                    h_fml = (distance_between_robot**2) - (robot_distance-epsilon)**2
                    d_h_fml = [current_loc[0]-current_loc_2[0],current_loc[1]-current_loc_2[1]]
                    d_h_fmu = [current_loc_2[0]-current_loc[0],current_loc_2[1]-current_loc[1]]
                    delta_hgp_over_x_regard_to_x_querry_good_form = np.array([delta_hgp_over_x_regard_to_x_querry,d_h_fml,d_h_fmu])
                    # print(delta_hgp_over_x_regard_to_x_querry_good_form)
                    G = -delta_hgp_over_x_regard_to_x_querry_good_form*multiplier
                    h = np.array([h_gp,h_fml,h_fmu])
                    h_gp_array_2.append(h_gp)
                    step_array.append(n)
                    n+=1

                    # h = np.matmul(np.array([np.linalg.norm(delta_hgp_over_x_regard_to_x_querry),h_gp]), np.array([np.linalg.norm(delta_hgp_over_x_regard_to_x_querry),h_gp]) )
                    # print(h)

                    sol = quadprog_solve_qp(P, q, G, h)
                    u_rec_dx_dy = np.array(sol)


                    # set speed limit
                    norm = np.hypot(u_rec_dx_dy[0], u_rec_dx_dy[1])
                    if norm > speed_limit: u_rec_dx_dy = speed_limit* u_rec_dx_dy / norm # max 


                    # append the current point before moving to data_array, calculate K and y_array
                    collected_point_x.append(current_loc[0])
                    collected_point_y.append(current_loc[1])
                    if np.sqrt((current_loc[0]-data_point_array[-1][0])**2+(current_loc[1]-data_point_array[-1][1])**2) > min_distance:
                        data_point_array.append(np.array([float(current_loc[0]), float(current_loc[1])]))

                        if len(data_point_array) > max_num_of_pts:
                            data_point_array.pop(0)
                        K_matrix_inv = K_matrix_inv_cal()
                        
                        y_array.append(latest_y)
                        y_array_all.append(latest_y)

                        if len(y_array) > max_num_of_pts:
                            y_array.pop(0)
                        y_array_output = np.array(y_array)
                    

                    l_matrix = np.matmul(np.array([[1, 0], [0, 1 / l]]), np.array(
                        [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]))
                    # turn u_rec_dx_dy into u_rec_ahead (velocity)
                    u_rec_ahead = np.matmul(l_matrix, u_rec_dx_dy.T)
                    #print("urec ahead raw", u_rec_ahead)
                    # print(sol)
                    """if u_rec_ahead[0] >= 0.1:
                        u_rec_ahead[0] = 0.1
                  
                    elif u_rec_ahead[0] <= -0.1:
                        u_rec_ahead[0] = -0.1
                    
                    
                    if u_rec_ahead[1] >= np.pi/2:
                        u_rec_ahead[1] = np.pi/2
                    elif u_rec_ahead[1] <= -np.pi/2:
                        u_rec_ahead[1] = -np.pi/2"""

                    vel_msg.linear.x = u_rec_ahead[0] #* 0.1
                    vel_msg.linear.y = 0
                    vel_msg.angular.z = u_rec_ahead[1]

                    velocity_publisher.publish(vel_msg)

                    #print("u_rec: ", u_rec_ahead, " h_gp: ", h_gp, "mu: ", mu, "cov", sig_querry, "d_hgp/d_x: ",delta_hgp_over_x_regard_to_x_querry_good_form, "distance to nearest obs: ",latest_y)

                    # update distance
                    distance_vec = current_goal - current_loc
                    distance_len = np.linalg.norm(distance_vec)
                    if distance_len <=0.1:
                        goal_idx+=1
                        print("reach goal: ", current_goal)
                    # print22(distance_len)
                    # print("robot pose",current_loc, " ", robot_angle)
                    
                except ValueError:
                    pass
                    # velocity_publisher.publish(Twist())
            else:

                velocity_publisher.publish(Twist())
                
        except Exception:
            pass
        
        


        c+=1
        vel_msg = Twist()
        try:
            c = 0

            if initiated_array_2 == False:
                data_point_array_2.append(current_loc_2)
                K_matrix_inv_2 = K_matrix_inv_cal_init_2()
                y_array_2.append(latest_y_2)
                y_array_output_2 = np.array(y_array_2)
                
                u_initial = current_goal_2 - current_loc_2
                l_matrix = np.matmul(np.array([[1, 0], [0, 1 / l]]), np.array([[np.cos(yaw_2), np.sin(yaw_2)], [-np.sin(yaw_2), np.cos(yaw_2)]]))
                u_rec_ahead = np.matmul(l_matrix, u_initial.T)

                vel_msg.linear.x = u_rec_ahead[0]*0.5
                vel_msg.linear.y = 0
                vel_msg.angular.z = u_rec_ahead[1]
                velocity_publisher_2.publish(vel_msg)
                # update distance
                distance_vec = current_goal_2 - current_loc_2
                distance_len = np.linalg.norm(distance_vec)
                
                initiated_array_2 = True

            elif not goal2_reached:
                k_querry_array = list()
                # print("data_point_arr", data_point_array)
                # print("current_loc", current_loc)
                #print(len(data_point_array))
                delta_kernel_array_sim = []

                for data_point in data_point_array_2:
                    diff = current_loc_2 - data_point
                    k = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff)) / 2)
                    k_querry_array.append(k)
                    kernel = np.matmul(-diff * k, Lminus2)
                    delta_kernel_array_sim.append(kernel)

                k_querry_array = np.array(k_querry_array)
                delta_kernel_array_sim = np.array(delta_kernel_array_sim)

                # average mu
                # print("datapointarra: ", data_point_array)
                # print("k_qerry: ",k_querry_array)

                # print("y_array: ",y_array_output)
                #print(current_goal)
                mu = np.matmul(np.matmul(k_querry_array, K_matrix_inv_2), y_array_output_2)

                # covariance
                sig_querry = 1 - np.matmul(np.matmul(k_querry_array, K_matrix_inv_2), np.transpose(k_querry_array))
                # print(x_querry, "mu: ", mu, ";sigma: ", sig_querry)
                # _looks like h_gp is wrong a bit, it does not fall under 0.060 even when collided
                h_gp = mu - sig_querry - delta_h_gp
                # u_array.append(goal-x_querry)speed_limit
                # calculate delta_hgp_over_x_regard_to_x_querry

                """for data_point in data_point_array:
                    diff = data_point - current_loc
                    k = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff)) / 2)
                    kernel = np.matmul(diff * k, Lminus2)
                    delta_kernel_array_sim.append(kernel)
                delta_kernel_array_sim = np.array(delta_kernel_array_sim)"""
                try:
                # delta_hgp_over_x_regard_to_x_querry is d_hgp/d_x regarding x_query
                    delta_hgp_over_x_regard_to_x_querry = np.matmul(np.matmul(y_array_2, K_matrix_inv_2),delta_kernel_array_sim) + 2 * np.matmul(np.matmul(k_querry_array, K_matrix_inv_2), delta_kernel_array_sim)
                    delta_hgp_over_x_regard_to_x_querry = np.array(delta_hgp_over_x_regard_to_x_querry)
                    u_nom_xy = current_goal_2 - current_loc_2
                    #print("u_nom",u_nom_xy)
                    #print("yaw", yaw)
                    norm = np.hypot(u_nom_xy[0], u_nom_xy[1])
                    if norm > speed_limit: u_nom_xy = speed_limit* u_nom_xy / norm
                    M = np.eye(2)
                    P = np.dot(M.T, M)
                    q = np.dot(M.T, -u_nom_xy)
                    # print(q)
                    
                    h_fmu = -(distance_between_robot**2) + (robot_distance+epsilon)**2
                    h_fml = (distance_between_robot**2) - (robot_distance-epsilon)**2
                    d_h_fml = [current_loc_2[0]-current_loc[0],current_loc_2[1]-current_loc[1]]
                    d_h_fmu = [current_loc[0]-current_loc_2[0],current_loc[1]-current_loc_2[1]]
                    delta_hgp_over_x_regard_to_x_querry_good_form = np.array([delta_hgp_over_x_regard_to_x_querry,d_h_fml,d_h_fmu])
                    # print(delta_hgp_over_x_regard_to_x_querry_good_form)
                    robot_distance_array.append(np.sqrt((current_loc[1]-current_loc_2[1])**2+(current_loc[0]-current_loc_2[0])**2))
                    G = -delta_hgp_over_x_regard_to_x_querry_good_form*multiplier
                    h = np.array([h_gp,h_fml,h_fmu])
                    h_gp_array_2.append(h_gp)
                    step_array_2.append(n)
                    n+=1

                    # h = np.matmul(np.array([np.linalg.norm(delta_hgp_over_x_regard_to_x_querry),h_gp]), np.array([np.linalg.norm(delta_hgp_over_x_regard_to_x_querry),h_gp]) )
                    # print(h)

                    sol = quadprog_solve_qp(P, q, G, h)
                    u_rec_dx_dy = np.array(sol)


                    # set speed limit
                    speed_limit = 0.1
                    norm = np.hypot(u_rec_dx_dy[0], u_rec_dx_dy[1])
                    if norm > speed_limit: u_rec_dx_dy = speed_limit* u_rec_dx_dy / norm # max 


                    # append the current point before moving to data_array, calculate K and y_array
                    collected_point_x_2.append(current_loc_2[0])
                    collected_point_y_2.append(current_loc_2[1])
                    if np.sqrt((current_loc_2[0]-data_point_array_2[-1][0])**2+(current_loc_2[1]-data_point_array_2[-1][1])**2) > min_distance:
                        data_point_array_2.append(np.array([float(current_loc_2[0]), float(current_loc_2[1])]))

                        if len(data_point_array_2) > max_num_of_pts:
                            data_point_array_2.pop(0)
                        K_matrix_inv_2 = K_matrix_inv_cal_2()
                        
                        y_array_2.append(latest_y_2)
                        y_array_all_2.append(latest_y_2)

                        if len(y_array_2) > max_num_of_pts:
                            y_array_2.pop(0)
                        y_array_output_2 = np.array(y_array_2)
                    

                    l_matrix = np.matmul(np.array([[1, 0], [0, 1 / l]]), np.array(
                        [[np.cos(yaw_2), np.sin(yaw_2)], [-np.sin(yaw_2), np.cos(yaw_2)]]))
                    # turn u_rec_dx_dy into u_rec_ahead (velocity)
                    u_rec_ahead = np.matmul(l_matrix, u_rec_dx_dy.T)
                    #print("urec ahead raw", u_rec_ahead)
                    # print(sol)
                    """if u_rec_ahead[0] >= 0.1:
                        u_rec_ahead[0] = 0.1
                  
                    elif u_rec_ahead[0] <= -0.1:
                        u_rec_ahead[0] = -0.1
                    
                    
                    if u_rec_ahead[1] >= np.pi/2:
                        u_rec_ahead[1] = np.pi/2
                    elif u_rec_ahead[1] <= -np.pi/2:
                        u_rec_ahead[1] = -np.pi/2"""

                    vel_msg.linear.x = u_rec_ahead[0] #* 0.1
                    vel_msg.linear.y = 0
                    vel_msg.angular.z = u_rec_ahead[1]

                    velocity_publisher_2.publish(vel_msg)

                    #print("u_rec: ", u_rec_ahead, " h_gp: ", h_gp, "mu: ", mu, "cov", sig_querry, "d_hgp/d_x: ",delta_hgp_over_x_regard_to_x_querry_good_form, "distance to nearest obs: ",latest_y)

                    # update distance
                    distance_vec = current_goal_2 - current_loc_2
                    distance_len = np.linalg.norm(distance_vec)
                    if distance_len <=0.1:
                        goal_idx_2+=1
                        print("reach goal: ", current_goal_2)
                    # print22(distance_len)
                    # print("robot pose",current_loc, " ", robot_angle)
                    
                except ValueError as err1:
                    print(err1)
                    pass
                    # velocity_publisher.publish(Twist())
            else:

                velocity_publisher_2.publish(Twist())
                
        except Exception as err2:
            print(err2)
            pass
    



#for getting u rec for each test point

#x_array_data_collected.append(0.)
#y_array_data_collected.append(0.)
#current_loc = np.array([0.,0.])
#data_point_array.append(current_loc)

# trial and error: work more likely than you think!

        #print(u_nom)


rospy.init_node('turtlebot3_controller', anonymous=True) # Creates a node with name 'turtlebot_controller' and make sure it is a unique node name (using anonymous=True).
velocity_publisher = rospy.Publisher('/tb3_2/cmd_vel', Twist, queue_size=10) # Publisher which will publish to the topic '/turtle1/cmd_vel'.
tf_subscriber=rospy.Subscriber('/tb3_2/pos', Pose2D, update_tf)
scan_subscriber = rospy.Subscriber('/tb3_2/scan', LaserScan, update_scan)

velocity_publisher_2 = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size=10) # Publisher which will publish to the topic '/turtle1/cmd_vel'.
tf_subscriber_2nd_rob=rospy.Subscriber('/tb3_0/pos', Pose2D, update_tf_2)
scan_subscriber_2nd_rob = rospy.Subscriber('/tb3_0/scan', LaserScan, update_scan_2)

pose_tf=TransformStamped()

rate = rospy.Rate(200)
rospy.sleep(1)
#tst()
move_robot()
x_test_point = np.random.uniform(-4., 4., 2000)
y_test_point = np.random.uniform(-4., 4., 2000)
point_array_testing = []
h_gp_test_array = []
h_gp_test_array_2 = []

print('total number of data point collected ',len(data_point_array))
for i in range(len(x_test_point)):
    #point_array_testing.append([x_test_point[i],y_test_point[i]])
    k_querry_array_test = []
    for point in data_point_array:
        diff = [x_test_point[i],y_test_point[i]] - point
        k = np.exp(-np.matmul(np.matmul(diff,Lminus2), np.transpose(diff))/2)
        k_querry_array_test.append(k)

    k_querry_array_test = np.array(k_querry_array_test)
    mu = np.matmul(np.matmul(k_querry_array_test,K_matrix_inv),y_array_output)
    sig_querry_test = 1- np.matmul(np.matmul(k_querry_array_test,K_matrix_inv), np.transpose(k_querry_array_test))
    h_gp_test = mu - sig_querry_test
    h_gp_test_array .append(h_gp_test)

for i in range(len(x_test_point)):
    #point_array_testing.append([x_test_point[i],y_test_point[i]])
    k_querry_array_test = []
    for point in data_point_array_2:
        diff = [x_test_point[i],y_test_point[i]] - point
        k = np.exp(-np.matmul(np.matmul(diff,Lminus2), np.transpose(diff))/2)
        k_querry_array_test.append(k)

    k_querry_array_test = np.array(k_querry_array_test)
    mu = np.matmul(np.matmul(k_querry_array_test,K_matrix_inv_2),y_array_output_2)
    sig_querry_test = 1- np.matmul(np.matmul(k_querry_array_test,K_matrix_inv_2), np.transpose(k_querry_array_test))
    h_gp_test = mu - sig_querry_test
    h_gp_test_array_2.append(h_gp_test)

#plt.figure('safety belief along robot path')
#plt.plot(step_array,h_gp_array_2)
plt.figure('safety map based on knowledge at final goal')
plt.plot(collected_point_x,collected_point_y,marker=".", markersize=1, markeredgecolor="black", label= "path")
plt.plot(-2,0,marker="x", markersize=10, markeredgecolor="white", label= "goal1")
plt.plot(-2,current_goal[1],marker="x", markersize=10, markeredgecolor="white", label= "goal2")
cset = plt.tricontourf(x_test_point,y_test_point, h_gp_test_array)
plt.colorbar(cset)
plt.figure('safety map based on knowledge at final goal robot 2')
plt.plot(collected_point_x_2,collected_point_y_2,marker=".", markersize=1, markeredgecolor="black", label= "path")
plt.plot(-2,0,marker="x", markersize=10, markeredgecolor="white", label= "goal1")
plt.plot(-2,current_goal[1],marker="x", markersize=10, markeredgecolor="white", label= "goal2")
cset2 = plt.tricontourf(x_test_point,y_test_point, h_gp_test_array_2)
plt.colorbar(cset2)
plt.figure('robot distance by step')
plt.plot(robot_distance_array)
plt.axhline(y = distance_between_robot+epsilon, color = 'r',  linestyle = '-')
plt.axhline(y = distance_between_robot-epsilon, color = 'r',  linestyle = '-')
plt.show()

#rospy.spin()
#hgp_array = np.array(hgp_array)
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
def shutdown():
    # stop turtlebot
    global velocity_publisher
    rospy.loginfo("Stop TurtleBot")
# a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
    velocity_publisher.publish(Twist())
# sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
    rospy.sleep(1)
rospy.on_shutdown(shutdown())
#oh no codes for the plots

"""f1 = plt.figure(1)
axs1 = f1.add_subplot(221)
axs2 = f1.add_subplot(222)
axs3 = f1.add_subplot(223)

# note to self: read widhi mail about improving the graph
obs1=plt.Circle((3,1),r,color="white", label= "obstacle")
obs2=plt.Circle((-6,-4),r,color="white", label= "obstacle")
obs3=plt.Circle(obstacle[2],r,color="white", label= "obstacle")
obs4=plt.Circle(obstacle[3],r,color="white", label= "obstacle")
#cset = axs1.tricontourf(xx_array_test,xy_array_test, hgp_array)
axs1.add_patch(obs1)
axs1.add_patch(obs2)
axs1.add_patch(obs3)
axs1.add_patch(obs4)

axs1.legend()
plt.tight_layout()

axs1.set_title("overall safety hgp")
#cset2 = axs2.tricontourf(xx_array_test,xy_array_test, mu_array)
obs12=plt.Circle((3,1),r,color="blue", label= "obstacle")
obs22=plt.Circle((-6,-4),r,color="blue")
obs32=plt.Circle(obstacle[2],r,color="blue")
obs42=plt.Circle(obstacle[3],r,color="blue")
axs2.add_patch(obs32)
axs2.add_patch(obs42)
axs2.add_patch(obs12)
axs2.add_patch(obs22)
axs2.set_title("safety belief mu")
axs2.legend()
plt.tight_layout()
#cset3 = axs3.tricontourf(xx_array_test,xy_array_test, sig_array)
obs13=plt.Circle((3,1),r,color="blue", label= "obstacle")
obs23=plt.Circle((-6,-4),r,color="blue")
axs3.add_patch(obs13)
axs3.add_patch(obs23)
obs33=plt.Circle(obstacle[2],r,color="blue")
obs43=plt.Circle(obstacle[3],r,color="blue")
axs3.add_patch(obs33)
axs3.add_patch(obs43)

axs3.set_title("safety uncertainty sigma_squared")
#plt.colorbar(cset)
#plt.colorbar(cset2)
#plt.colorbar(cset3)
axs3.legend()
plt.tight_layout()
f2 = plt.figure(2)
axs4 = f2.add_subplot(111)
#csetf2 = axs4.tricontourf(xx_array_test,xy_array_test, hgp_array)
obs14=plt.Circle((3,1),r,color="blue", label= "obstacle")
obs24=plt.Circle((-6,-4),r,color="blue")
axs4.add_patch(obs14)
axs4.add_patch(obs24)
obs34=plt.Circle(obstacle[2],r,color="blue")
obs44=plt.Circle(obstacle[3],r,color="blue")
axs4.add_patch(obs34)
axs4.add_patch(obs44)
axs4.plot(6,1,marker="x", markersize=10, markeredgecolor="red", label= "goal1")
axs4.plot(-10,-8,marker="x", markersize=10, markeredgecolor="green", label= "goal2")
axs4.plot(-1,1,marker="x", markersize=10, markeredgecolor="orange", label= "goal3")

axs4.plot(-10,0,marker="x", markersize=10, markeredgecolor="black", label= "goal4")
axs4.plot(0,0,marker="o", markersize=5, markeredgecolor="red",markerfacecolor="blue" , label= "starting_point")
axs4.plot(x_array_data_collected,y_array_data_collected,marker=".", markersize=1, markeredgecolor="black",markerfacecolor="black",label= "robot_path")
axs4.set_title("robot path and safety belief area map")
#plt.colorbar(csetf2)
plt.tight_layout()
plt.legend()

f3 = plt.figure(3)
axs5 = f3.add_subplot(111)
axs5.plot(h_gp_array_2)
plt.show()
"""

