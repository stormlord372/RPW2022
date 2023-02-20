import time

import numpy as np
import random
import cvxopt
from matplotlib import pyplot as plt
from sensor_msgs.msg import LaserScan
from matplotlib import cm
import quadprog

from std_msgs.msg import Header
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import rospy
from geometry_msgs.msg import Twist, Point, TransformStamped
#from nav_msgs.msg import Path
from math import pow, atan2, sqrt
from tf2_msgs.msg import TFMessage
#y_i: scalar target


"""xx_array = np.random.uniform(-10.,10.,1000)
xy_array = np.random.uniform(-10.,10.,1000)
xx_array_test = np.random.uniform(-20.,20.,2000)
xy_array_test = np.random.uniform(-20.,20.,2000)"""
y_array = []
sigma_f = 1
sigma_y = 0.01
r = 1
l = 0.06
lidar_range = 4
max_num_of_pts =100
#querry_point = np.array([1,2])
data_point_array = []




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
        """k_value_new_point_regard_to_existing_point = sigma_f*np.exp(-np.matmul(np.matmul(-diff, Lminus2), np.transpose(diff))/2)
        new_k_array.append(k_value_new_point_regard_to_existing_point)
    new_k_array.append(sigma_f + sigma_y)"""
    for j in range(len(data_point_array)):
        diff =last_point_added- data_point_array[j]
        k_new_point = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff)) / 2)+int((len(data_point_array)-1)==j)*sigma_y
        new_k_array.append(k_new_point)
    K_matrix.append(new_k_array)
    if len(K_matrix)>max_num_of_pts:
        K_matrix.pop(0)
    """K_matrix = []
    # calculate kernel (smallest_k)
    for i in range(len(data_point_array)):
        medium_k = []
        for j in range(len(data_point_array)):
            diff = data_point_array[i] - data_point_array[j]

            smallest_k = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff)) / 2) + int(i == j) * sigma_y
            medium_k.append(smallest_k)
        K_matrix.append(medium_k)
    K_matrix = np.array(K_matrix)
    #print(np.array(K_matrix))"""
    K_matrix_numpy = np.array(K_matrix)
    #prev_cov = 0
    K_matrix_inv = np.linalg.inv(K_matrix_numpy)
    return K_matrix_inv

#u_array = []
hgp_array = []
mu_sig_array =[]
mu_array = []
sig_array =[]
#goal_list = np.array([[6,1],[-10,-8],[-1,1],[-10,0]])
goal_list = np.array([[-7.,0.]])
#robot_angle = np.pi/6
goal_idx = 0
x_querry_test = np.array([[0,0]])
collected_data = []

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
def update_scan(data):
    global y_array,y_array_output, latest_y, initiated_y
    distance_array = data.ranges
    min_distance = min(distance_array)
    latest_y = min(min_distance,lidar_range)
    #y_array.append(min_distance)
    #y_array_output = np.array(y_array)

    initiated_y = True

x_array_data_collected = []
y_array_data_collected = []
current_loc = np.array([0.,0.])
h_gp_array_2 = []
c = 0
yaw = 0
latest_y = 0
import math


def euler_from_quaternion_fnc(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians
def update_tf(data): #Callback function which is called when a new message of type Pose is received by the subscriber.
    global roll, pitch, yaw
    global pose_tf
    global goal_list,goal_idx, K_matrix_inv, y_array_output,data_point_array, velocity_publisher,initiated_array ,latest_y,initiated_y,latest_y,c,current_loc,initiated_loc



    pose_tf = data.transforms[0]

    x_coord = round(pose_tf.transform.translation.x, 4)
    y_coord = round(pose_tf.transform.translation.y, 4)
    current_loc = np.array([float(x_coord),float(y_coord)])
#    print('pose_tf: ',pose_tf)
    orientation_list = [pose_tf.transform.rotation.x, pose_tf.transform.rotation.y, pose_tf.transform.rotation.z, pose_tf.transform.rotation.w]
    #print("rotation data",pose_tf.transform.rotation.x, pose_tf.transform.rotation.y, pose_tf.transform.rotation.z, pose_tf.transform.rotation.w)
    (roll, pitch, yaw) = euler_from_quaternion_fnc(pose_tf.transform.rotation.x, pose_tf.transform.rotation.y, pose_tf.transform.rotation.z, pose_tf.transform.rotation.w)
    initiated_loc = True

def move_robot():
    global yaw, goal_list,goal_idx, K_matrix_inv, y_array_output,data_point_array, velocity_publisher,initiated_array ,latest_y,initiated_y,c,current_loc
    vel_msg = Twist()
    while True:
        try:
            current_goal = goal_list[goal_idx]
            point_ahead_of_robot = current_loc + l * np.array([np.cos(yaw), np.sin(yaw)])
            distance_vec = current_goal - current_loc
            distance_len = np.linalg.norm(distance_vec)
            l_matrix = np.matmul(np.array([[1, 0], [0, 1 / l]]), np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]))
        except Exception:
            print("final point reached")
            velocity_publisher.publish(Twist())
            break



        c+=1
        try:
            c = 0

            if initiated_array == False:
                data_point_array.append(current_loc)
                K_matrix_inv = K_matrix_inv_cal_init()
                y_array.append(latest_y)
                y_array_output = np.array(y_array)
                x_array_data_collected.append(current_loc[0])
                y_array_data_collected.append(current_loc[1])
                u_initial = current_goal - current_loc
                u_rec_ahead = np.matmul(l_matrix, u_initial.T)

                vel_msg.linear.x = u_rec_ahead[0]
                vel_msg.linear.y = 0
                vel_msg.angular.z = u_rec_ahead[1]
                velocity_publisher.publish(vel_msg)
                # update distance
                distance_vec = current_goal - current_loc
                distance_len = np.linalg.norm(distance_vec)
                x_array_data_collected.append(float(current_loc[0]))
                y_array_data_collected.append(float(current_loc[1]))
                initiated_array = True

            elif goal_idx+1 <= len(goal_list):
                k_querry_array = list()
                # print("data_point_arr", data_point_array)
                # print("current_loc", current_loc)
                print(len(data_point_array))
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
                print(current_goal)
                mu = np.matmul(np.matmul(k_querry_array, K_matrix_inv), y_array_output)

                # covariance
                sig_querry = 1 - np.matmul(np.matmul(k_querry_array, K_matrix_inv), np.transpose(k_querry_array))
                # print(x_querry, "mu: ", mu, ";sigma: ", sig_querry)
                # _looks like h_gp is wrong a bit, it does not fall under 0 even when collided
                h_gp = mu - sig_querry-0.5
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
                    M = np.eye(2)
                    P = np.dot(M.T, M)
                    q = np.dot(M.T, -u_nom_xy)
                    # print(q)
                    # transform delta_hgp_over_x_regard_to_x_querry to good form to suit the calculation
                    delta_hgp_over_x_regard_to_x_querry_good_form = np.array([delta_hgp_over_x_regard_to_x_querry])
                    # print(delta_hgp_over_x_regard_to_x_querry_good_form)
                    G = -delta_hgp_over_x_regard_to_x_querry_good_form
                    h = np.array([h_gp])
                    # h = np.matmul(np.array([np.linalg.norm(delta_hgp_over_x_regard_to_x_querry),h_gp]), np.array([np.linalg.norm(delta_hgp_over_x_regard_to_x_querry),h_gp]) )
                    # print(h)

                    sol = quadprog_solve_qp(P, q, G, h)
                    u_rec_dx_dy = np.array(sol)
                    # append the current point before moving to data_array, calculate K and y_array
                    data_point_array.append(np.array([float(current_loc[0]), float(current_loc[1])]))
                    if len(data_point_array) > max_num_of_pts:
                        data_point_array.pop(0)
                    K_matrix_inv = K_matrix_inv_cal()
                    y_array.append(latest_y)
                    if len(y_array) > max_num_of_pts:
                        y_array.pop(0)
                    y_array_output = np.array(y_array)
                    l_matrix = np.matmul(np.array([[1, 0], [0, 1 / l]]), np.array(
                        [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]))
                    # turn u_rec_dx_dy into u_rec_ahead (velocity)
                    u_rec_ahead = np.matmul(l_matrix, u_rec_dx_dy.T)
                    #print("urec ahead raw", u_rec_ahead)
                    # print(sol)
                    if u_rec_ahead[0] >= 4:
                        u_rec_ahead[0] = 4
                    elif u_rec_ahead[0] > 0 and u_rec_ahead[0] < 0.5:
                        u_rec_ahead[0] = 0.5
                    elif u_rec_ahead[0] < -4:
                        u_rec_ahead[0] = -4
                    elif u_rec_ahead[0] < 0 and u_rec_ahead[0] > -0.5:
                        u_rec_ahead[0] = -0.5
                    #u_rec_ahead[1] = (u_rec_ahead[1] + np.pi) % (2 * np.pi) - np.pi
                    if u_rec_ahead[1] >= np.pi:
                        u_rec_ahead[1] = np.pi
                    elif u_rec_ahead[1] <= -np.pi:
                        u_rec_ahead[1] = -np.pi

                    vel_msg.linear.x = u_rec_ahead[0] * 0.1
                    vel_msg.linear.y = 0
                    vel_msg.angular.z = u_rec_ahead[1]* 0.5

                    velocity_publisher.publish(vel_msg)

                    print("u_rec: ", u_rec_ahead, " h_gp: ", h_gp, "mu: ", mu, "cov", sig_querry, "d_hgp/d_x: ",
                          delta_hgp_over_x_regard_to_x_querry_good_form, "distance to nearest obs: ",latest_y)

                    # update distance
                    distance_vec = current_goal - current_loc
                    distance_len = np.linalg.norm(distance_vec)
                    if distance_len <=0.1:
                        goal_idx+=1
                        print("reach goal: ", current_goal)
                    # print22(distance_len)
                    # print("robot pose",current_loc, " ", robot_angle)
                    x_array_data_collected.append(float(current_loc[0]))
                    y_array_data_collected.append(float(current_loc[1]))
                except ValueError:
                    velocity_publisher.publish(Twist())
            else:

                velocity_publisher.publish(Twist())
        except Exception:
            pass


#for getting u rec for each test point

#x_array_data_collected.append(0.)
#y_array_data_collected.append(0.)
#current_loc = np.array([0.,0.])
#data_point_array.append(current_loc)

# trial and error: work more likely than you think!

        #print(u_nom)


rospy.init_node('turtlebot3_controller', anonymous=True) # Creates a node with name 'turtlebot_controller' and make sure it is a unique node name (using anonymous=True).
velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10) # Publisher which will publish to the topic '/turtle1/cmd_vel'.
tf_subscriber=rospy.Subscriber('/tf', TFMessage, update_tf)


scan_subscriber = rospy.Subscriber('/scan', LaserScan, update_scan)

pose_tf=TransformStamped()

rate = rospy.Rate(200)
rospy.sleep(1)
#tst()
move_robot()
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

