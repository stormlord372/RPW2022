import time

import numpy as np
import random
import cvxopt
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from matplotlib import cm
import quadprog
from visualize_mobile_robot import sim_mobile_robot
IS_SHOWING_2DVISUALIZATION = True
#y_i: scalar target
obstacle = np.array([[3.,1.],[-6.,-4.],[-8,0],[8,-10]])

xx_array = np.random.uniform(-10.,10.,1000)
xy_array = np.random.uniform(-10.,10.,1000)
xx_array_test = np.random.uniform(-20.,20.,2000)
xy_array_test = np.random.uniform(-20.,20.,2000)
y_array = []
sigma_f = 1
sigma_y = 0.01
r = 1
l = 0.06
lidar_range = 4
#querry_point = np.array([1,2])
data_point_array = []
data_point_array_test = []
def steering_angle_cal(a,b):
    inner = np.inner(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    return rad

for i in range(len(xx_array)):
    data_point_array.append([xx_array[i],xy_array[i]])

for i in range(len(xx_array_test)):
    data_point_array_test.append([xx_array_test[i],xy_array_test[i]])

#1000 data points,
data_point_array = np.array(data_point_array)
data_point_array_test = np.array(data_point_array_test)
#calculate y from X = shortest distance from current location to the nearest obstacle
for data_point in data_point_array:
    distance_1 = np.sqrt((data_point[0]-obstacle[0][0])**2 + (data_point[1]-obstacle[0][1])**2)-r
    distance_2 = np.sqrt((data_point[0]-obstacle[1][0])**2 + (data_point[1]-obstacle[1][1])**2)-r
    distance_3 = np.sqrt((data_point[0] - obstacle[2][0]) ** 2 + (data_point[1] - obstacle[2][1]) ** 2) - r
    distance_4 = np.sqrt((data_point[0] - obstacle[3][0]) ** 2 + (data_point[1] - obstacle[3][1]) ** 2) - r
    y_array.append(min(distance_1,distance_2,distance_3,distance_4,lidar_range))

y_array = np.array(y_array)
L = np.diag([0.1, 0.1])
Lminus2 = np.linalg.inv(np.linalg.inv(L))
K_matrix =[]
#calculate kernel (smallest_k), use it to contruct K matrix and its invert
for i in range(len(data_point_array)):
    medium_k = []
    for j in range(len(data_point_array)):
        diff = data_point_array[i]-data_point_array[j]

        smallest_k = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff))/2)+int(i==j)*sigma_y
        medium_k.append(smallest_k)
    K_matrix.append(medium_k)
K_matrix = np.array(K_matrix)

print(K_matrix)
#prev_cov = 0
K_matrix_inv = np.linalg.inv(K_matrix)
x_querry_array = data_point_array_test
print(x_querry_array.shape)
#u_array = []
hgp_array = []
mu_sig_array =[]
mu_array = []
sig_array =[]
goal_list = np.array([[6,1],[-10,-8],[-1,1],[-10,0]])
robot_angle = np.pi/6

x_querry_test =np.array([[0,0]])
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


#for getting color map for h_gp, use data set of 2000 points
for x_querry in x_querry_array:
    k_querry_array = list()
    for i in range(len(data_point_array)):
        diff = x_querry - data_point_array[i]
        k = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff))/2)
        k_querry_array.append(k)
    k_querry_array = np.array(k_querry_array)
    #average mu
    mu = np.matmul(np.matmul(k_querry_array,K_matrix_inv),y_array)
    #covariance
    sig_querry = 1 - np.matmul(np.matmul(k_querry_array,K_matrix_inv),np.transpose(k_querry_array))
    #print(x_querry, "mu: ", mu, ";sigma: ", sig_querry) control the distance by adjusting the offset
    h_gp = mu - sig_querry - 0.5
    mu_array.append(mu)
    sig_array.append(sig_querry)
    hgp_array.append(h_gp)
    mu_sig_pair = np.array([mu,sig_querry])
    mu_sig_array.append(mu_sig_pair)

#for getting u rec for each test point
x_array_data_collected = []
y_array_data_collected = []
x_array_data_collected.append(0.)
y_array_data_collected.append(0.)
x_querry_test = np.array([[0.,0.]])
current_loc = np.array([0.,0.])

h_gp_array_2 =[]


for goal in goal_list:
    #get distance and steering angle
    point_ahead_of_robot = current_loc + l*np.array([np.cos(robot_angle), np.sin(robot_angle)])
    distance_vec = goal - point_ahead_of_robot
    distance_len = np.linalg.norm(distance_vec)

    while distance_len >=0.1:
        point_ahead_of_robot = current_loc + l * np.array([np.cos(robot_angle), np.sin(robot_angle)])
        distance_vec = goal - point_ahead_of_robot
        distance_len = np.linalg.norm(distance_vec)
        """steering_angle = ((np.arctan2((goal[1] - current_loc[1]), (goal[0] - current_loc[0]))) - robot_angle)
        if steering_angle < -np.pi:
            steering_angle += 2 * np.pi
        elif steering_angle > np.pi:
            steering_angle -= 2 * np.pi

        u_nom = np.array([distance_len, steering_angle])"""
        #print(u_nom)
        #calculate h_gp for each point
        k_querry_array = list()
        for i in range(len(data_point_array)):
            diff = current_loc - data_point_array[i]
            k = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff))/2)
            k_querry_array.append(k)
        k_querry_array = np.array(k_querry_array)
        #average mu
        mu = np.matmul(np.matmul(k_querry_array,K_matrix_inv),y_array)
        #covariance
        sig_querry = 1 - np.matmul(np.matmul(k_querry_array,K_matrix_inv),np.transpose(k_querry_array))
        #print(x_querry, "mu: ", mu, ";sigma: ", sig_querry)
        #_off set h_gp to get better safe distance
        h_gp = mu-sig_querry-0.5
        #u_array.append(goal-x_querry)
        #calculate delta_hgp_over_x_regard_to_x_querry (step 4, equation 14-15)
        delta_kernel_array_sim = []

        for i in range(len(data_point_array)):
            diff = data_point_array[i] - current_loc
            k = np.exp(-np.matmul(np.matmul(diff, Lminus2), np.transpose(diff)) / 2)
            kernel = np.matmul(diff*k,Lminus2)
            delta_kernel_array_sim.append(kernel)
        delta_kernel_array_sim = np.array(delta_kernel_array_sim)


        #delta_hgp_over_x_regard_to_x_querry is d_hgp/d_x regarding x_query
        # set up constrain equation 16
        delta_hgp_over_x_regard_to_x_querry = np.matmul(np.matmul(y_array,K_matrix_inv),delta_kernel_array_sim) + 2 * np.matmul(np.matmul(k_querry_array, K_matrix_inv), delta_kernel_array_sim)
        delta_hgp_over_x_regard_to_x_querry = np.array(delta_hgp_over_x_regard_to_x_querry)
        #u_nom_xy is the vector between goal and current location
        u_nom_xy = goal-point_ahead_of_robot
        M = np.eye(2)
        P = np.dot(M.T, M)
        q = np.dot(M.T, -u_nom_xy)
        #print(q)
        #transform delta_hgp_over_x_regard_to_x_querry to array form to suit the calculation
        delta_hgp_over_x_regard_to_x_querry_good_form = np.array([delta_hgp_over_x_regard_to_x_querry])
        #print(delta_hgp_over_x_regard_to_x_querry_good_form)
        G = -delta_hgp_over_x_regard_to_x_querry_good_form
        h = np.array([h_gp])

        h_gp_array_2.append(h_gp)

        sol = quadprog_solve_qp(P, q, G, h)
        #sol is the result calculation from quadratic calc
        u_rec =np.array(sol)
        #use u_rec (x,y)
        l_matrix = np.matmul(np.array([[1,0],[0, 1/l]]),np.array([[np.cos(robot_angle),np.sin(robot_angle)],[-np.sin(robot_angle),np.cos(robot_angle)]]))
        u_rec_ahead = np.matmul(l_matrix,u_rec.T)

        G_matrix = np.array([[np.cos(robot_angle),0],
                             [np.sin(robot_angle),0],
                             [0,1.]])
        #print(sol)
        distance_to_obstacle = min((np.sqrt((current_loc[0]-obstacle[0][0])**2 + (current_loc[1]-obstacle[0][1])**2)-r), (np.sqrt((current_loc[0]-obstacle[1][0])**2 + (current_loc[1]-obstacle[1][1])**2)-r),(np.sqrt((current_loc[0]-obstacle[2][0])**2 + (current_loc[1]-obstacle[2][1])**2)-r),(np.sqrt((current_loc[0]-obstacle[3][0])**2 + (current_loc[1]-obstacle[3][1])**2)-r))
        print("result u_nom: ", "u_rec: ", u_rec_ahead," h_gp: ", h_gp, "d_hgp/d_x: ",delta_hgp_over_x_regard_to_x_querry_good_form , "distance to nearest obs: ", distance_to_obstacle)
        delta_pose = np.matmul(G_matrix, u_rec_ahead)
        delta_position = np.array([delta_pose[0], delta_pose[1]])
        current_loc += 0.01 * delta_position
        robot_angle += 0.01 * delta_pose[2]
        robot_angle= (robot_angle + np.pi) % (2 * np.pi) - np.pi
        #update distance
        distance_vec = goal - current_loc
        distance_len = np.linalg.norm(distance_vec)
        #print22(distance_len)
        #print("robot pose",current_loc, " ", robot_angle)
        x_array_data_collected.append(float(current_loc[0]))
        y_array_data_collected.append(float(current_loc[1]))
        if len(x_array_data_collected) > 10000:
            break

        #time.sleep(0.01)

#hgp_array = np.array(hgp_array)
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
print(len(x_array_data_collected))
mu_sig_array = np.array(mu_sig_array)
#oh no codes for the plots

f1 = plt.figure(1)
axs1 = f1.add_subplot(221)
axs2 = f1.add_subplot(222)
axs3 = f1.add_subplot(223)

# note to self: read widhi mail about improving the graph
obs1=plt.Circle((3,1),r,color="white", label= "obstacle")
obs2=plt.Circle((-6,-4),r,color="white", label= "obstacle")
obs3=plt.Circle(obstacle[2],r,color="white", label= "obstacle")
obs4=plt.Circle(obstacle[3],r,color="white", label= "obstacle")
cset = axs1.tricontourf(xx_array_test,xy_array_test, hgp_array)
axs1.add_patch(obs1)
axs1.add_patch(obs2)
axs1.add_patch(obs3)
axs1.add_patch(obs4)

axs1.legend()
plt.tight_layout()

axs1.set_title("overall safety hgp")
cset2 = axs2.tricontourf(xx_array_test,xy_array_test, mu_array)
obs12=plt.Circle((3,1),r,color="white", label= "obstacle")
obs22=plt.Circle((-6,-4),r,color="white")
obs32=plt.Circle(obstacle[2],r,color="white")
obs42=plt.Circle(obstacle[3],r,color="white")
axs2.add_patch(obs32)
axs2.add_patch(obs42)
axs2.add_patch(obs12)
axs2.add_patch(obs22)
axs2.set_title("safety belief mu")
axs2.legend()
plt.tight_layout()
cset3 = axs3.tricontourf(xx_array_test,xy_array_test, sig_array)
obs13=plt.Circle((3,1),r,color="white", label= "obstacle")
obs23=plt.Circle((-6,-4),r,color="white")
axs3.add_patch(obs13)
axs3.add_patch(obs23)
obs33=plt.Circle(obstacle[2],r,color="white")
obs43=plt.Circle(obstacle[3],r,color="white")
axs3.add_patch(obs33)
axs3.add_patch(obs43)

axs3.set_title("safety uncertainty sigma_squared")
plt.colorbar(cset)
plt.colorbar(cset2)
plt.colorbar(cset3)
axs3.legend()
plt.tight_layout()
f2 = plt.figure(2)
axs4 = f2.add_subplot(111)
csetf2 = axs4.tricontourf(xx_array_test,xy_array_test, hgp_array)
obs14=plt.Circle((3,1),r,color="white", label= "obstacle")
obs24=plt.Circle((-6,-4),r,color="white")
axs4.add_patch(obs14)
axs4.add_patch(obs24)
obs34=plt.Circle(obstacle[2],r,color="white")
obs44=plt.Circle(obstacle[3],r,color="white")
axs4.add_patch(obs34)
axs4.add_patch(obs44)
axs4.plot(6,1,marker="x", markersize=10, markeredgecolor="red", label= "goal1")
axs4.plot(-10,-8,marker="x", markersize=10, markeredgecolor="white", label= "goal2")
axs4.plot(-1,1,marker="x", markersize=10, markeredgecolor="orange", label= "goal3")

axs4.plot(-10,0,marker="x", markersize=10, markeredgecolor="black", label= "goal4")
axs4.plot(0,0,marker="o", markersize=5, markeredgecolor="red",markerfacecolor="blue" , label= "starting_point")
axs4.plot(x_array_data_collected,y_array_data_collected,marker=".", markersize=1, markeredgecolor="black",markerfacecolor="black",label= "robot_path")
axs4.set_title("robot path and safety belief area map")
plt.colorbar(csetf2)
plt.tight_layout()
plt.legend()

f3 = plt.figure(3)
axs5 = f3.add_subplot(111)
axs5.plot(h_gp_array_2)
plt.show()


