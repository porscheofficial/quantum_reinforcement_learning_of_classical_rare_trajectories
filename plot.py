import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import pandas as pd
import os
import csv
import tensorflow as tf
from scipy.interpolate import griddata
import learn

### CONSTANTS ###
# global settings
global_size = 18
font_size = 18

mpl.rcParams["font.serif"] = "cmr10"
mpl.rcParams["axes.labelsize"] = global_size
mpl.rcParams["axes.titlesize"] = font_size
mpl.rcParams["font.size"] = font_size
mpl.rcParams["xtick.labelsize"] = global_size
mpl.rcParams["ytick.labelsize"] = global_size
plt.rcParams.update({"font.size": font_size})

color_lines = "#2B8A9D"



def interpolate_array(array):
    """Helper function to interpolate an array for plotting the policy.

        Input
        -----
        array : [float]
            array to be interpolated

        Output
        ------
        array_interpl : [float]
            interpolated array
    """

    x = np.arange(0, np.array(array).shape[1])
    y = np.arange(0, np.array(array).shape[0])

    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)

    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    array_interpl = griddata((x1, y1), newarr.ravel(),(xx, yy),method="linear")

    return array_interpl



def map_index(x, T):
    """Helper function to map a state position x to an index in an array for plotting the policy.

        Input
        -----
        x : int
            x position of a environment state
        T : int
            final endtime for trajectory

        Output
        ------
        int
            index to be used for x
    """

    if x < 0:
        return T + (-1)*x
    else:
        return T - x



def save_plot(path, name, title, xlabel, ylabel, figure_num=0):
    """Helper function to save a plot."""
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title)
    pylab.savefig(f"{path}/Plots/{name} {title}.png", bbox_inches="tight")
    pylab.close(figure_num)

def process_trajectory_file(file, colors, color_index, traj_counter, color_change, t_col, x_col):
    """Helper function to process a trajectory file."""
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # Skip header
        for row in reader:
            if len(row) > max(t_col, x_col):
                t = list(map(int, row[t_col].replace("(", "").replace(")", "").replace(",", "").split()))
                x = list(map(int, row[x_col].replace("(", "").replace(")", "").replace(",", "").split()))
                try:
                    pylab.plot(t, x, color=colors[color_index])
                except:
                    pylab.plot(t, x, color=colors[-1])
                traj_counter += 1
                if traj_counter >= color_change:
                    traj_counter = 0
                    color_index += 1
    return color_index, traj_counter

def plot_trajectories(path, name, n_episodes, folder, colors, t_col, x_col, labels=None):
    """Helper function to plot trajectories."""
    files = [f"{path}/CSVs/{folder}/{f}" for f in os.listdir(f"{path}/CSVs/{folder}/")]
    color_index = 0
    traj_counter = 0
    pylab.figure(0)
    for file in files:
        color_index, traj_counter = process_trajectory_file(file, colors, color_index, traj_counter, int(n_episodes / len(colors)), t_col, x_col)
    save_plot(path, name, f"Trajectories {folder.replace('_', ' ')}", "t", "x")

def plot_data(path, name, data, xlabel, ylabel, title, figure_num=0):
    """Helper function to plot data."""
    pylab.figure(figure_num)
    pylab.plot(range(1, len(data) + 1), data, color=color_lines)
    save_plot(path, name, title, xlabel, ylabel, figure_num)

def plot_final_policy(path, name, T, actor, agent_type):
    """Plot the policy of a trained model.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        T : int
            final endtime for trajectory
        actor : tf.keras.Model
            reinforcement learning model for policy
        agent_type : str
            information wether the agent is a quantum or a NN agent
    """

    header = ["t", "x","probability of action_0", "probability of action_1"]
    data = []

    # For plotting surfaces
    t_array= []
    x_array= []
    prob0_array= []
    prob1_array= []

    array0 = [[np.NaN for i in range(T)] for j in range(2*T+1)]
    array1 = [[np.NaN for i in range(T)] for j in range(2*T+1)]

    ts = range(0,T)
    xs = range(-T,T+1)

    for t in ts:
        for x in xs:
            if (-t <= x <= t) and ((((t % 2) == 0) and ((x % 2) == 0)) or (((t % 2) != 0) and ((x % 2) != 0))):

                t_array.append(t)
                x_array.append(x)

                j = map_index(x,T)

                # Compute policy
                state = np.array([t,x])
                state = tf.convert_to_tensor([state])
                action_probability = actor([state]) if agent_type == "PQC" else actor(state)
                policy = action_probability.numpy()[0]

                array0[j][t] = policy[0]
                array1[j][t] = policy[1]

                prob0_array.append(policy[0])
                prob1_array.append(policy[1])

                data.append([t,x, policy[0], policy[1]])

    data_csv = pd.DataFrame(data, columns=header)
    data_csv.to_csv(path+"/"+name+" Final policy probabilities.csv", index=False)

    extent =  0, T, -T, T

    for i in range(0,int(T/2)):
        array0 = np.delete(array0,0,0)
        array0 = np.delete(array0,-1,0)

    extent =  0, T, -int(T/2), int(T/2)
    plt.figure(0)
    plt.imshow(array0, cmap="viridis", extent=extent,aspect="auto")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.savefig(path+"/"+name+" Half Probability of action 0 (going down).png", bbox_inches="tight")
    plt.close(0)

    array2 = interpolate_array(array0)

    plt.figure(2)
    plt.imshow(array2, cmap="viridis", extent=extent, interpolation= "bilinear",aspect="auto")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.savefig(path+"/"+name+" Half Probability of action 0 (going down) Interpolated.png", bbox_inches="tight")
    plt.close(2)

    return



def plot_final_value_function(path, name, T, critic, agent_type):
    """Plot the value function approximation of a trained model.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        T : int
            final endtime for trajectory
        critic : tf.keras.Model
            reinforcement learning model for value function approximation
        agent_type : str
            information wether the agent is a quantum or a NN agent
    """

    header = ["t", "x","state value"]
    data = []

    # For plotting surfaces
    t_array= []
    x_array= []
    state_values= []

    array0 = [[np.NaN for i in range(T+1)] for j in range(2*T+1)]

    ts = range(0,T+1)
    xs = range(-T,T+1)

    for t in ts:
        for x in xs:
            if (-t <= x <= t) and ((((t % 2) == 0) and ((x % 2) == 0)) or (((t % 2) != 0) and ((x % 2) != 0))):

                t_array.append(t)
                x_array.append(x)

                j = map_index(x,T)

                # Compute policy
                state = np.array([t,x])
                state = tf.convert_to_tensor([state]) if agent_type == "PQC" else tf.convert_to_tensor(state)
                state_value = critic([state])
                value = state_value.numpy()[0][0]

                array0[j][t] = value

                state_values.append(value)

                data.append([t,x,value])

    data_csv = pd.DataFrame(data, columns=header)
    data_csv.to_csv(path+"/"+name+" Final value function.csv", index=False)

    extent =  0, T, -T, T

    plt.figure(0)
    plt.imshow(array0, cmap="plasma", extent=extent,aspect="auto")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.savefig(path+"/"+name+" Final value function.png", bbox_inches="tight")
    plt.close(0)

    array2 = interpolate_array(array0)

    plt.figure(2)
    plt.imshow(array2, cmap="plasma", extent=extent, interpolation= "bilinear",aspect="auto")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.savefig(path+"/"+name+" Final value function Interpolated.png", bbox_inches="tight")
    plt.close(2)

    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(20, -50)
    trisurf = ax.plot_trisurf(x_array,t_array,state_values,cmap="plasma")
    fig.colorbar(trisurf, cax=fig.add_axes([0.85, 0.25, 0.05, 0.5]))
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("Probabilities for action 0")
    fig.savefig(path+"/"+name+" Final value function Surface.png", bbox_inches="tight")
    plt.close(4)

    return



def plot_trajectories_learning(path, name, n_episodes):
    """Plot all trajectories generated during learning.
        Trajectories are plottet with a gradient starting with a clear gray for early generated trajectories
        and blue for trajectories generated at the end of the learning process.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        n_episodes : int
            number of total training episodes
    """

    colors = ["#f8f8f8","#f6f6f6", "#f4f4f4","#f2f2f2", "#f0f0f0","#eeeeee", "#ececec", "#eaeaea", "#e8e8e8", "#e6e6e6", "#e4e4e4","#e2e2e2", "#e0e0e0","#dedede", "#dcdcdc","#dadada", "#d8d8d8","#d6d6d6", "#d4d4d4"]
    colors.append(color_lines)

    plot_trajectories(path, name, n_episodes, "Trajectories during learning", colors, 4, 5)

    return



def plot_trajectories_after_learning(path, name, n_episodes):
    """Plot all trajectories generated after learning.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        n_episodes : int
            number of total trajectories generated after learning
    """

    colors = [color_lines,"#B2B2B2"]
    labels = ["Rare", "Failed"]

    plot_trajectories(path, name, int(n_episodes), "Trajectories after learning", colors, 3, 4)

    return



def plot_return_per_episode(path, name, return_per_episode):
    """Plot return per episode.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        return_per_episode : [float]
            list of returns per episode
    """

    plot_data(path, name, return_per_episode, "Episode", "Return per Episode", "Return per episode")

    return



def plot_avg_return_per_batch(path, name, return_per_batch):
    """Plot mean return per batch.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        return_per_batch : [float]
            list of returns per batch
    """

    plot_data(path, name, return_per_batch, "Batch", "Return per batch", "Return per batch")

    return



def plot_dif_rare_per_episode(path, name, dif_counts):
    """Plot acumulated number of different rare trajectories generated up to each episode during the lerning process.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        dif_counts : [int]
            list of number of different rare trajectories generated up to each episode
    """

    plot_data(path, name, dif_counts, "Episode", "Count of dif rare trajectory per episode", "Count of dif rare trajectories per episode")

    return



def plot_avg_probability(path, name, avg_probs):
    """Plot probabilities of generating a rare trajectory for each episode.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        avg_probs : [float]
            list of probabilities of generating a rare trajectory for each episode
    """

    plot_data(path, name, avg_probs, "Episode", "Probability of generating rare trajectory", "Probability of generating rare trajectory")

    return



def plot_rare_count_per_batch(path, name, batch_counts, batch_size):
    """Plot count of rare trajectories generated in a batch during the lerning process.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        batch_counts : [int]
            list of count of rare trajectories generated in each batch
        batch_size : int
            batch size
    """

    batch_counts_sum = 0
    batch_sum = 0

    batches = range(1, len(batch_counts)+1)
    batch_probs = []
    batch_probs_running = []

    for i, n in enumerate(batch_counts):
        batch_counts_sum += n
        batch_sum += 1
        batch_probs.append(n/batch_size)
        batch_probs_running.append(batch_counts_sum/batch_sum)

    plot_data(path, name, batch_counts, "Batch", "Count of rare trajectories generated in batch", "Count of rare trajectories generated in batch")
    plot_data(path, name, batch_probs, "Batch", "Probability of rare trajectory generated in batch", "Probability of rare trajectory generated in batch", figure_num=1)

    return



def plot_critic_loss(path, name, loss):
    """Plot critic loss.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        loss : [float]
            list of critic loss per batch
    """

    plot_data(path, name, loss, "Batch", "Critic loss", "Critic loss")

    return



def plot_actor_loss(path, name, loss):
    """Plot actor loss.

        Input
        -----
        path : str
            path where plot is to be saved
        name : str
            name for plot title
        loss : [float]
            list of actor loss per batch
    """

    plot_data(path, name, loss, "Batch", "Actor loss", "Actor loss")

    return