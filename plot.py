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
                if agent_type == "PQC":
                    action_probability = actor([state])
                if agent_type == "NN":
                    action_probability = actor(state)
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
                if agent_type == "PQC":
                    state = tf.convert_to_tensor([state])
                if agent_type == "NN":
                    state = tf.convert_to_tensor(state)
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

    color_change = int(n_episodes/len(colors))

    files = [path+"/CSVs/Trajectories during learning/"+f for f in os.listdir(path+"/CSVs/Trajectories during learning/")]

    color_index = 0
    traj_counter = 0

    pylab.figure(0)

    for file in files:

        with open(file,"r") as f:

            reader = csv.reader(f, delimiter=",")

            # Skip header
            next(reader)

            for row in reader:

                t = row[4].replace("(", "").replace(")", "").replace(",", "")
                t = t.split()
                t = [int(ts) for ts in t]
                x = row[5].replace("(", "").replace(")", "").replace(",", "")
                x = x.split()
                x = [int(xs) for xs in x]

                try:
                    pylab.plot(t, x, color=colors[color_index])
                except:
                    pylab.plot(t, x, color=colors[len(colors)-1])

                traj_counter += 1

                if traj_counter >= color_change:
                    traj_counter = 0
                    color_index +=1

    pylab.xlabel("t")
    pylab.ylabel("x")
    title="Trajectories generated during learning"
    pylab.title(title)
    pylab.savefig(path+"/Plots/"+name+" "+title+".png", bbox_inches="tight")
    pylab.close(0)

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

    files = [path+"/CSVs/Trajectories after learning/"+f for f in os.listdir(path+"/CSVs/Trajectories after learning/")]

    colors = [color_lines,"#B2B2B2"]
    labels = ["Rare", "Failed"]

    traj_counter = 1

    rare_traj_counter = 0

    pylab.figure(0)

    for file in files:

        with open(file,"r") as f:

            reader = csv.reader(f, delimiter=",")

            # Skip header
            next(reader)

            for row in reader:

                t = row[3].replace("(", "").replace(")", "").replace(",", "")
                t = t.split()
                t = [int(ts) for ts in t]
                x = row[4].replace("(", "").replace(")", "").replace(",", "")
                x = x.split()
                x = [int(xs) for xs in x]

                if str(row[1]) == "True":
                    color = 0
                    rare_traj_counter += 1
                else:
                    color = 1

                try:
                    pylab.plot(t, x, color=colors[color])
                except:
                    pylab.plot(t, x, color=colors[len(colors)-1])

    pylab.xlabel("t")
    pylab.ylabel("x")
    title="Trajectories generated after learning (" +str(rare_traj_counter)+" rare trajectories out of "+str(n_episodes)+" total)"
    pylab.title(title)
    legend_elements = [mpl.patches.Patch(color=colors[0], label="Rare"),mpl.patches.Patch(color=colors[1], label="Fail")]
    pylab.legend(loc="upper left", handles=legend_elements)
    pylab.savefig(path+"/Plots/"+name+" "+title+".png", bbox_inches="tight")
    pylab.close(0)

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

    episodes = range(1, len(return_per_episode)+1)

    pylab.figure(0)
    pylab.plot(episodes, return_per_episode, color=color_lines)
    pylab.xlabel("Episode")
    pylab.ylabel("Return per Episode")
    title="Return per episode"
    pylab.savefig(path+"/Plots/"+name+" "+title+".png", bbox_inches="tight")
    pylab.close(0)

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

    batches = range(1, len(return_per_batch)+1)

    pylab.figure(0)
    pylab.plot(batches, return_per_batch, color=color_lines)
    pylab.xlabel("Batch")
    pylab.ylabel("Return per batch")
    title="Return per batch"
    pylab.savefig(path+"/Plots/"+name+" "+title+".png", bbox_inches="tight")
    pylab.close(0)

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

    episodes = range(1, len(dif_counts)+1)

    pylab.figure(0)
    pylab.plot(episodes, dif_counts, color=color_lines)
    pylab.xlabel("Episode")
    pylab.ylabel("Count of dif rare trajectory per episode")
    title="Count of dif rare trajectories per episode"
    pylab.savefig(path+"/Plots/"+name+" "+title+".png", bbox_inches="tight")
    pylab.close(0)

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

    episodes = range(1, len(avg_probs)+1)

    pylab.figure(0)
    pylab.plot(episodes, avg_probs, color=color_lines)
    pylab.xlabel("Episode")
    pylab.ylabel("Probability of generating rare trajectory")
    title="Probability of generating rare trajectory"
    pylab.savefig(path+"/Plots/"+name+" "+title+".png", bbox_inches="tight")
    pylab.close(0)

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

    pylab.figure(0)
    pylab.plot(batches, batch_counts, color=color_lines)
    pylab.xlabel("Batch")
    pylab.ylabel("Count of rare trajectories generated in batch")
    title="Count of rare trajectories generated in batch"
    pylab.savefig(path+"/Plots/"+str(name)+" "+title+".png", bbox_inches="tight")
    pylab.close(0)

    pylab.figure(1)
    pylab.plot(batches, batch_probs, color=color_lines)
    pylab.xlabel("Batch")
    pylab.ylabel("Probability of rare trajectory generated in batch")
    title="Probability of rare trajectory generated in batch"
    pylab.savefig(path+"/Plots/"+str(name+1)+" "+title+".png", bbox_inches="tight")
    pylab.close(1)

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

    batches = range(1, len(loss)+1)

    pylab.figure(0)
    pylab.plot(batches, loss, color=color_lines)
    pylab.xlabel("Batch")
    pylab.ylabel("Critic loss")
    title="Critic loss"
    pylab.savefig(path+"/Plots/"+name+" "+title+".png", bbox_inches="tight")
    pylab.close(0)

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

    batches = range(1, len(loss)+1)

    pylab.figure(0)
    pylab.plot(batches, loss, color=color_lines)
    pylab.xlabel("Batch")
    pylab.ylabel("Actor loss")
    title="Actor loss"
    pylab.savefig(path+"/Plots/"+name+" "+title+".png", bbox_inches="tight")
    pylab.close(0)

    return
