# quantum_reinforcement_learning_of_classical_rare_trajectories
Quantum Reinforcement Learning of Classical Rare Trajectories
This repository contains the code used to generate the results of the following publication (ToDo:add link to paper). The code includes four different reinforcement learning implementations to learn rare dynamics of the random walker model.

## Abstract
Rare events are essential for understanding the behavior of non-equilibrium and industrial
systems. It is an open problem to effectively search for such rare events. To this end, specific
methods are required for generating rare events and sampling their statistics in an automated and
statistically meaningful way. With the advent of quantum computing and its potential advantages
over classical computing for applications like sampling certain probability distributions, the
question arises whether quantum computers could also provide an advantage or inspire new
methods for sampling the statistics of rare events. In this article, we propose several quantum
reinforcement learning (QRL) methods for studying rare dynamics and investigate their benefits
over classical approaches based on neural networks. As paradigmatic example, we demonstrate
that our QRL agents can learn and generate the rare dynamics of random walks, and we are
able to explain this success as well as the different contributing factors to it. Furthermore, we
show better learning behavior with fewer parameters compared to classical approaches. This
is the first investigation of QRL applied to generating rare events and suggests that QRL is a
promising method to study their dynamics and statistics.

## Installation
To run this code with all its packages you will need to use Python version 3.7. The TensorFlow and TensorFlow Quantum packages are required to run this code. See the installation commands below.\
A complete list on all requirements is provided in the 'requirements.txt' file.

```bash
pip install tensorflow=2.4.1
```

```bash
pip install tensorflow-quantum
```

Tutorials on how to install Tensorflow Quantum can be found online (https://www.tensorflow.org/quantum/install).

## Usage
You can configure the input parameters for the reinforcement learning agents in the config_files and then run the 'main'-scrips to train a new reinforcement learning agent.

Run
* main_QAC.py for a PQC-based actor-critic agent.
* main_QPG.py for a PQC-based policy gradient agent.
* main_NNAC.py for a NN-based actor-critic agent.
* main_NNPG.py for a NN-based policy gradient agent.

When you run the script, you will constantly get prints on the simulation process. After the learning process has finished, the code also automatically generates plots of the learning process.

Each script generates a folder in the 'Simulations'-folder with all the data generated during the script run. The resulting folder contains the following sub-folders:
* 'CSVs' folder: contains a CSV file with the implementation settings (all defined input parameters from the configuration files) and agent performance (returns per episode, returns per batch, etc.). Contains CSV files with all trajectories generated during learning. Contains CSV files with all trajectories generated with the trained agent.
* 'Models' folder: contains all model parameters. These files are crucial for loading the agent afterward.
* 'Plots' folder: All plots generated after learning will be saved here. After learning, the code automatically provides plots on the learning performance, policy, and value function.

The other scripts provide the functions for building the reinforcement learning agents, learning, and plotting.
* models_Q.py: contains the classes and functions to build a reinforcement learning agent with parameterized quantum circuits (PQC).
* models_NN.py: contains the classes and functions to build a reinforcement learning agent with classical neural networks.
* learn.py: contains the classes and functions for learning the rare dynamics of a random walker with either a policy gradient or an actor-critic approach.
* plot.py: contains all functions for generating the plots.

## Contributing

Contributions are highly appreciated. See CONTRIBUTING_LICENCE_AGREEMENT.md on how to get started.

If you have any feedback or want to propose a new feature, please open an issue. 😊


## Authors and acknowledgment

Laura Ohff, Alissa Wilms

The origin of this project is part of the Quantum research of Porsche Digital and the BMBF (HYBRID). ✨

## License

See [LICENSE](./LICENSE.md).

## Disclaimer
* This repository is based on Tensorflow 2.7.0 and other pinned dependencies that contain vulnerabilities. To ensure the reproducibility of the results in the published paper, these dependencies are not updated. We therefore ask you to use this scientific software with caution and not in productive environments.
