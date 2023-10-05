import cirq, sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq



def build_OneQubit_quantum_circuit(state_dim, n_layers, noise):
    """ Builds a PQC for learning.

        Input
        -----
        state_dim : int
            dimension of environment state
        n_layers : int
            number of data re-uploading layers
        noise : float
            noise rate

        Output
        ------
        circuit : cirq.Circuit
            quantum circuit
        qubit : cirq.GridQubit.rect
            qubits of quantum circuit
        inputs : sympy.symbols
            input data and input weights on the input data
        variational_params : sympy.symbols
            trainable parameters for single qubit rotations in circuit
    """

    # Init circuit
    circuit = cirq.Circuit()

    # Init qubit
    qubit = cirq.GridQubit.rect(1, 1)

    # Inputs: input data and input weights on the input data
    inputs = sympy.symbols(f"x_(0:{n_layers})"+f"_(0:{state_dim})")

    # Variational parameters: trainable parameters for single qubit rotations in circuit
    variational_params = sympy.symbols(f"theta(0:{2 * n_layers * state_dim})")

    param_counter = 0

    # Add n_layers layers to circuit, consisting of single qubit rotations on each qubit and an entangling layer on all qubits
    for l in range(n_layers):

        # Add single qubit rotation gates to each qubit
        for i in range(state_dim):

            circuit.append(cirq.rx(inputs[i+l*state_dim])(qubit[0]))

            circuit.append(cirq.ry(variational_params[param_counter])(qubit[0]))
            circuit.append(cirq.rz(variational_params[param_counter+1])(qubit[0]))

            param_counter += 2

    circuit + cirq.Circuit(cirq.depolarize(noise).on_each(*circuit.all_qubits()))

    return circuit, qubit, inputs, variational_params



def build_quantum_circuit(n_qubits, n_layers, noise):
    """ Builds a PQC for learning.

        Input
        -----
        state_dim : int
            dimension of environment state
        n_layers : int
            number of data re-uploading layers
        noise : float
            noise rate

        Output
        ------
        circuit : cirq.Circuit
            quantum circuit
        qubit : cirq.GridQubit.rect
            qubits of quantum circuit
        inputs : sympy.symbols
            input data and input weights on the input data
        variational_params : sympy.symbols
            trainable parameters for single qubit rotations in circuit
    """

    # Init circuit:
    circuit = cirq.Circuit()

    # Init qubits:
    qubits = cirq.GridQubit.rect(1, n_qubits)

    # Inputs: input data and input weights on the input data
    inputs = sympy.symbols(f"x_(0:{n_layers})"+f"_(0:{n_qubits})")

    # Variational parameters: trainable parameters for single qubit rotations in circuit
    variational_params = sympy.symbols(f"theta(0:{2 * n_layers * n_qubits})")

    param_counter = 0

    # Add n_layers layers to circuit, consisting of single qubit rotations on each qubit and an entangling layer on all qubits:
    for l in range(n_layers):

        # Add single qubit rotation gates to each qubit:
        for q in range(n_qubits):

            circuit.append(cirq.rx(inputs[q+l*n_qubits])(qubits[q]))

            circuit.append(cirq.ry(variational_params[param_counter])(qubits[q]))
            circuit.append(cirq.rz(variational_params[param_counter+1])(qubits[q]))

            param_counter += 2

        # Add entangling layer, circular arrangement of cz gates:
        for q in range(n_qubits):

            if (q == (n_qubits-1)) and (n_qubits != 2):
                circuit.append(cirq.CZ(qubits[q], qubits[0]))

            elif (q == (n_qubits-1)) and (n_qubits == 2):
                pass

            else:
                circuit.append(cirq.CZ(qubits[q], qubits[q+1]))

    circuit + cirq.Circuit(cirq.depolarize(noise).on_each(*circuit.all_qubits()))

    return circuit, qubits, inputs, variational_params



class ReUploadingPQC(tf.keras.layers.Layer):

    """
    Class for reuploading PQC as a custom Keras layer.
    Class manages the trainable parameters (variational angles theta and input-scaling parameters lamda)
    Class resolves the input values (input states) into the appropriate symbols in the circuit.
    """

    def __init__(self, n_qubits, state_dim, n_layers, observables, noise, activation = "linear", name = "re-uploading-PQC"):

        super(ReUploadingPQC, self).__init__(name=name)

        self.n_layers = n_layers
        self.state_dim = state_dim

        # Construct circuit:
        if n_qubits == 1:
            circuit, qubits, inputs, variational_params = build_OneQubit_quantum_circuit(state_dim, n_layers, noise)
        elif n_qubits == 2:
            circuit, qubits, inputs, variational_params = build_quantum_circuit(state_dim, n_layers, noise)
        else:
            print("No implementation for defined number of qubits.")

        # Init trainable theta variables for single qubit rotations in PQC as random angle on x-axis:
        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable( initial_value=theta_init(shape=(1, len(variational_params)), dtype="float32"), trainable=True, name="thetas")

        # Init trainable variables for weights on input data as ones:
        lambda_init = tf.ones(shape=(self.state_dim * self.n_layers))
        self.lambdas = tf.Variable(initial_value=lambda_init, dtype="float32", trainable=True, name="lambdas")

        # Define symbol order (on variational params and weights on input data):
        symbols = [str(symbol) for symbol in variational_params + inputs]
        self.indicies = tf.constant([symbols.index(i) for i in sorted(symbols)])
        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computational_layer = tfq.layers.ControlledPQC(circuit, observables)


    def call(self, inputs):

        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuit = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim,1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1,self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lambdas, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)
        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indicies, axis=1)

        return self.computational_layer([tiled_up_circuit, joined_vars])



class Rescaling(tf.keras.layers.Layer):
    """ Layer for rescaling and weightening of observables (Pauli Z products) for Value function learning PQC """

    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.weight = tf.Variable(initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply(inputs, tf.repeat(self.weight,repeats=tf.shape(inputs)[0],axis=0))



class Alternating(tf.keras.layers.Layer):
    """ Layer for rescaling and weightening of observables (Pauli Z products) for policy-gradient PQC """

    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.)**i for i in range(output_dim)]]), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)



def generate_model_Critic(n_qubits, state_dim, n_layers, observables, noise):
    """ Generates a model for data re-uploading PQC value function approximator.

        Input
        -----
        n_qubits : int
            number of qubits in circuit
        state_dim : int
            dimension of environment state
        n_layers : int
            number of data re-uploading layers
        observables : [cirq.Z]
            circuit observables
        noise : float
            noise rate

        Output
        ------
        model : tf.keras.Model
            quantum machine learning model for value function approximation
    """

    input_tensor = tf.keras.Input(shape=(state_dim, ), dtype=tf.dtypes.float32, name="input")
    re_uploading_pqc = ReUploadingPQC(n_qubits, state_dim, n_layers, observables, noise, activation="tanh")([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name="values")
    values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=values)

    return model



def generate_model_Actor(n_qubits, state_dim, n_layers, n_actions, beta, observables, noise):
    """ Generates a PQC model for data re-uploading policy PQC.

        Input
        -----
        n_qubits : int
            number of qubits in circuit
        state_dim : int
            dimension of environment state
        n_layers : int
            number of data re-uploading layers
        n_actions : int
            number of possible actions
        beta : float
            inverse temperature parameter
        observables : [cirq.Z]
            circuit observables
        noise : float
            noise rate

        Output
        ------
        model : tf.keras.Model
            quantum machine learning model for value function approximation
    """

    input_tensor = tf.keras.Input(shape=(state_dim, ), dtype=tf.dtypes.float32, name="input")
    re_uploading_pqc = ReUploadingPQC(n_qubits, state_dim, n_layers, observables, noise)([input_tensor])
    process = tf.keras.Sequential([Alternating(n_actions),tf.keras.layers.Lambda(lambda x: x * beta),tf.keras.layers.Softmax()], name="observables-policy")
    policy = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

    return model



def create_pg_model(config, pg_path):
    """ Creates a quantum policy gradient model with optimizers to be used for a learning task.

        Input
        -----
        config : configparser.ConfigParser
            configuration settings for all parameters of the learning task
        pg_path : str
            path where model is to be saved

        Output
        ------
        pg_model : dict
            dictionary containing reinforcement learning model and optimizers
    """

    # Define observables:
    if config.getint("actor_circuit","qubits") == 1:
        qubits_a = cirq.GridQubit.rect(1, 1)
        ops_a = [cirq.Z(q) for q in qubits_a]
        observables_actor = [ops_a[0]]
    else:
        qubits_a = cirq.GridQubit.rect(1, config.getint("actor_circuit","qubits"))
        ops_a = [cirq.Z(q) for q in qubits_a]
        observables_actor = [ops_a[0]*ops_a[1]]

    state_dim = config.get("random_walker","start_state").split(",")
    state_dim = [int(i) for i in state_dim]

    # Init PG model for learning
    actor = generate_model_Actor(config.getint("actor_circuit","qubits"), len(state_dim), config.getint("actor_circuit","layers"), len(config.get("environment","actions").split(",")), config.getfloat("actor_learning_rates","beta"), observables_actor, config.getfloat("actor_circuit","noise"))

    # Init optimizers for learning:
    optimizer_in_pg = tf.keras.optimizers.Adam(learning_rate=config.getfloat("actor_learning_rates","a_in"), amsgrad=True)
    optimizer_var_pg = tf.keras.optimizers.Adam(learning_rate=config.getfloat("actor_learning_rates","a_var"), amsgrad=True)
    optimizer_out_pg = tf.keras.optimizers.Adam(learning_rate=config.getfloat("actor_learning_rates","a_out"), amsgrad=True)

    # Assign the model parameters to each optimizer
    w_in_pg, w_var_pg, w_out_pg = 1, 0, 2

    # Define dictionary for policy gradient model
    pg_model = {
        "actor": actor,
        "op_in": optimizer_in_pg,
        "op_var": optimizer_var_pg,
        "op_out": optimizer_out_pg,
        "w_in": w_in_pg,
        "w_var": w_var_pg,
        "w_out": w_out_pg,
        "actor_path": pg_path+"/",
    }
    return pg_model



def create_ac_model(config, actor_path, critic_path):
    """Creates a quantum actor-critic model with optimizers to be used for a learning task.

        Input
        -----
        config : configparser.ConfigParser
            configuration settings for all parameters of the learning task
        actor_path : str
            path where the actor quantum model is to be saved
        critic_path : str
            path where the critic quantum model is to be saved

        Output
        ------
        ac_model : dict
            dictionary containing reinforcement learning models and optimizers
    """

    # Define observables:
    if config.getint("critic_circuit","qubits") == 1:
        qubits_c = cirq.GridQubit.rect(1, 1)
        ops_c = [cirq.Z(q) for q in qubits_c]
        observables_critic = [ops_c[0]]
    else:
        qubits_c = cirq.GridQubit.rect(1, config.getint("critic_circuit","qubits"))
        ops_c = [cirq.Z(q) for q in qubits_c]
        observables_critic = [ops_c[0]*ops_c[1]]

    if config.getint("actor_circuit","qubits") == 1:
        qubits_a = cirq.GridQubit.rect(1, 1)
        ops_a = [cirq.Z(q) for q in qubits_a]
        observables_actor = [ops_a[0]]
    else:
        qubits_a = cirq.GridQubit.rect(1, config.getint("actor_circuit","qubits"))
        ops_a = [cirq.Z(q) for q in qubits_a]
        observables_actor = [ops_a[0]*ops_a[1]]

    state_dim = config.get("random_walker","start_state").split(",")
    state_dim = [int(i) for i in state_dim]

    # Init AC models for learning
    model_critic = generate_model_Critic(config.getint("critic_circuit","qubits"), len(state_dim), config.getint("critic_circuit","layers"), observables_critic, config.getfloat("critic_circuit","noise"))
    model_actor = generate_model_Actor(config.getint("actor_circuit","qubits"), len(state_dim), config.getint("actor_circuit","layers"), len(config.get("environment","actions").split(",")), config.getfloat("actor_learning_rates","beta"), observables_actor, config.getfloat("actor_circuit","noise"))

    # Init AC optimizers for learning:
    optimizer_in_actor = tf.keras.optimizers.Adam(learning_rate=config.getfloat("actor_learning_rates","a_in"), amsgrad=True)
    optimizer_var_actor = tf.keras.optimizers.Adam(learning_rate=config.getfloat("actor_learning_rates","a_var"), amsgrad=True)
    optimizer_out_actor = tf.keras.optimizers.Adam(learning_rate=config.getfloat("actor_learning_rates","a_out"), amsgrad=True)

    optimizer_in_critic = tf.keras.optimizers.Adam(learning_rate=config.getfloat("critic_learning_rates","c_in"), amsgrad=True)
    optimizer_var_critic = tf.keras.optimizers.Adam(learning_rate=config.getfloat("critic_learning_rates","c_var"), amsgrad=True)
    optimizer_out_critic = tf.keras.optimizers.Adam(learning_rate=config.getfloat("critic_learning_rates","c_out"), amsgrad=True)

    # Assign the model parameters to each optimizer
    w_in_actor, w_var_actor, w_out_actor = 1, 0, 2
    w_in_critic, w_var_critic, w_out_critic = 1, 0, 2

    # Define dictionary for Actor-Critic models
    ac_model = {
        "critic": model_critic,
        "op_in_c": optimizer_in_critic,
        "op_var_c": optimizer_var_critic,
        "op_out_c": optimizer_out_critic,
        "w_in_c": w_in_critic,
        "w_var_c": w_var_critic,
        "w_out_c": w_out_critic,
        "critic_path": critic_path+"/",

        "actor": model_actor,
        "op_in_a": optimizer_in_actor,
        "op_var_a": optimizer_var_actor,
        "op_out_a": optimizer_out_actor,
        "w_in_a": w_in_actor,
        "w_var_a": w_var_actor,
        "w_out_a": w_out_actor,
        "actor_path": actor_path+"/"
    }

    return ac_model
