import numpy as np
from qiskit import *
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.algorithms.optimizers import SPSA
import matplotlib.pyplot as plt

########## TIC TAC TOE FUNCTIONS ##########

# helper function to determine valid tic tac toe board positions
def get_winner(board):
        # Check the board for any winning combinations
    winning_combinations = [
        # Rows
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        # Columns
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        # Diagonals
        (0, 4, 8),
        (2, 4, 6),
    ]
    x_wins = False
    o_wins = False
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != '':
            if board[combo[0]] == 'x':
                return [-1,-1,1]
            else:
                return [1,-1,-1]
    return [-1,1,-1]
    
def is_valid_tic_tac_toe(board):
    # Check that the board has exactly 9 elements
    if len(board) != 9:
        return False
    # Count the number of 'x' and 'o' on the board
    count_x = board.count('x')
    count_o = board.count('o')
    # Check that the difference in count between 'x' and 'o' is 0 or 1
    if abs(count_x - count_o) > 1:
        return False
    # Check the board for any winning combinations
    winning_combinations = [
        # Rows
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        # Columns
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        # Diagonals
        (0, 4, 8),
        (2, 4, 6),
    ]
    x_wins = False
    o_wins = False
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != '':
            if board[combo[0]] == 'x':
                x_wins = True
            else:
                o_wins = True
    # Check if both 'x' and 'o' won or if neither won
    if x_wins and o_wins or (not x_wins and not o_wins):
        return False
    # Check that the board is a valid final board configuration
    if (x_wins and count_x != count_o + 1) or (o_wins and count_x != count_o):
        return False
    # All checks have passed, so the board is valid
    return True

def generate_tic_tac_toe_configs():
    valid_configs = []
    winners = []
    # Generate all possible configurations of the board
    for i in range(3**9):
        board = []
        for j in range(9):
            symbol = ''
            if i % 3 == 0:
                symbol = 'x'
            elif i % 3 == 1:
                symbol = 'o'
            board.append(symbol)
            i //= 3
        # Check if the configuration is valid
        if is_valid_tic_tac_toe(board):
            valid_configs.append(board)
            winners.append(get_winner(board))
    return valid_configs, winners

########## ENCODING TTT INTO CIRCUITS ##########

def encode_data(tic_tac_toe_field, circuit):
    data_g = [1 if entry == 'x' else -1 if entry == 'o' else 0 for entry in tic_tac_toe_field ]
    for entry, index in zip(data_g, range(len(data_g))):
        circuit.rx(entry * 2 * np.pi / 3, index)
    return circuit

def add_single_qubit_gates(params, circuit):
    corner_qubits = [0,2,6,8]
    edge_qubits = [1,3,5,7]
    center_qubit = 4
    # corners (green)
    for i in corner_qubits:
        circuit.rx(params[0],i)
        circuit.ry(params[1],i)
    # edges (red)
    for i in edge_qubits: 
        circuit.rx(params[2],i)
        circuit.ry(params[3],i)
    # middle (yellow)
    circuit.rx(params[4],center_qubit)
    circuit.rx(params[5],center_qubit)
    return circuit

def add_two_qubit_gates(params, circuit):
    # corners (green)
    corner_qubits = [0,2,6,8]
    edge_qubits = [1,3,5,7]
    center_qubit = 4
    ## For the parameters, the first is for the yellow, the second one for the red, and the last one for the green. 
    # yellow two-qubit gates
    for i in corner_qubits:
        circuit.cry(params[0],center_qubit,i)
    # red two-qubit gates
    for i in edge_qubits: 
        circuit.cry(params[1],i,center_qubit)
    # green two-qubit gates
    circuit.cry(params[2],0,1)
    circuit.cry(params[2],0,3)
    circuit.cry(params[2],2,1)
    circuit.cry(params[2],2,5)
    circuit.cry(params[2],6,3)
    circuit.cry(params[2],6,7)
    circuit.cry(params[2],8,5)
    circuit.cry(params[2],8,7)
    return circuit

def l2_loss(output, target):
    output, target = np.array(output), np.array(target)
    return np.sum(np.abs(output - target)**2)

########## GENRATE DATA ##########

boards, winners = generate_tic_tac_toe_configs()
x = boards
y = winners
# shuffle the indices
shuffle_indices = np.random.permutation(len(x))
train_size = int(len(x) * 0.3)
# split the indices into training and testing sets
train_indices = np.array(shuffle_indices[:train_size])
test_indices = np.array(shuffle_indices[train_size:])
# create the training and testing sets
x_train, y_train = np.take(x, train_indices, axis=0), np.take(y, train_indices, axis=0)
x_test, y_test = np.take(x, test_indices, axis=0), np.take(y, test_indices, axis=0)
# print("Example train data: ", x_train[17], y_train[17])

########## TRAIN THE CIRCUIT ##########

# Define the optimizer
class OptimizerLog:
    """Log to store optimizer's intermediate results"""
    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
    def update(self, evaluation, parameter, cost, _stepsize, _accept):
        """Save intermediate results. Optimizer passes five values
        but we ignore the last two."""
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)

log = OptimizerLog()
optimizer = SPSA(maxiter=50,callback=log.update)

def predict(data, params):
    circ = QuantumCircuit(9)
    circ.params = params
    encode_data(data, circ)
    circ.barrier()
    add_single_qubit_gates(params[0:6], circ)
    circ.barrier()
    add_two_qubit_gates(params[6:9], circ)
    estimator = Estimator()
    circuits = (
        circ,
        circ,
        circ
    )
    observables = (
        SparsePauliOp("ZIZIIIZIZ") / 4.0,
        SparsePauliOp("IZIZIZIZI") / 4.0,
        SparsePauliOp("IIIIZIIII")
    )
    job = estimator.run(circuits, observables)
    result = job.result()
    results = result.values.tolist()
    exp_val_o = results[0]
    exp_val_draw = results[2]
    exp_val_x = results[1]
    return [exp_val_o, exp_val_draw, exp_val_x]

def cost_function(params, data, labels):
    y = labels
    cost = 0
    for i in range(len(data)):
        output = predict(data[i],params)
        pos = np.argmax(np.abs(output))
        output = np.zeros(3)
        output[pos] = 1
        cost += l2_loss(output, y[i])
    return cost/len(data)

def objective_function(variational):
    """Cost function of circuit parameters on training data.
    The optimizer will attempt to minimize this."""
    return cost_function(variational,x_train, y_train)

# Initialize the parameters
params = np.random.rand(9)*2*np.pi
# Train the circuit
print('Initial parameters:', params)
# print(cost_function(params,x_train,y_train))
result = optimizer.minimize(fun=objective_function,x0= params)
opt_var = result.x
opt_value = result.fun

fig = plt.figure()
plt.plot(log.evaluations, log.costs)
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.show()

