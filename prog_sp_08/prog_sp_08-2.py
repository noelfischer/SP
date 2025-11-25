#prog_sp_08-2.py viterbi algorithm  v.1
import numpy as np

def viterbi_algorithm(o_seq_idx, state, prob_init, prob_a, prob_b):
    # initialization
    len_s = len(state)
    len_o = len(o_seq_idx)
    viterbi = np.zeros((len_s, len_o))
    backpointer = np.zeros((len_s, len_o), dtype=int)

    # first step
    viterbi[:, 0] = prob_init * prob_b[:, o_seq_idx[0]]
    print("viterbi[:,0] =",viterbi[:,0])

    # recursive step
    for t in range(1, len_o):
        for s in range(len_s):
            trans_prob_t = viterbi[:, t-1] * prob_a[:, s]
            max_trans_prob = np.max(trans_prob_t)
            viterbi[s, t] = max_trans_prob * prob_b[s, o_seq_idx[t]]
            backpointer[s, t] = np.argmax(trans_prob_t)
        print(f"viterbi[{s},{t}]={viterbi[s,t]}")

    # final step
    best_path_prob = np.max(viterbi[:, -1])
    best_last_state = np.argmax(viterbi[:, -1])

    # recover the optimal paths
    best_path = np.zeros(len_o, dtype=int)
    best_path[-1] = best_last_state
    for t in range(len_o - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]

    return best_path, best_path_prob

#---- main ----
# states
state = ['1', '2', '3']

# observation sequnces (u:up, d:down, n:unchanged)
o_seq   = ['u','u','u','d','d','n','n','d','u' ]
o_seq_dict = {'u':0, 'd':1, 'n':2}
#o_seq_idx = [value for outcode in o_seq for key, value in o_seq_dict.items() if outcode == key]
o_seq_idx = [o_seq_dict[key] for key in o_seq]

print("o_seq =",o_seq)
print("o_seq_dict =",o_seq_dict)
print("o_seq_dict.items =",o_seq_dict.items())
print("o_seq_idx =",o_seq_idx)

# initial state probability: prob_init[i]
prob_init = np.array([0.5, 0.2, 0.3])

# transition probability prob_a[i,j] : transition states i -> j
prob_a = np.array([[0.6, 0.2, 0.2], \
                   [0.5, 0.3, 0.2], \
                   [0.4, 0.1, 0.5]])

# output probability prob_b[i]
prob_b = np.array([[0.7, 0.1, 0.2], \
                   [0.1, 0.6, 0.3], \
                   [0.3, 0.3, 0.4]])

# execute viterbi algorithm
best_path, best_path_prob = viterbi_algorithm(o_seq_idx, state, prob_init, prob_a, prob_b)
best_path_states = [state[i_st] for i_st in best_path]

print(f"optimal state sequences: {best_path_states}")
print(f"optimal state sequences probability: {best_path_prob}")

#end

"""
#--- result 
o_seq = ['u', 'u', 'u', 'd', 'd', 'n', 'n', 'd', 'u']
o_seq_dict = {'u': 0, 'd': 1, 'n': 2}
o_seq_dict.items = dict_items([('u', 0), ('d', 1), ('n', 2)])
o_seq_idx = [0, 0, 0, 1, 1, 2, 2, 1, 0]
viterbi[:,0] = [0.35 0.02 0.09]
viterbi[2,1]=0.020999999999999998
viterbi[2,2]=0.00882
viterbi[2,3]=0.003704399999999999
viterbi[2,4]=0.0005556599999999998
viterbi[2,5]=0.00011113199999999997
viterbi[2,6]=2.2226399999999994e-05
viterbi[2,7]=3.333959999999999e-06
viterbi[2,8]=5.000939999999998e-07
optimal state sequences: ['1', '1', '1', '3', '3', '3', '3', '3', '1']
optimal state sequences probability: 9.335087999999998e-07
"""
