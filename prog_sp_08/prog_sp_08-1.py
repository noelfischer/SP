#prog_sp_08-1.py  forward algorithm ver.1
#time t starts from 0. (different from the text(t=1 in the text).)

import numpy as np

def forward_algorithm(o_seq_idx, state, prob_init, prob_a, prob_b):
    # initialization
    len_o_seq = len(o_seq_idx)
    len_state = len(state)
    alpha = np.zeros((len_o_seq, len_state))
    alpha[0, :] = prob_init * prob_b[:, o_seq_idx[0]]
    print(f"@t=0 alpha = {alpha[0]}")

    # recursive calculation
    for t in range(1, len_o_seq):
        for j in range(len_state):
            alpha[t, j] = np.sum(alpha[t-1, :] * prob_a[:, j]) * prob_b[j, o_seq_idx[t]]
        print(f"@t={t} alpha = {alpha[t]}")

    #print("alpha=\n",alpha)
    # final output probability
    return alpha[-1], np.sum(alpha[-1, :])

#----- main -----
# states
state = ['1', '2', '3']

# observation sequnces (u:up, d:down, n:unchanged)
o_seq = ['u', 'u', 'u', 'd', 'd', 'n', 'n', 'd', 'u' ]
o_seq_dict = {'u':0, 'd':1, 'n':2}
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

# execute forward algorithm
alpha_last, prob_o_seq = forward_algorithm(o_seq_idx, state, prob_init, prob_a, prob_b)

print("alpha_last =", alpha_last)
print("prob_o_seq =", prob_o_seq)

#end

""" 
#---- result
o_seq = ['u', 'u', 'u', 'd', 'd', 'n', 'n', 'd', 'u']
o_seq_dict = {'u': 0, 'd': 1, 'n': 2}
o_seq_dict.items = dict_items([('u', 0), ('d', 1), ('n', 2)])
o_seq_idx = [0, 0, 0, 1, 1, 2, 2, 1, 0]
@t=0 alpha = [0.35 0.02 0.09]
@t=1 alpha = [0.1792 0.0085 0.0357]
@t=2 alpha = [0.088235 0.004196 0.016617]
@t=3 alpha = [0.00616858 0.0123405  0.00803841]
@t=4 alpha = [0.00130868 0.00344382 0.00231631]
@t=5 alpha = [0.00068673 0.00045795 0.00084346]
@t=6 alpha = [0.00019568 0.00010772 0.00026027]
@t=7 alpha = [2.75376216e-05 5.84877769e-05 5.72442054e-05]
@t=8 alpha = [4.80649005e-05 2.87782779e-06 1.37481547e-05]
alpha_last = [4.80649005e-05 2.87782779e-06 1.37481547e-05]
prob_o_seq = 6.4690882996878e-05


"""
