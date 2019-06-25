import numpy  as np
import pandas as pd
from operator import itemgetter
import re


def parse_State_File(State_File):

    # state information
    state_count = np.genfromtxt(State_File, int, max_rows=1)
    state_names = np.genfromtxt(State_File, str, skip_header=1, max_rows=state_count)
    state_id = np.array(range(0, state_count), int)
    states = pd.DataFrame({'id': state_id, 'state_name': state_names})
    BEGIN_index = int(states.loc[states['state_name'] == 'BEGIN', 'id'])
    END_index = int(states.loc[states['state_name'] == 'END', 'id'])

    # read transition frequencies from file and convert to dataframe
    transition_frequency = np.genfromtxt(State_File, int, skip_header=1 + state_count)
    freq = pd.DataFrame({'f1': transition_frequency[:, 0],
                         'f2': transition_frequency[:, 1],
                         'f3': transition_frequency[:, 2]})

    state_frequency = {}  # dictionary to store number of transitions from each state

    for state in state_id:
        sumOfState = 0
        for e in transition_frequency:
            if e[0] == state:
                sumOfState += e[-1]
        state_frequency[state] = sumOfState

    # transition probabilites matrix
    transition_matrix = np.zeros(shape=(state_count, state_count))

    N = state_count

    for i in range(N):
        for j in range(N):
            try:
                n_ij = int(freq.loc[(freq['f1'] == i) & (freq['f2'] == j), 'f3'])
            except KeyError:
                n_ij = 0
            except TypeError:
                n_ij = 0
            if j == BEGIN_index:
                transition_matrix[i, j] = 0
            elif i == END_index:
                transition_matrix[i, j] = 0
            else:
                n_i = state_frequency[i]
                transition_matrix[i, j] = (n_ij + 1) / (n_i + N - 1)

    return state_count, states, transition_matrix


def parse_Symbol_File(Symbol_File, state_count):
    # print("Parsing Symbol File:", Symbol_File)

    # read from symbol file:
    symbol_count = np.genfromtxt(Symbol_File, int, max_rows=1)
    symbol_names = np.genfromtxt(Symbol_File, str, skip_header=1, max_rows=symbol_count)

    # adding UNKNOWN symbol
    unk = np.array(['UNK'])
    symbol_names = np.append(symbol_names, unk, axis=0)
    symbol_count += 1
    symbol_id = np.array(range(0, symbol_count), int)

    symbols = pd.DataFrame({'id': symbol_id, 'symbol_name': symbol_names})

    # read emission frequencies from file
    emission_frequency = np.genfromtxt(Symbol_File, int, skip_header=symbol_count)


    # state_symbol_frequency
    state_symbol_frequency = dict()
    for state in range(state_count):
        emissions = emission_frequency[emission_frequency[:, 0] == state]
        f3 = emissions[:, [2]]
        n_i = np.sum(f3)
        state_symbol_frequency[state] = n_i

    # creating the emission probability matrix
    emission_matrix = np.zeros(shape=(state_count, symbol_count))

    for row in emission_frequency:
        state = row[0]
        symbol = row[1]
        freq = row[2]
        emission_matrix[state, symbol] = freq

    for state in range(state_count):
        for symbol in range(symbol_count):
            n_i = state_symbol_frequency[state]
            n_ij = emission_matrix[state, symbol]
            emission_matrix[state, symbol] = (n_ij + 1) / (n_i + symbol_count)

    # remove emission probability for begin and end
    emission_matrix[state_count - 1, symbol_count - 1] = 0
    emission_matrix[state_count - 2, symbol_count - 1] = 0


    known_symbols_count = symbol_count - 1

    return known_symbols_count, symbols, emission_matrix


def parse_Query_File(Query_File, symbols):
    num_lines = 0
    with open(Query_File, 'r') as f:
        for line in f:
            num_lines += 1
    queryFile = open(Query_File, 'r')
    lines = [line for line in queryFile.read().split('\n')]
    list = []

    for e in lines:
        #s = [i.strip() for i in re.split(r'(\W+)', e) if i.strip()]
        v = re.compile(r"([&,()/-])")
        u = v.sub(" \\1 ", e)
        s = u.split()
        l2 = []
        for e in s:
            try:
                l2.append(int(symbols.loc[symbols['symbol_name'] == e, 'id']))
            except KeyError:
                l2.append(int(symbols.loc[symbols['symbol_name'] == 'UNK', 'id']))
            except TypeError:
                l2.append(int(symbols.loc[symbols['symbol_name'] == 'UNK', 'id']))
        if l2 not in list:
            list.append(l2)
    return list


def viterbi(O, A, B, START2state, state2END, states):
    # VITERBI Algorithm:
    # Q  = Set of states
    # V  = Set of symbols
    # O  = Query, observed sequence
    # A  = state transition matrix
    # B  = emission probabilites
    # START2state = probability of initial state
    # delta  = probability matrix
    # sai = path

    mStates = A.shape[0]

    Q = np.arange(mStates)
    # print("Q",Q)
    nObs = len(O)

    # δ
    delta = np.zeros(shape=(mStates, nObs))
    # ψ
    sai = np.zeros(shape=(mStates, nObs), dtype=np.int)

    # 1. initialization:
    for q in Q:
        delta[q, 0] = np.log(START2state[q]) + np.log(B[q, O[0]])
        sai[q, 0] = q  # BEGIN

    prob_vector = list()
    path_vector = list()

    # 2. Recursion
    for t in range(1, nObs):
        for q in Q:
            for qi in Q:
                prob_vector.append(delta[qi, t - 1] + np.log(A[qi, q]) + np.log(B[q, O[t]]))
                path_vector.append(delta[qi, t - 1] + np.log(A[qi, q]))
            # update δ and ψ
            delta[q, t] = np.max(prob_vector)
            sai[q, t] = np.argmax(path_vector)

            # reset for next iteration
            prob_vector = list()
            path_vector = list()

    last_state_index = np.argmax(delta[:, - 1])  # Last state with max probability
    END_index = int(states.loc[states['state_name'] == 'END', 'id'])

    # find the path probability
    last_obs_probability = delta[last_state_index, nObs - 1]

    #    end_probability = np.log(A[last_state_index,mStates+1])
    end_probability = np.log(state2END[last_state_index])[0]

    path_probability = last_obs_probability + end_probability

    # 3. Finding the path sequence
    path = list()
    path.append(path_probability)
    path.append(END_index)
    path.append(last_state_index)

    row = last_state_index
    column = sai.shape[1] - 1

    for i in range(column, 0, -1):
        row = sai[row, column]
        column -= 1
        path.append(row)

    BEGIN_index = int(states.loc[states['state_name'] == 'BEGIN', 'id'])
    path.append(BEGIN_index)

    path.reverse()  # since it was back tracking

    return path

#--------------
# Question 1
#-------------
def viterbi_algorithm(State_File, Symbol_File, Query_File):  # do not change the heading of the function

    state_count, states, transition_matrix = parse_State_File(State_File)
    symbols_count, symbols, emission_matrix = parse_Symbol_File(Symbol_File, state_count)
    queryList = parse_Query_File(Query_File, symbols)

    index_BEGIN = int(states.loc[states['state_name'] == 'BEGIN', 'id'])
    index_END = int(states.loc[states['state_name'] == 'END', 'id'])

    A = transition_matrix

    START2state = A[index_BEGIN]  # transition probability FROM the begin state
    state2END = A[:, [A.shape[1] - 1]]  # transition probability TO end state

    B = np.delete(emission_matrix, (index_BEGIN, index_END), axis=0)

    # remove begin and end state probabilites
    a = np.delete(A, (index_BEGIN, index_END), axis=0)
    a = np.delete(a, (A.shape[1] - 1, A.shape[1] - 2), axis=1)
    likely_state_sequence_outer_list = []

    qList = []

    qList = [x for x in queryList if x != []]
    for query in range(0, len(qList)):
        likely_state_sequence = viterbi(qList[query], a, B, START2state, state2END, states)
        likely_state_sequence_outer_list.append(likely_state_sequence)

    return likely_state_sequence_outer_list


# -------------------------
# Question 2
# ------------------------

def viterbi_top_k(O, A, B, START2state, state2END, states, k):


    mStates = A.shape[0]

    nObs = len(O)

    beta = list()
    for i in range(nObs + 1):
        beta.append([])

    BEGIN_index = int(states.loc[states['state_name'] == 'BEGIN', 'id'])
    END_index = int(states.loc[states['state_name'] == 'END', 'id'])

    t_prob_list = list()
    # 1. initialization:
    for q in range(mStates):
        prob_list = (BEGIN_index, q, np.log(START2state[q]) + np.log(B[q, O[0]]))
        t_prob_list.append(prob_list)
#        # We look at the last state, and prefer the sequence with a smaller state
        t_prob_list = sorted(t_prob_list, key=lambda tup: tup[0], reverse=False)
        t_prob_list.sort(key=itemgetter(2), reverse=True)

    beta[0] = t_prob_list

    # 2. Iteratively calcualte probabiilites of the path
    for t in range(1, nObs):
        t_prob_list = list()
        n = len(beta[t - 1])
        for l in range(n):
            for state_j in range(mStates):
                # (predecessor,current state,probability )
                state_i = beta[t - 1][l][1]
                previous_probability = beta[t - 1][l][2]
                trans = np.log(A[state_i, state_j])
                emis = np.log(B[state_j, O[t]])
                probability = previous_probability + trans + emis
                pre_index = l
                t_prob_list.append((state_i, state_j, probability, pre_index))
         # We need to store top k per state.
         # We look at the last state, and prefer the sequence with a smaller state
        t_prob_list.sort(key=lambda k: (k[1], -k[2]), reverse=False)
        top_k_per_observation = list()
        for s in range(mStates):
            top_per_state = [ x for x in t_prob_list if x[1] == s ][:k]
            for row in top_per_state: 
                top_k_per_observation.append(row)
        
        beta[t] = top_k_per_observation

    # 3. Adding the END state probability
    t_prob_list = list()
    for i in range(len(beta[nObs - 1])):
        last_state = beta[nObs - 1][i]
        last_state_index = last_state[1]
        last_state_prob = last_state[2]
        end_probability = np.log(state2END[last_state_index])[0]
        path_probability = last_state_prob + end_probability
        t_prob_list.append((last_state_index, END_index, path_probability, i))

    t_prob_list.sort(key=lambda k: (k[1], -k[2]), reverse=False)

    beta[nObs] = t_prob_list

    # 4. Find the top paths
    topk_paths = list()

    for ki in range(k):
        kth_path = list()
        path_prob = beta[nObs][ki][2]
        end_state = beta[nObs][ki][1]
        pre_state = beta[nObs][ki][0]
        pre_index = beta[nObs][ki][3]
        kth_path.append(path_prob)
        kth_path.append(end_state)
        kth_path.append(pre_state)

        for t in range(nObs - 1, 0, -1):
            pre_state = beta[t][pre_index][0]
            kth_path.append(pre_state)
            pre_index = beta[t][pre_index][3]
        kth_path.append(BEGIN_index)
        kth_path.reverse()

        topk_paths.append(kth_path)

    return topk_paths


def top_k_viterbi(State_File, Symbol_File, Query_File, k):  # do not change the heading of the function
    
    state_count, states, transition_matrix = parse_State_File(State_File)
    symbols_count, symbols, emission_matrix = parse_Symbol_File(Symbol_File, state_count)
    queryList = parse_Query_File(Query_File, symbols)

    index_BEGIN = int(states.loc[states['state_name'] == 'BEGIN', 'id'])
    index_END = int(states.loc[states['state_name'] == 'END', 'id'])

    A = transition_matrix

    START2state = A[index_BEGIN]  # transition probability FROM the begin state
    state2END = A[:, [A.shape[1] - 1]]  # transition probability TO end state

    B = np.delete(emission_matrix, (index_BEGIN, index_END), axis=0)

    # remove begin and end state probabilites
    a = np.delete(A, (index_BEGIN, index_END), axis=0)
    a = np.delete(a, (A.shape[1] - 1, A.shape[1] - 2), axis=1)

    top_k_result = list()

    for query in queryList:  # [:1]:
        if len(query) > 0:
            top_k = viterbi_top_k(query, a, B, START2state, state2END, states, k)
            for row in top_k:
                top_k_result.append(row)

    return top_k_result

#----------------
# Question 3
#----------------    
def advanced_decoding(State_File, Symbol_File, Query_File):
    
    state_count, states, transition_matrix = parse_State_File(State_File)
    symbols_count, symbols, emission_matrix = parse_Symbol_File(Symbol_File, state_count)

    #add a class symbol for lvl,shop, etc
    symbols.loc[symbols.index.max()+1] = [symbols.index.max()+1,'subnumClass']

    #add a class symbol for Lotxx
    symbols.loc[symbols.index.max()+1] = [symbols.index.max()+1,'lotClass']

    #add a class symbol for commercial unit types
    symbols.loc[symbols.index.max()+1] = [symbols.index.max()+1,'CommercialUnitTypeClass']

    id_SubNumber = int(states.loc[states['state_name'] == 'SubNumber', 'id'])
    id_StreetNumber = int(states.loc[states['state_name'] == 'StreetNumber', 'id'])
    id_CommercialUnitType = int(states.loc[states['state_name'] == 'CommercialUnitType', 'id'])

    #adding a state,emission probability
    #this new symbol has highest probability from state SubNumber
    id_SubNumClass = int(symbols.loc[symbols['symbol_name'] == 'subnumClass', 'id'])

    #adding for lot
    id_LotClass = int(symbols.loc[symbols['symbol_name'] == 'lotClass', 'id'])
    
    #addring symbol for CommercialUnitTypeClass
    id_CommercialUnitTypeClass = int(symbols.loc[symbols['symbol_name'] == 'CommercialUnitTypeClass', 'id'])

    #create a column for new symbol
    new_col = np.full((state_count, 1), 1/(symbols_count+2))
    #increase the probability of seeing this symbol from state subNumber
    new_col[id_SubNumber] = 0.2
    #add to emission matrix
    B = np.hstack((emission_matrix,new_col))

    #create a column for Lot 
    new_col2 = np.full((state_count, 1), 1/(symbols_count+3))
    #increase the probability of seeing this symbol from state subNumber
    new_col2[id_StreetNumber] = 0.2
    #add to emission matrix
    B = np.hstack((B,new_col2))

    #create a column for commercial unit types 
    new_col3 = np.full((state_count, 1), 1/(symbols_count+4))
    #increase the probability of seeing this symbol from state subNumber
    new_col3[id_CommercialUnitType] = 0.2
    #add to emission matrix
    B = np.hstack((B,new_col3))    


    
    id_BEGIN = int(states.loc[states['state_name'] == 'BEGIN', 'id'])
    id_END = int(states.loc[states['state_name'] == 'END', 'id'])

    A = transition_matrix
    
    START2state = A[id_BEGIN]  # transition probability FROM the begin state
    state2END = A[:, [A.shape[1] - 1]]  # transition probability TO end state

#    factor = 2.5

#    #Heuristic rule for Entity name
#    #1. increasing the probability of address starting with entity name
#    id_EntityName = int(states.loc[states['state_name'] == 'EntityName', 'id'])
#    id_ampsand = int(states.loc[states['state_name'] == '&', 'id'])
#    START2state[id_EntityName] = START2state[id_EntityName] * factor
#    #2. increasing probability of observing an '&' after entity name
#    A[id_EntityName,id_ampsand] = A[id_EntityName,id_ampsand] *  factor
#    #
#    A[id_EntityName,id_EntityName] = A[id_EntityName,id_EntityName] *  factor
#    id_bracket = int(states.loc[states['state_name'] == '(', 'id'])
#    START2state[id_bracket] = START2state[id_bracket] * factor


    B = np.delete(B, (id_BEGIN, id_END), axis=0)

    # remove begin and end state probabilites
    a = np.delete(A, (id_BEGIN, id_END), axis=0)
    a = np.delete(a, (A.shape[1] - 1, A.shape[1] - 2), axis=1)



    queries = list()
    with open(Query_File) as query_file:
        lines = query_file.readlines()
        for i in range(len(lines)):
            tokenized_line = list(lines[i].split())
            query_line = list()
            for token in tokenized_line:
                v = re.compile(r"([&,()/-])")
                u = v.sub(" \\1 ", token)
                splits = u.split()
                for symbol in splits:
                    try:
                        symbol_id = int(symbols.loc[symbols['symbol_name'] == symbol, 'id'])
                    except TypeError:
                        if re.match(r'Shp\d*',symbol):
                            symbol_id = id_SubNumClass
                        elif re.match(r'Lvl\d*',symbol):
                            symbol_id = id_SubNumClass
                        elif re.match(r'Ste\d*',symbol):
                            symbol_id = id_SubNumClass
                        elif re.match(r'Ksk\d*',symbol):
                            symbol_id = id_SubNumClass
                        elif re.match(r'^[1-9]+\w',symbol):
                            symbol_id = id_SubNumClass
                        elif re.match(r'^Lot\d*',symbol):
                            symbol_id = id_LotClass
                        elif re.match(r'^Hang.r\w*',symbol):
                            symbol_id = id_CommercialUnitTypeClass
                        elif re.match(r'^Basement\w*',symbol):
                            symbol_id = id_CommercialUnitTypeClass
                        elif re.match(r'^Exit\w*',symbol):
                            symbol_id = id_CommercialUnitTypeClass
                        else:
                            symbol_id = int(symbols.loc[symbols['symbol_name'] == 'UNK', 'id'])
                    
                    query_line.append(symbol_id)
                    #add to list for line
            queries.append(query_line)
    query_file.close()
    
    v_result = list()

    for query in queries:
        if len(query) > 0:
            result = viterbi(query, a, B, START2state, state2END, states)
            v_result.append(result)

    return v_result

#--------------END-------------------#