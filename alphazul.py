import numpy as np
import tensorflow as tf
import mcts
from multiprocessing import Process,Pipe,Queue
import azul
import time


DTYPE = tf.float32

NUM_HIDDEN = 384
NUM_LAYER = 4

STATES_SIZE = 153
MASK_SIZE = 180
REGULARIZATION_FACTOR = 1e-4
LEARNING_RATE = 1e-2
BATCH_SIZE = 32
EPOCH = 10




class InferenceNetwork(object):
    """docstring for InferenceNetwork"""
    def __init__(self, input_size, output_size, num_layer=NUM_LAYER, num_hidden = NUM_HIDDEN):
        self._graph = tf.Graph()

        with self._graph.as_default():
            with tf.name_scope('input_layer'):
                self._input_states = tf.placeholder(DTYPE, [None,input_size], 'input_states')
                self._mask = tf.placeholder(DTYPE, [None,output_size], 'mask')

            with tf.name_scope('labels'):
                self._label_value = tf.placeholder(DTYPE, [None,1], 'label_value')
                self._label_distribution = tf.placeholder(tf.int32, [None], 'label_distribution')

            with tf.name_scope('MLP'):
                layer_out = self._input_states
                for i in range(NUM_LAYER):
                    layer_out = tf.layers.dense(layer_out, NUM_HIDDEN, tf.nn.relu, name='MLP_layer_{}'.format(i))

            with tf.name_scope('value_header'):
                self._prediction_value = tf.layers.dense(layer_out, 1, tf.nn.tanh, name='value_layer')

            with tf.name_scope('distribution_header'):
                logits = tf.layers.dense(layer_out, output_size, name='logits')
                # logits_min = tf.reduce_min(logits, axis = 1)
                # masked_min = (self._mask - 1.) * logits_min
                # masked_logits = logits * self._mask - masked_min
                # masked_max = tf.reduce_max(masked_logits,axis=1)
                # self._prediction_distribution = tf.exp(logits-masked_max)*self._mask/tf.reduce_sum(tf.exp(masked_logits-masked_max)*self._mask,axis=1)

                masked_logits = logits + (self._mask - 1.) * tf.float32.max / 10
                self._prediction_distribution = tf.nn.softmax(masked_logits)

            with tf.name_scope('losses'):
                self._policy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._label_distribution, logits = logits))
                self._value_loss = tf.losses.mean_squared_error(labels=self._label_value, predictions=self._prediction_value)
                self._reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * REGULARIZATION_FACTOR
                ####################
                # self._policy_loss = self._policy_loss / 4.
                self._loss = self._policy_loss + self._value_loss + self._reg_loss


            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
                self._train_op = optimizer.minimize(self._loss)

            init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()




        self._sess = tf.Session(graph = self._graph)

        try:
            self.restore()
        except ValueError as e:
            print(e)
            self._sess.run(init)
            self.save()

        writer = tf.summary.FileWriter("./tensorboard/log/", self._sess.graph)
        writer.close()


    def predict(self, input_states, mask):
        feed_dict = {
            self._input_states:input_states,
            self._mask:mask,
            }
        value_p, distribution_p = self._sess.run([self._prediction_value,self._prediction_distribution],feed_dict=feed_dict)
        return value_p, distribution_p


    def train(self,states,acitons,values):
        feed_dict = {
            self._input_states:states,
            self._label_value:values,
            self._label_distribution:acitons
        }
        policy_loss,value_loss,reg_loss,loss,_ = self._sess.run([self._policy_loss,self._value_loss,self._reg_loss,self._loss,self._train_op],feed_dict=feed_dict)
        print('\n')
        print('policy_loss',policy_loss)
        print('value_loss:',value_loss)
        print('reg_loss:',reg_loss)
        print('loss:',loss)
        print('\n')
        self.save()


    def save(self, path="./model/latest.ckpt"):
        self._saver.save(self._sess, path)

    def restore(self, path="./model/latest.ckpt"):
        self._saver.restore(self._sess, path)

    def close(self):
        self._sess.close()



class InfHelper(object):
    """docstring for InfHelper"""
    def __init__(self, w2s_conn):
        self._w2s_conn = w2s_conn

    def __call__(self, game):
        self._w2s_conn.send((game.states(), game.flat_mask(), False))
        value, prior = self._w2s_conn.recv()
        return value, prior

class InfHelperS(object):
    """docstring for InfHelperS"""
    def __init__(self, state_size = STATES_SIZE, mask_size = MASK_SIZE):
        self._infnet = InferenceNetwork(state_size,mask_size)

    def __call__(self,game):
        states = game.states().reshape([1,-1])
        mask = game.flat_mask().reshape([1,-1])
        return self._infnet.predict(states,mask)



def worker_routine(game, w2s_conn, public_q):
    commands = np.argwhere(np.ones((6,5,6))==1)
    inf_helper = InfHelper(w2s_conn)

    search = mcts.MCTSearch(game, inf_helper, commands)

    accumulated_data = []
    winner = None
    while True:
        action_command, training_data = search.start_search(100)
        accumulated_data.append(training_data)
        is_turn_end = game.take_command(action_command)
        if is_turn_end:
            game.turn_end(verbose = False)
            if game.is_terminal:
                game.final_score()
                w2s_conn.send([True]*3)
                winner = game.leading_player_num
                break
            else:
                game.start_turn()
                if game.turn >= 9:
                    w2s_conn.send([True]*3)
                    game.final_score()
                    winner = game.leading_player_num
                    print('exceeding turn 8')
                    break
                search = mcts.MCTSearch(game, inf_helper, commands)
        else:
            search.change_root()

    state_data,action_data,value_data = [],[],[]
    for state, action_index, player in accumulated_data:
        state_data.append(state)
        action_data.append(action_index)
        if player == winner:
            value_data.append(1.)
        else:
            value_data.append(-1.)

    public_q.put((state_data,action_data,value_data))


def server_routine(s2w_conns, num_processes=8):
    infnet = InferenceNetwork(STATES_SIZE, MASK_SIZE)
    done_flags = [False] * 8
    dummy = azul.Azul(2)
    dummy.start()
    dummy_status = (dummy.states(), dummy.flat_mask())
    while True:
        if all(done_flags):
            break
        states,masks = [],[]
        for i in range(num_processes):
            if done_flags[i]:
                state, mask = dummy_status
            else:
                state, mask, flag = s2w_conns[i].recv()
                if flag == True:
                    done_flags[i] = True
                    state, mask = dummy_status
            states.append(state)
            masks.append(mask)
        states = np.stack(states, axis=0)
        masks = np.stack(masks, axis=0)
        values, priors = infnet.predict(states, masks)
        for i in range(num_processes):
            if not done_flags[i]:
                s2w_conns[i].send((values[i], priors[i]))
    infnet.close()




def self_play():
    processes = []
    s2w_conns = []
    public_q = Queue()

    # define workers
    for i in range(8):
        game = azul.Azul(2)
        game.start()
        w2s_conn, s2w_conn = Pipe()
        s2w_conns.append(s2w_conn)
        p = Process(target=worker_routine, args=(game, w2s_conn, public_q))
        processes.append(p)


    # define server
    server = Process(target=server_routine, args=(s2w_conns,))


    # start process
    server.start()
    for p in processes:
        p.start()
        

    state_data_all,action_data_all,value_data_all = [],[],[]
    for i in range(8):
        state_data,action_data,value_data = public_q.get()
        state_data_all.extend(state_data)
        action_data_all.extend(action_data)
        value_data_all.extend(value_data)

    state_data_all = np.stack(state_data_all)
    action_data_all = np.stack(action_data_all)
    value_data_all = np.stack(value_data_all).reshape((-1,1))

    assert len(state_data_all) == len(action_data_all) and len(state_data_all) == len(value_data_all)

    permutated_index = np.random.permutation(len(state_data_all))
    permutated_state = state_data_all[permutated_index]
    permutated_action = action_data_all[permutated_index]
    permutated_value = value_data_all[permutated_index]

    for p in processes:
        p.join()
    server.join()

    num_iter = len(permutated_state)//BATCH_SIZE
    infnet = InferenceNetwork(STATES_SIZE, MASK_SIZE)
    for i in range(num_iter):
        infnet.train(permutated_state[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
            permutated_action[i*BATCH_SIZE:(i+1)*BATCH_SIZE],permutated_value[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        print(i)
    infnet.close()


def debug():
    game = azul.Azul(2)
    game.start()
    commands = np.argwhere(np.ones((6,5,6))==1)
    inf_helper = InfHelperS()

    search = mcts.MCTSearch(game, inf_helper, commands)

    accumulated_data = []
    winner = None
    while True:
        action_command, training_data = search.start_search(100)
        accumulated_data.append(training_data)
        is_turn_end = game.take_command(action_command)
        if is_turn_end:
            game.turn_end(verbose = False)
            if game.is_terminal:
                game.final_score()
                winner = game.leading_player_num
                break
            else:
                game.start_turn()
                search = mcts.MCTSearch(game, inf_helper, commands)
        else:
            search.change_root()

    state_data,action_data,value_data = [],[],[]
    for state, action_index, player in accumulated_data:
        state_data.append(state)
        action_index = str(action_index//30) + str((action_index%30)//6) + str(action_index%6)
        action_data.append(action_index)
        if player == winner:
            value_data.append(1.)
        else:
            value_data.append(-1.)
    return state_data,action_data,value_data


if __name__ == '__main__':
    self_play()

    # state_data,action_data,value_data = debug()

    # print(state_data)
    # print(action_data)