import tensorflow as tf
import numpy as np
import alphazul
from copy import deepcopy
import azul
import policy
import mcts




if __name__ == '__main__':
    game = azul.Azul(2)
    game.start()
    # states = game.states()
    # mask = game.flat_mask()
    # inf = alphazul.InferenceNetwork(states.shape[0],mask.shape[0])
    # value, prior = inf.predict([states],[mask])
    # print(prior,np.sum(prior))
    # print(np.sum(prior>0),np.sum(mask))


    search = mcts.MCTSearch(game, policy.rollout, commands = np.argwhere(np.ones((6,5,6))==1))
    while True:

        # game.display()
        # print('------------------------------------')
        # search._root.game.display()
        # print('\n\n')


        action,_ = search.start_search(100)
        is_turn_end = game.take_command(action)
        if is_turn_end:
            game.turn_end(verbose = False)
            break
        else:
            search.change_root()





    # while True:
    #     game = azul.Azul(2)
    #     game.start()
    #     search = mcts.MCTSearch(game, policy.rollout, commands = np.argwhere(np.ones((6,5,6))==1))
    #     action,(_,index,_) = search.start_search(100)
