import tensorflow as tf
import numpy as np
import alphazul
from copy import deepcopy
import azul
import policy



if __name__ == '__main__':
    game = azul.Azul(2)
    game.start()

    states = game.states()
    mask = game.flat_mask()

    print(states.shape[0],mask.shape[0])

    inf = alphazul.InferenceNetwork(states.shape[0],mask.shape[0])


    value, prior = inf.predict([states],[mask])
    print(prior,np.sum(prior))
    print(np.sum(prior>0),np.sum(mask))

