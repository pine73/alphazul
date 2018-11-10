import azul
import numpy as np
import tensorflow as tf
from copy import deepcopy

CPUCT = 2.0



class _MCTNode(object):
    """docstring for _MCTNode"""
    commands = None

    def __init__(self, game, prior, is_turn_end):
        super(_MCTNode, self).__init__()

        self.visit_count = 0

        self.game = game

        # na length vectors
        self.prior = np.squeeze(prior)
        mask = ((self.prior > 0).astype(int) - 1) *  1e8
        self.child_Ws = np.zeros(self.prior.shape) + mask
        self.child_Qs = np.zeros(self.prior.shape) + mask
        self.child_Ns = np.zeros(self.prior.shape,dtype=int)

        self.parent = None
        self.action_index = None
        self.child = {}

        self.is_turn_end = is_turn_end

        

class MCTSearch(object):
    """docstring for MCTSearch"""
    def __init__(self, game, inference_fuction, commands):
        root_game = deepcopy(game)
        self._infnet = inference_fuction

        root_value, root_prior = self._infnet(root_game)
        self._root = _MCTNode(root_game,root_prior,False)
        _MCTNode.commands = commands

    def select(self, node):
        #terminal node
        if node.is_turn_end:
            return node,[]

        #normal node
        node.visit_count += 1
        Us = CPUCT * node.prior * np.sqrt(node.visit_count) / (1 + node.child_Ns)
        action_index = np.argmax(node.child_Qs + Us)


        if not action_index in node.child.keys():
            return node,action_index
        else:
            return self.select(node.child[action_index])


    def expand_and_evaluate(self,node,action_index):
        #terminal node
        if node.is_turn_end:
            game_prime = deepcopy(node.game)
            #evaluate
            game_prime.turn_end(verbose = False)
            if game_prime.is_terminal:
                return 1.,game_prime.leading_player_num
            game_prime.start_turn()
            value, prior = self._infnet(game_prime)
            return value, game_prime.active_player_num

        #normal node
        action_command = node.commands[action_index]
        game_prime = deepcopy(node.game)
        is_turn_end = game_prime.take_command(action_command)
        #found a terminal node
        if is_turn_end:
            #expand
            child = _MCTNode(deepcopy(game_prime),[],is_turn_end)
            child.parent = node
            child.action_index = action_index
            node.child[action_index] = child
            #evaluate
            game_prime.turn_end(verbose = False)
            if game_prime.is_terminal:
                return 1.,game_prime.leading_player_num
            game_prime.start_turn()
            value, prior = self._infnet(game_prime)
            
        #found a normal node
        else:
            #evaluate
            value, prior = self._infnet(game_prime)
            #expand
            child = _MCTNode(game_prime,prior,is_turn_end)
            child.parent = node
            child.action_index = action_index
            child.visit_count += 1
            node.child[action_index] = child

        return value, game_prime.active_player_num


    def backup(self,node,value,action_index,player_num):
        if node.game.active_player_num != player_num:
            v = -value
        else:
            v = value

        if not node.is_turn_end:
            node.child_Ws[action_index] += v
            node.child_Ns[action_index] += 1
            node.child_Qs = node.child_Ws / (node.child_Ns + 1e-10)


        if node.parent is not None:
            self.backup(node.parent,value,node.action_index,player_num)


    def search(self):
        #condition check
        if self._root.is_turn_end:
            raise Exception('root turn end')

        under_node,action_index = self.select(self._root)
        value, player_num = self.expand_and_evaluate(under_node,action_index)
        self.backup(under_node,value,action_index,player_num)

    def start_search(self,num_search):
        for i in range(num_search):
            self.search()
        action_index = np.argmax(self._root.child_Ns)
        action_command = _MCTNode.commands[action_index]
        return action_command, (self._root.game.states(), action_index, self._root.game.active_player_num)

    def change_root(self,node):
        self._root = node

        

