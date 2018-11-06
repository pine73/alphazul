import numpy as np
import azul
import time

def random_policy(game):
    mask = game.mask(game._active_player)
    valid_commands = np.argwhere(mask == 1)
    random_index = np.random.randint(valid_commands.shape[0])
    command = valid_commands[random_index]
    return command

def slightly_less_random_policy(game):
    epsilon = 0.1
    mask = game.mask(game._active_player)
    valid_commands_without_floor = np.argwhere(mask[:,:,:5] == 1)
    valid_commands = np.argwhere(mask == 1)
    if valid_commands_without_floor.shape[0] == 0:
        valid_commands_without_floor = valid_commands
    if np.random.random() >= epsilon:
        random_index = np.random.randint(valid_commands_without_floor.shape[0])
        command = valid_commands_without_floor[random_index]
        return command
    else:
        random_index = np.random.randint(valid_commands.shape[0])
        command = valid_commands[random_index]
        return command




if __name__ == '__main__':
    a = time.time()
    game = azul.Azul(2)
    game.start()
    while True:
        game.turn_2ai(slightly_less_random_policy)
        if game.is_terminal:
            break
        game.start_turn()
    game.final_score()
    print(time.time()-a)