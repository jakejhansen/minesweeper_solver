from environment import MinesweeperEnvironment


if __name__ == "__main__":

    game = MinesweeperEnvironment(10, 10, 13, True)
    #game.env.printState()

    game.new_game()
    game.env.drawState()

    while True:
        inp = input("Enter input (ROW,COL)")
        row = int(1)
        col = int(1)


    print(game.num_actions)
    print(game.legal_actions)
    print(game.state)
    action = game.act_random()
    print(action)
    state, r, d =  game.act(action)
    print(state)
    print(r)
    print(d)

    print(game.avg_reward_per_episode())
    print(game.max_reward_per_episode)
    print(game.avg_steps_per_episode())
    print(game.episode_number)

    game.new_random_game()

    state, r, d =  game.act(action)
    print(state)
    game.env.printState()
    print(r)
    print(d)



    # i = 0
    # #start = time.time()
    # while True:
    #     inp = input("Enter input (ROW,COL)")
    #     if inp == 'bb':
    #         import IPython
    #         IPython.embed()
    #     row = int(inp[1])
    #     col = int(inp[3])
    #     #v = game.action((row, col))
    #     #print("\nReward = {}".format(v["r"]))

    #     # Input is just an index into the game
    #     # We then need a random action
    #     #print(game.act_random())
    #     print(action)
        
    #     #game.env.printState()
    #     print(state)
    #     print(r)
    #     print(d)
        # if d:
        # 	game.reset()

    """
    #Test how fast it can run:
    i += 1
    print(i)
    act = [np.random.randint(0,10), np.random.randint(0,10)]
    env = game.action(act[0],act[1])
    state = stateConverter(env['state'])
    reward = env['reward']
    if i >= 1000:
        break
    """

    #print("Took: " + str(time.time()-start))

