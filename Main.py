import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from AlphaZero import MCTS
from AlphaZeroParallel import AlphaZeroParallel
from Game import make_game
from NeuralNet import ResNet
from utils import load_config


torch.manual_seed(0)


def model_test():
    game = make_game("gomoku")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = game.get_initial_state()
    state = game.get_next_state(state, 44, 1)
    encoded_state = game.get_encoded_state(state)
    tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

    model = ResNet(game, 4, 64, device=device, input_channels=game.input_channels)
    model.eval()

    policy, value = model(tensor_state)
    value = value.item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    print("value:", value)
    print("state:\n", state)

    plt.bar(range(game.action_size), policy)
    plt.title("Policy Distribution (Gomoku10)")
    plt.show()


def model_learn(config_name):
    args = load_config(f"./configs/learn/{config_name}.yaml")
    game_name = args.get("game", "gomoku")
    game = make_game(game_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 4, 64, device, input_channels=game.input_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args["device"] = str(device)
    args["game"] = game_name

    trainer = AlphaZeroParallel(model, optimizer, game, args, monitor=True)
    trainer.learn()


def model_play(version, config_name="play0", human_player=1):
    args = load_config(f"./configs/play/{config_name}.yaml")
    game = make_game(args.get("game", "gomoku"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 4, 64, device, input_channels=game.input_channels)
    model_path = f"./saved_model/model_{version}_{game.__repr__()}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mcts = MCTS(game, args, model)
    state = game.get_initial_state()
    player = 1

    while True:
        print(state)
        if player == human_player:
            valid_moves = game.get_valid_moves(state)
            print("valid:", [i for i in range(game.action_size) if valid_moves[i] == 1])
            action = int(input(f"player {player} action (0-99): "))
            if action < 0 or action >= game.action_size or valid_moves[action] == 0:
                print("invalid action")
                continue
        else:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = int(np.argmax(mcts_probs))

        state = game.get_next_state(state, action, player)
        value, is_terminal = game.get_value_and_terminated(game.change_perspective(state, player), action)
        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            elif value == -1:
                print(game.get_opponent(player), "won")
            else:
                print("draw")
            break
        player = game.get_opponent(player)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    test_parser = subparsers.add_parser("test")
    learn_parser = subparsers.add_parser("learn")
    learn_parser.add_argument("--config", type=str, default="exp0")
    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--version", type=str, default="0")
    play_parser.add_argument("--config", type=str, default="play0")
    play_parser.add_argument("--human-player", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "test":
        model_test()
    elif args.mode == "learn":
        model_learn(args.config)
    elif args.mode == "play":
        model_play(args.version, args.config, args.human_player)
