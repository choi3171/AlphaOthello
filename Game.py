import numpy as np


class Gomoku10:
    def __init__(self):
        self.row_count = 10
        self.column_count = 10
        self.action_size = self.row_count * self.column_count
        self.interval = 16

    def __repr__(self):
        return "Gomoku10"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=np.int8)

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def _count_in_direction(self, state, r, c, dr, dc, player):
        count = 1

        nr, nc = r + dr, c + dc
        while 0 <= nr < self.row_count and 0 <= nc < self.column_count and state[nr, nc] == player:
            count += 1
            nr += dr
            nc += dc

        nr, nc = r - dr, c - dc
        while 0 <= nr < self.row_count and 0 <= nc < self.column_count and state[nr, nc] == player:
            count += 1
            nr -= dr
            nc -= dc

        return count

    def check_win(self, state, action):
        if action is None:
            return False

        row = action // self.column_count
        col = action % self.column_count
        player = state[row, col]
        if player == 0:
            return False

        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            if self._count_in_direction(state, row, col, dr, dc, player) >= 5:
                return True
        return False

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(state == 0) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack((state == -1, state == 0, state == 1)).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        return encoded_state

    def get_visualized_state(self, state):
        size = self.interval
        visualized_state = np.zeros((3, self.row_count * size, self.column_count * size), dtype=np.float32)

        for i in range(self.row_count):
            for j in range(self.column_count):
                r0, r1 = i * size, (i + 1) * size
                c0, c1 = j * size, (j + 1) * size

                if state[i, j] == 1:
                    visualized_state[:, r0:r1, c0:c1] = 0.0
                elif state[i, j] == -1:
                    visualized_state[:, r0:r1, c0:c1] = 1.0
                else:
                    visualized_state[1, r0:r1, c0:c1] = 1.0

        return visualized_state
