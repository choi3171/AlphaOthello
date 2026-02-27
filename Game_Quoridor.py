import numpy as np


class Gomoku10:
    def __init__(self):
        self.row_count = 10
        self.column_count = 10
        self.action_size = self.row_count * self.column_count
        self.input_channels = 3
        self.encoded_state_size = self.input_channels * self.row_count * self.column_count
        self.interval = 16
        self.game_name = "gomoku"

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

    def reshape_encoded_state(self, flat):
        return flat.reshape(self.input_channels, self.row_count, self.column_count)

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

    def final_state_size(self):
        return self.row_count * self.column_count

    def decode_final_state(self, raw):
        return np.frombuffer(raw, dtype=np.int8).copy().reshape(self.row_count, self.column_count)


class Quoridor7:
    def __init__(self):
        self.row_count = 7
        self.column_count = 7
        self.action_size = 77
        self.input_channels = 6
        self.encoded_state_size = self.input_channels * self.row_count * self.column_count
        self.interval = 20
        self.game_name = "quoridor7"

    def __repr__(self):
        return "Quoridor7"

    def get_visualized_state(self, state):
        if isinstance(state, dict):
            board = np.asarray(state.get("board"), dtype=np.int8)
            walls_h = np.asarray(state.get("walls_h"), dtype=np.int8)
            walls_v = np.asarray(state.get("walls_v"), dtype=np.int8)
        else:
            board = np.asarray(state, dtype=np.int8)
            walls_h = np.zeros((self.row_count + 1, self.column_count + 1), dtype=np.int8)
            walls_v = np.zeros((self.row_count + 1, self.column_count + 1), dtype=np.int8)

        size = self.interval
        height = self.row_count * size + 1
        width = self.column_count * size + 1
        visualized_state = np.zeros((3, height, width), dtype=np.float32)

        # board background
        visualized_state[0].fill(0.95)
        visualized_state[1].fill(0.86)
        visualized_state[2].fill(0.70)

        # grid lines
        grid_color = 0.25
        for r in range(self.row_count + 1):
            y = r * size
            y1 = min(height, y + 1)
            visualized_state[:, y:y1, :] = grid_color
        for c in range(self.column_count + 1):
            x = c * size
            x1 = min(width, x + 1)
            visualized_state[:, :, x:x1] = grid_color

        # walls from walls_h / walls_v (black thick line)
        wall_thickness = max(2, size // 4)
        for wr in range(walls_h.shape[0]):
            for wc in range(walls_h.shape[1]):
                if walls_h[wr, wc] <= 0:
                    continue
                y = wr * size
                x0 = (wc - 1) * size
                x1 = (wc + 1) * size
                yy0 = max(0, y - wall_thickness // 2)
                yy1 = min(height, y + wall_thickness // 2 + 1)
                xx0 = max(0, x0)
                xx1 = min(width, x1 + 1)
                if yy0 < yy1 and xx0 < xx1:
                    visualized_state[:, yy0:yy1, xx0:xx1] = 0.0

        for wr in range(walls_v.shape[0]):
            for wc in range(walls_v.shape[1]):
                if walls_v[wr, wc] <= 0:
                    continue
                x = wc * size
                y0 = (wr - 1) * size
                y1 = (wr + 1) * size
                xx0 = max(0, x - wall_thickness // 2)
                xx1 = min(width, x + wall_thickness // 2 + 1)
                yy0 = max(0, y0)
                yy1 = min(height, y1 + 1)
                if yy0 < yy1 and xx0 < xx1:
                    visualized_state[:, yy0:yy1, xx0:xx1] = 0.0

        # pawns
        radius = max(2, size // 3)
        rr = radius * radius
        for r in range(self.row_count):
            for c in range(self.column_count):
                val = int(board[r, c])
                if val == 0:
                    continue
                cy = r * size + size // 2
                cx = c * size + size // 2
                for dy in range(-radius, radius + 1):
                    yy = cy + dy
                    if yy < 0 or yy >= height:
                        continue
                    for dx in range(-radius, radius + 1):
                        if dx * dx + dy * dy > rr:
                            continue
                        xx = cx + dx
                        if xx < 0 or xx >= width:
                            continue
                        if val > 0:
                            visualized_state[0, yy, xx] = 0.10
                            visualized_state[1, yy, xx] = 0.30
                            visualized_state[2, yy, xx] = 0.95
                        else:
                            visualized_state[0, yy, xx] = 0.92
                            visualized_state[1, yy, xx] = 0.20
                            visualized_state[2, yy, xx] = 0.15
        return visualized_state

    def final_state_size(self):
        wall_area = (self.row_count + 1) * (self.column_count + 1)
        return self.row_count * self.column_count + 2 * wall_area

    def decode_final_state(self, raw):
        area = self.row_count * self.column_count
        wall_area = (self.row_count + 1) * (self.column_count + 1)
        arr = np.frombuffer(raw, dtype=np.int8).copy()
        if arr.size == area:
            # backward compatibility: old format had only board plane
            board = arr.reshape(self.row_count, self.column_count)
            walls_h = np.zeros((self.row_count + 1, self.column_count + 1), dtype=np.int8)
            walls_v = np.zeros((self.row_count + 1, self.column_count + 1), dtype=np.int8)
            return {"board": board, "walls_h": walls_h, "walls_v": walls_v}
        expected = area + 2 * wall_area
        if arr.size != expected:
            raise ValueError(f"Unexpected quoridor final_state size: got {arr.size}, expected {expected}")
        board = arr[:area].reshape(self.row_count, self.column_count)
        walls_h = arr[area : area + wall_area].reshape(self.row_count + 1, self.column_count + 1)
        walls_v = arr[area + wall_area : area + 2 * wall_area].reshape(
            self.row_count + 1, self.column_count + 1
        )
        return {"board": board, "walls_h": walls_h, "walls_v": walls_v}

    def _precompute_move_masks(self):
        masks = []
        for i in range(self.num_squares):
            r, c = divmod(i, self.size)
            m = {}
            if r > 0: m['U'] = 1 << (i - self.size)
            if r < self.size - 1: m['D'] = 1 << (i + self.size)
            if c > 0: m['L'] = 1 << (i - 1)
            if c < self.size - 1: m['R'] = 1 << (i + 1)
            masks.append(m)
        return masks

    def _precompute_h_wall_masks(self):
        blocks = {}
        for r in range(self.size - 1):
            for c in range(self.size - 1):
                wall_bit = 1 << (r * (self.size - 1) + c)
                for col_off in [0, 1]:
                    u = r * self.size + c + col_off
                    d = (r + 1) * self.size + c + col_off
                    blocks[(u, d)] = blocks.get((u, d), 0) | wall_bit
                    blocks[(d, u)] = blocks.get((d, u), 0) | wall_bit
        return blocks

    def _precompute_v_wall_masks(self):
        blocks = {}
        for r in range(self.size - 1):
            for c in range(self.size - 1):
                wall_bit = 1 << (r * (self.size - 1) + c)
                for row_off in [0, 1]:
                    l = (r + row_off) * self.size + c
                    r_idx = (r + row_off) * self.size + c + 1
                    blocks[(l, r_idx)] = blocks.get((l, r_idx), 0) | wall_bit
                    blocks[(r_idx, l)] = blocks.get((r_idx, l), 0) | wall_bit
        return blocks

    def get_initial_state(self):
        return {
            'p_bits': [1 << (self.size // 2), 1 << (self.num_squares - 1 - self.size // 2)],
            'walls_h': 0,
            'walls_v': 0,
            'walls_left': np.array([self.initial_walls_left, self.initial_walls_left])
        }

    def is_blocked(self, state, f_idx, t_idx):
        if self.h_wall_masks.get((f_idx, t_idx), 0) & state['walls_h']: return True
        if self.v_wall_masks.get((f_idx, t_idx), 0) & state['walls_v']: return True
        return False

    def has_path(self, state, player_idx):
        reachable = state['p_bits'][player_idx]
        goal_mask = self.goal_masks[player_idx]
        visited = reachable

        while True:
            temp = reachable
            reachable = 0
            
            while temp:
                curr_bit = temp & -temp
                temp ^= curr_bit
                curr_idx = curr_bit.bit_length() - 1
                
                for _, target_bit in self.move_masks[curr_idx].items():
                    target_idx = target_bit.bit_length() - 1
                    if not (visited & target_bit) and not self.is_blocked(state, curr_idx, target_idx):
                        reachable |= target_bit
            
            if reachable & goal_mask: return True
            if reachable == 0: return False
            
            visited |= reachable
            temp = reachable

    def get_valid_moves(self, state, player):
        p_idx = 0 if player == 1 else 1
        opp_idx = 1 - p_idx
        valid_moves = np.zeros(self.action_size, dtype=int)
        
        curr_bit = state['p_bits'][p_idx]
        curr_idx = curr_bit.bit_length() - 1
        opp_bit = state['p_bits'][opp_idx]
        opp_idx_pos = opp_bit.bit_length() - 1
        
        for direction, target_bit in self.move_masks[curr_idx].items():
            target_idx = target_bit.bit_length() - 1
            
            if not self.is_blocked(state, curr_idx, target_idx):
                if target_bit != opp_bit:
                    valid_moves[target_idx] = 1
                else:
                    can_straight_jump = False
                    if direction in self.move_masks[target_idx]:
                        jump_bit = self.move_masks[target_idx][direction]
                        jump_idx = jump_bit.bit_length() - 1
                        if not self.is_blocked(state, target_idx, jump_idx):
                            valid_moves[jump_idx] = 1
                            can_straight_jump = True
                    
                    if not can_straight_jump:
                        for diag_dir, diag_bit in self.move_masks[target_idx].items():
                            if diag_dir != direction:
                                diag_idx = diag_bit.bit_length() - 1
                                if not self.is_blocked(state, target_idx, diag_idx):
                                     valid_moves[diag_idx] = 1

        if state['walls_left'][p_idx] > 0:
            for r in range(self.size - 1):
                for c in range(self.size - 1):
                    wall_bit = 1 << (r * (self.size - 1) + c)
                    
                    if not (state['walls_h'] & wall_bit) and not (state['walls_v'] & wall_bit):
                        is_overlap = False
                        if c > 0 and (state['walls_h'] & (wall_bit >> 1)): is_overlap = True
                        if c < self.size - 2 and (state['walls_h'] & (wall_bit << 1)): is_overlap = True
                        
                        if not is_overlap:
                            state['walls_h'] |= wall_bit
                            if self.has_path(state, 0) and self.has_path(state, 1):
                                valid_moves[self.piece_action_size + r * (self.size-1) + c] = 1
                            state['walls_h'] &= ~wall_bit

                    if not (state['walls_v'] & wall_bit) and not (state['walls_h'] & wall_bit):
                        is_overlap = False
                        shift = self.size - 1
                        if r > 0 and (state['walls_v'] & (wall_bit >> shift)): is_overlap = True
                        if r < self.size - 2 and (state['walls_v'] & (wall_bit << shift)): is_overlap = True
                        
                        if not is_overlap:
                            state['walls_v'] |= wall_bit
                            if self.has_path(state, 0) and self.has_path(state, 1):
                                valid_moves[self.piece_action_size + self.walls_action_size + r * (self.size-1) + c] = 1
                            state['walls_v'] &= ~wall_bit

        return valid_moves

    def get_next_state(self, state, action_idx, player):
        p_idx = 0 if player == 1 else 1
        next_state = {
            'p_bits': state['p_bits'][:],
            'walls_h': state['walls_h'],
            'walls_v': state['walls_v'],
            'walls_left': state['walls_left'].copy()
        }
        
        if action_idx < self.piece_action_size:
            next_state['p_bits'][p_idx] = 1 << action_idx
        elif action_idx < self.piece_action_size + self.walls_action_size:
            idx = action_idx - self.piece_action_size
            next_state['walls_h'] |= (1 << idx)
            next_state['walls_left'][p_idx] -= 1
        else:
            idx = action_idx - self.piece_action_size - self.walls_action_size
            next_state['walls_v'] |= (1 << idx)
            next_state['walls_left'][p_idx] -= 1
            
        return next_state

    def check_win(self, state, player):
        p_idx = 0 if player == 1 else 1
        return bool(state['p_bits'][p_idx] & self.goal_masks[p_idx])

    def get_value_and_terminated(self, state, player):
        if self.check_win(state, player):
            return 1, True
        if self.check_win(state, -player):
            return -1, True
        if np.sum(self.get_valid_moves(state, player)) == 0:
            return 0, True
            
        return 0, False

    def is_blocked(self, state, f_idx, t_idx):
        if self.h_wall_masks.get((f_idx, t_idx), 0) & state['walls_h']: return True
        if self.v_wall_masks.get((f_idx, t_idx), 0) & state['walls_v']: return True
        return False

    def has_path(self, state, player_idx):
        reachable = state['p_bits'][player_idx]
        goal_mask = self.goal_masks[player_idx]
        
        while True:
            next_reachable = reachable
            temp = reachable
            while temp:
                curr_bit = temp & -temp
                curr_idx = curr_bit.bit_length() - 1
                for _, target_bit in self.move_masks[curr_idx].items():
                    if not self.is_blocked(state, curr_idx, target_bit.bit_length() - 1):
                        next_reachable |= target_bit
                temp &= temp - 1
            
            if next_reachable & goal_mask: return True
            if next_reachable == reachable: return False
            reachable = next_reachable

    def change_perspective(self, state, player):
            if player == 1:
                return state
            
            def flip_pos_bit(bit):
                if bit == 0: return 0
                idx = bit.bit_length() - 1
                new_idx = (self.num_squares - 1) - idx
                return 1 << new_idx

            new_p1 = flip_pos_bit(state['p_bits'][1])
            new_p2 = flip_pos_bit(state['p_bits'][0])

            wall_grid_size = (self.size - 1) ** 2
            
            def flip_wall_bits(walls_int):
                new_walls = 0
                temp = walls_int
                while temp:
                    bit = temp & -temp
                    idx = bit.bit_length() - 1
                    new_idx = (wall_grid_size - 1) - idx
                    new_walls |= (1 << new_idx)
                    temp &= temp - 1
                return new_walls

            new_wh = flip_wall_bits(state['walls_h'])
            new_wv = flip_wall_bits(state['walls_v'])

            return {
                'p_bits': [new_p1, new_p2],
                'walls_h': new_wh,
                'walls_v': new_wv,
                'walls_left': np.array([state['walls_left'][1], state['walls_left'][0]])
            }

    def get_opponent_value(self, value):
        return -value

    def get_opponent(self, player):
        return -player

    def get_encoded_state(self, state):
        encoded_state = np.zeros((6, self.size, self.size), dtype=np.float32)
        
        p1_bit = state['p_bits'][0]
        if p1_bit > 0:
            idx = p1_bit.bit_length() - 1
            r, c = divmod(idx, self.size)
            encoded_state[0, r, c] = 1.0

        p2_bit = state['p_bits'][1]
        if p2_bit > 0:
            idx = p2_bit.bit_length() - 1
            r, c = divmod(idx, self.size)
            encoded_state[1, r, c] = 1.0

        wall_size = self.size - 1
        
        wh = state['walls_h']
        for i in range(self.walls_action_size):
            if (wh >> i) & 1:
                r, c = divmod(i, wall_size)
                encoded_state[2, r, c] = 1.0

        wv = state['walls_v']
        for i in range(self.walls_action_size):
            if (wv >> i) & 1:
                r, c = divmod(i, wall_size)
                encoded_state[3, r, c] = 1.0

        p1_wall_ratio = state['walls_left'][0] / self.initial_walls_left
        p2_wall_ratio = state['walls_left'][1] / self.initial_walls_left
        
        encoded_state[4].fill(p1_wall_ratio)
        encoded_state[5].fill(p2_wall_ratio)
        
        return encoded_state

class Quoridor9(Quoridor7):
    def __init__(self):
        self.row_count = 9
        self.column_count = 9
        self.action_size = 133
        self.input_channels = 6
        self.encoded_state_size = self.input_channels * self.row_count * self.column_count
        self.interval = 16
        self.game_name = "quoridor9"

    def __repr__(self):
        return "Quoridor9"

class Quoridor5(Quoridor7):
    def __init__(self):
        self.row_count = 5
        self.column_count = 5
        self.action_size = 37
        self.input_channels = 6
        self.encoded_state_size = self.input_channels * self.row_count * self.column_count
        self.interval = 24
        self.game_name = "quoridor5"

    def __repr__(self):
        return "Quoridor5"

def make_game(game_name: str):
    normalized = str(game_name).strip().lower()
    if normalized in ("gomoku", "gomoku10"):
        return Gomoku10()
    if normalized in ("quoridor5", "quoridor_5"):
        return Quoridor5()
    if normalized in ("quoridor", "quoridor7"):
        return Quoridor7()
    if normalized in ("quoridor9", "quoridor_9"):
        return Quoridor9()
    raise ValueError(f"Unsupported game: {game_name}")