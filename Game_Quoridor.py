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
        self._init_quoridor(size=7, initial_walls_left=5, interval=20, game_name="quoridor7")

    def __repr__(self):
        return "Quoridor7"

    def _init_quoridor(self, size, initial_walls_left, interval, game_name):
        self.size = int(size)
        self.row_count = self.size
        self.column_count = self.size
        self.wall_size = self.size + 1
        self.inner_wall = self.size - 1
        self.inner_wall_cnt = self.inner_wall * self.inner_wall
        self.wall_cnt = self.wall_size * self.wall_size
        self.action_pass = 4 + 2 * self.inner_wall_cnt
        self.action_size = self.action_pass + 1
        self.input_channels = 6
        self.encoded_state_size = self.input_channels * self.row_count * self.column_count
        self.interval = int(interval)
        self.game_name = game_name

        self.initial_walls_left = int(initial_walls_left)
        self.piece_action_size = 4
        self.walls_action_size = self.inner_wall_cnt

        self._dr = (-1, 1, 0, 0)
        self._dc = (0, 0, -1, 1)
        self._flip_dir = {0: 1, 1: 0, 2: 3, 3: 2}

        self._goal_rows = (self.size - 1, 0)
        self._valid_wall_mask = 0
        for r in range(1, self.size):
            for c in range(1, self.size):
                self._valid_wall_mask |= self._bit(r * self.wall_size + c)

        self._h_expand_lut = [0] * self.wall_cnt
        self._v_expand_lut = [0] * self.wall_cnt
        for wr in range(1, self.size):
            for wc in range(1, self.size):
                idx = wr * self.wall_size + wc
                self._h_expand_lut[idx] = self._bit((wr - 1) * self.size + (wc - 1)) | self._bit(
                    (wr - 1) * self.size + wc
                )
                self._v_expand_lut[idx] = self._bit((wr - 1) * self.size + (wc - 1)) | self._bit(
                    wr * self.size + (wc - 1)
                )

        edge_h = 0
        edge_v = 0
        for i in range(self.wall_size):
            edge_h |= self._bit(i)
            edge_h |= self._bit(self.size * self.wall_size + i)
            edge_v |= self._bit(i * self.wall_size)
            edge_v |= self._bit(i * self.wall_size + self.size)
        self._edge_walls_h = edge_h
        self._edge_walls_v = edge_v

    @staticmethod
    def _bit(idx):
        return 1 << int(idx)

    @staticmethod
    def _lsb_index(v):
        return (int(v & -v).bit_length() - 1) if v else -1

    def _compact_to_wall_idx(self, compact_idx):
        r = 1 + (compact_idx // self.inner_wall)
        c = 1 + (compact_idx % self.inner_wall)
        return r * self.wall_size + c

    def _wall_idx_to_compact(self, wall_idx):
        r = wall_idx // self.wall_size
        c = wall_idx % self.wall_size
        return (r - 1) * self.inner_wall + (c - 1)

    def _flip_bits(self, val, max_bits):
        res = 0
        x = int(val)
        while x:
            bit = x & -x
            idx = bit.bit_length() - 1
            res |= self._bit((max_bits - 1) - idx)
            x ^= bit
        return res

    def _copy_state(self, state):
        return {
            "p_bits": [int(state["p_bits"][0]), int(state["p_bits"][1])],
            "walls_h": int(state["walls_h"]),
            "walls_v": int(state["walls_v"]),
            "h_block": int(state["h_block"]),
            "v_block": int(state["v_block"]),
            "walls_left": np.array(state["walls_left"], dtype=np.int8).copy(),
            "turn": int(state["turn"]),
            "is_jumping": bool(state["is_jumping"]),
            "jumper_idx": int(state["jumper_idx"]),
            "jump_dir": int(state["jump_dir"]),
        }

    def _board_from_state(self, state):
        board = np.zeros((self.size, self.size), dtype=np.int8)
        if not isinstance(state, dict):
            return np.asarray(state, dtype=np.int8)
        if "board" in state:
            return np.asarray(state["board"], dtype=np.int8)
        p0 = self._lsb_index(int(state.get("p_bits", [0, 0])[0]))
        p1 = self._lsb_index(int(state.get("p_bits", [0, 0])[1]))
        if p0 >= 0:
            board[p0 // self.size, p0 % self.size] = 1
        if p1 >= 0:
            board[p1 // self.size, p1 % self.size] = -1
        return board

    def _walls_grid_from_bits(self, walls):
        grid = np.zeros((self.wall_size, self.wall_size), dtype=np.int8)
        x = int(walls)
        while x:
            bit = x & -x
            idx = bit.bit_length() - 1
            grid[idx // self.wall_size, idx % self.wall_size] = 1
            x ^= bit
        return grid

    def _recompute_blocks_from_walls(self, state):
        h_block = 0
        v_block = 0
        wh = int(state["walls_h"])
        while wh:
            bit = wh & -wh
            idx = bit.bit_length() - 1
            if 1 <= idx // self.wall_size <= self.size - 1 and 1 <= idx % self.wall_size <= self.size - 1:
                h_block |= self._h_expand_lut[idx]
            wh ^= bit
        wv = int(state["walls_v"])
        while wv:
            bit = wv & -wv
            idx = bit.bit_length() - 1
            if 1 <= idx // self.wall_size <= self.size - 1 and 1 <= idx % self.wall_size <= self.size - 1:
                v_block |= self._v_expand_lut[idx]
            wv ^= bit
        state["h_block"] = h_block
        state["v_block"] = v_block

    def _is_move_blocked(self, state, from_idx, to_idx):
        r1, c1 = divmod(from_idx, self.size)
        r2, c2 = divmod(to_idx, self.size)
        if r1 == r2:
            wc = max(c1, c2)
            w1 = self._bit(r1 * self.wall_size + wc)
            w2 = self._bit((r1 + 1) * self.wall_size + wc)
            return (int(state["walls_v"]) & (w1 | w2)) != 0
        wr = max(r1, r2)
        w1 = self._bit(wr * self.wall_size + c1)
        w2 = self._bit(wr * self.wall_size + (c1 + 1))
        return (int(state["walls_h"]) & (w1 | w2)) != 0

    def _has_path(self, state, p_idx):
        start = self._lsb_index(int(state["p_bits"][p_idx]))
        if start < 0:
            return False
        goal_row = self._goal_rows[p_idx]
        q = [start]
        head = 0
        visited = self._bit(start)
        while head < len(q):
            curr = q[head]
            head += 1
            r, c = divmod(curr, self.size)
            if r == goal_row:
                return True
            for d in range(4):
                nr = r + self._dr[d]
                nc = c + self._dc[d]
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    continue
                nxt = nr * self.size + nc
                nxt_bit = self._bit(nxt)
                if visited & nxt_bit:
                    continue
                if self._is_move_blocked(state, curr, nxt):
                    continue
                visited |= nxt_bit
                q.append(nxt)
        return False

    def action_from_canonical(self, action, player):
        if player != -1:
            return action
        if action < 0 or action >= self.action_size:
            return action
        if action == self.action_pass:
            return action
        if action < 4:
            return (1, 0, 3, 2)[action]
        if action < 4 + self.inner_wall_cnt:
            compact = action - 4
            r = compact // self.inner_wall
            c = compact % self.inner_wall
            rotated = (self.inner_wall - 1 - r) * self.inner_wall + (self.inner_wall - 1 - c)
            return 4 + rotated
        compact = action - (4 + self.inner_wall_cnt)
        r = compact // self.inner_wall
        c = compact % self.inner_wall
        rotated = (self.inner_wall - 1 - r) * self.inner_wall + (self.inner_wall - 1 - c)
        return 4 + self.inner_wall_cnt + rotated

    def reshape_encoded_state(self, flat):
        return flat.reshape(self.input_channels, self.row_count, self.column_count)

    def get_encoded_state(self, state):
        encoded_state = np.zeros(
            (self.input_channels, self.row_count, self.column_count), dtype=np.float32
        )
        if not isinstance(state, dict):
            board = np.asarray(state, dtype=np.int8)
            encoded_state[0] = (board == 1).astype(np.float32)
            encoded_state[1] = (board == -1).astype(np.float32)
            encoded_state[4].fill(1.0)
            encoded_state[5].fill(1.0)
            return encoded_state

        p0 = self._lsb_index(int(state["p_bits"][0]))
        p1 = self._lsb_index(int(state["p_bits"][1]))
        if p0 >= 0:
            encoded_state[0, p0 // self.size, p0 % self.size] = 1.0
        if p1 >= 0:
            encoded_state[1, p1 // self.size, p1 % self.size] = 1.0

        hb = int(state["h_block"])
        while hb:
            bit = hb & -hb
            idx = bit.bit_length() - 1
            encoded_state[2, idx // self.size, idx % self.size] = 1.0
            hb ^= bit

        vb = int(state["v_block"])
        while vb:
            bit = vb & -vb
            idx = bit.bit_length() - 1
            encoded_state[3, idx // self.size, idx % self.size] = 1.0
            vb ^= bit

        p1_ratio = float(state["walls_left"][0]) / float(self.initial_walls_left)
        p2_ratio = float(state["walls_left"][1]) / float(self.initial_walls_left)
        encoded_state[4].fill(p1_ratio)
        encoded_state[5].fill(p2_ratio)
        return encoded_state

    def get_visualized_state(self, state):
        if isinstance(state, dict):
            board = self._board_from_state(state)
            if "walls_h" in state and isinstance(state["walls_h"], np.ndarray):
                walls_h = np.asarray(state.get("walls_h"), dtype=np.int8)
                walls_v = np.asarray(state.get("walls_v"), dtype=np.int8)
            else:
                walls_h = self._walls_grid_from_bits(int(state.get("walls_h", 0)))
                walls_v = self._walls_grid_from_bits(int(state.get("walls_v", 0)))
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

    def get_initial_state(self):
        state = {
            "p_bits": [self._bit(self.size // 2), self._bit(self.size * (self.size - 1) + self.size // 2)],
            "walls_h": self._edge_walls_h,
            "walls_v": self._edge_walls_v,
            "h_block": 0,
            "v_block": 0,
            "walls_left": np.array([self.initial_walls_left, self.initial_walls_left], dtype=np.int8),
            "turn": 0,
            "is_jumping": False,
            "jumper_idx": -1,
            "jump_dir": -1,
        }
        self._recompute_blocks_from_walls(state)
        return state

    def get_next_state(self, state, action, player):
        next_state = self._copy_state(state)
        next_state["turn"] = 0 if player == 1 else 1

        action_idx = int(action)
        if action_idx >= self.action_pass:
            next_state["turn"] = 1 - next_state["turn"]
            return next_state

        turn = next_state["turn"]
        if action_idx < 4:
            from_idx = self._lsb_index(next_state["p_bits"][turn])
            if from_idx < 0:
                next_state["turn"] = 1 - next_state["turn"]
                return next_state
            r, c = divmod(from_idx, self.size)
            to_idx = (r + self._dr[action_idx]) * self.size + (c + self._dc[action_idx])

            if next_state["is_jumping"]:
                next_state["p_bits"][turn] = self._bit(to_idx)
                next_state["is_jumping"] = False
                next_state["jumper_idx"] = -1
                next_state["jump_dir"] = -1
            else:
                opp_idx = self._lsb_index(next_state["p_bits"][1 - turn])
                if to_idx == opp_idx:
                    next_state["p_bits"][turn] = self._bit(to_idx)
                    next_state["is_jumping"] = True
                    next_state["jumper_idx"] = turn
                    next_state["jump_dir"] = action_idx
                else:
                    next_state["p_bits"][turn] = self._bit(to_idx)
        elif action_idx < 4 + self.inner_wall_cnt:
            compact_idx = action_idx - 4
            wall_idx = self._compact_to_wall_idx(compact_idx)
            next_state["walls_h"] |= self._bit(wall_idx)
            next_state["h_block"] |= self._h_expand_lut[wall_idx]
            next_state["walls_left"][turn] -= 1
        else:
            compact_idx = action_idx - (4 + self.inner_wall_cnt)
            wall_idx = self._compact_to_wall_idx(compact_idx)
            next_state["walls_v"] |= self._bit(wall_idx)
            next_state["v_block"] |= self._v_expand_lut[wall_idx]
            next_state["walls_left"][turn] -= 1

        next_state["turn"] = 1 - next_state["turn"]
        return next_state

    def get_valid_moves(self, state):
        valid = np.zeros(self.action_size, dtype=np.uint8)
        p_idx = int(state["turn"])
        opp_idx = 1 - p_idx

        if state["is_jumping"]:
            if p_idx != int(state["jumper_idx"]):
                valid[self.action_pass] = 1
                return valid
            curr_idx = self._lsb_index(int(state["p_bits"][p_idx]))
            if curr_idx < 0:
                return valid
            r, c = divmod(curr_idx, self.size)
            direction = int(state["jump_dir"])
            for d in (direction, direction ^ 2, direction ^ 3):
                nr = r + self._dr[d]
                nc = c + self._dc[d]
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    continue
                to_idx = nr * self.size + nc
                if not self._is_move_blocked(state, curr_idx, to_idx):
                    valid[d] = 1
                    if d == direction:
                        return valid
            return valid

        curr_idx = self._lsb_index(int(state["p_bits"][p_idx]))
        opp_pos = self._lsb_index(int(state["p_bits"][opp_idx]))
        if curr_idx < 0 or opp_pos < 0:
            return valid
        r, c = divmod(curr_idx, self.size)

        for i in range(4):
            nr = r + self._dr[i]
            nc = c + self._dc[i]
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue
            target_idx = nr * self.size + nc
            if self._is_move_blocked(state, curr_idx, target_idx):
                continue
            if target_idx == opp_pos:
                escapes = 0
                for j in range(4):
                    if j == (i ^ 1):
                        continue
                    br = nr + self._dr[j]
                    bc = nc + self._dc[j]
                    if not (0 <= br < self.size and 0 <= bc < self.size):
                        continue
                    if not self._is_move_blocked(state, target_idx, br * self.size + bc):
                        escapes += 1
                if escapes == 0:
                    continue
            valid[i] = 1

        if int(state["walls_left"][p_idx]) <= 0:
            return valid

        occupied = int(state["walls_h"]) | int(state["walls_v"])
        candidates = (~occupied) & self._valid_wall_mask
        while candidates:
            bit = candidates & -candidates
            idx = bit.bit_length() - 1
            candidates ^= bit
            wall_bit = self._bit(idx)

            overlap_h = bool((int(state["walls_h"]) & (wall_bit >> 1)) or (int(state["walls_h"]) & (wall_bit << 1)))
            if not overlap_h:
                test_state = self._copy_state(state)
                test_state["walls_h"] |= wall_bit
                test_state["h_block"] |= self._h_expand_lut[idx]
                if self._has_path(test_state, 0) and self._has_path(test_state, 1):
                    valid[4 + self._wall_idx_to_compact(idx)] = 1

            overlap_v = bool(
                (int(state["walls_v"]) & (wall_bit >> self.wall_size))
                or (int(state["walls_v"]) & (wall_bit << self.wall_size))
            )
            if not overlap_v:
                test_state = self._copy_state(state)
                test_state["walls_v"] |= wall_bit
                test_state["v_block"] |= self._v_expand_lut[idx]
                if self._has_path(test_state, 0) and self._has_path(test_state, 1):
                    valid[4 + self.inner_wall_cnt + self._wall_idx_to_compact(idx)] = 1

        return valid

    def check_win(self, state, player):
        p_idx = 0 if player == 1 else 1
        return self.check_win_by_index(state, p_idx)

    def check_win_by_index(self, state, p_idx):
        pos = self._lsb_index(int(state["p_bits"][p_idx]))
        if pos < 0:
            return False
        row = pos // self.size
        return row == self._goal_rows[p_idx]

    def get_value_and_terminated(self, state, action):
        _ = action
        if self.check_win_by_index(state, 1):
            return 1, True
        if self.check_win_by_index(state, 0):
            return -1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        if player == 1:
            return self._copy_state(state)
        if player != -1:
            return self._copy_state(state)

        out = {
            "p_bits": [
                self._flip_bits(int(state["p_bits"][1]), self.size * self.size),
                self._flip_bits(int(state["p_bits"][0]), self.size * self.size),
            ],
            "walls_h": self._flip_bits(int(state["walls_h"]), self.wall_cnt),
            "walls_v": self._flip_bits(int(state["walls_v"]), self.wall_cnt),
            "h_block": 0,
            "v_block": 0,
            "walls_left": np.array([state["walls_left"][1], state["walls_left"][0]], dtype=np.int8),
            "turn": 1 - int(state["turn"]),
            "is_jumping": bool(state["is_jumping"]),
            "jumper_idx": -1,
            "jump_dir": -1,
        }
        if state["is_jumping"]:
            out["jumper_idx"] = 1 - int(state["jumper_idx"])
            out["jump_dir"] = self._flip_dir.get(int(state["jump_dir"]), -1)
        self._recompute_blocks_from_walls(out)
        return out


class Quoridor9(Quoridor7):
    def __init__(self):
        self._init_quoridor(size=9, initial_walls_left=10, interval=16, game_name="quoridor9")

    def __repr__(self):
        return "Quoridor9"


class Quoridor5(Quoridor7):
    def __init__(self):
        self._init_quoridor(size=5, initial_walls_left=3, interval=24, game_name="quoridor5")

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
