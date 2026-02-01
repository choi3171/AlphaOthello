import numpy as np
import heapq

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        
    def __repr__(self):
        return "TicTacToe"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state


class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        
    def __repr__(self):
        return "ConnectFour"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state
    
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state


'''
변수 이름

state: 판의 상태이다. row*col의 2차원 np.array로 구현. 1,-1은 돌이며 0은 아무것도 없음음
action: state를 변화 시키는 것(ex: 새로운 돌을 놓고 그 돌로 인해 뒤집어지는거 계산). np.array 1차원 행렬 [x,y]
player: int로 표현; 1돌의 주인이 1이고 -1돌의 주인이 -1
'''


class Othello:
    def __init__(self):
        self.row_count = 6 #should be even!
        self.column_count = 6
        self.action_size = self.row_count * self.column_count + 1
        self.interval = 16

    def __repr__(self):
        return "Othello"
    
    def get_initial_state(self):
        initialstate=np.zeros(shape=(self.row_count,self.column_count),dtype=int)
        initialstate[self.row_count//2-1:self.row_count//2+1,self.column_count//2-1:self.column_count//2+1]=np.array([[-1,1],[1,-1]])
        # print(initialstate)
        return initialstate
    
    def restrict(self, row, col):
        return row >= 0 and row < self.row_count and col >= 0 and col < self.column_count
    
    def get_next_state(self, state, action, player):
        #둘다 -1 이면 돌 못 놓은거
        if action == self.action_size - 1:
            pass
        else:
            row = action // self.column_count
            col = action % self.column_count
            state[row,col]=player
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if dx==0 and dy==0: continue
                    x=row
                    y=col
                    while True:
                        x+=dx
                        y+=dy
                        # print(x,y,dx,dy)
                        if (not self.restrict(x,y)) or state[x,y]==0:
                            # print('fuck')
                            break
                        elif state[x,y]==player:
                            x-=dx
                            y-=dy
                            while x!=row or y!=col:
                                state[x,y]=player
                                x-=dx
                                y-=dy
                            break

        return state

    def get_valid_moves(self, state):
        ans=np.zeros(shape=self.action_size,dtype=np.uint8)
        for ij in range(self.row_count*self.column_count):
            i=ij//self.column_count
            j=ij%self.column_count

            if(state[i,j]): continue

            for dx in [-1,0,1]:
                if(ans[ij]): break
                for dy in [-1,0,1]:
                    if(ans[ij]): break
                    if dx==0 and dy==0: continue
                    x=i
                    y=j
                    flip = False
                    while True:
                        x+=dx
                        y+=dy
                        if (not self.restrict(x,y)) or state[x,y]==0:
                            break
                        if state[x,y]==1:
                            if flip:
                                ans[ij]=1
                                break
                            else:
                                break
                        elif state[x,y]==-1:
                            flip=True
        if np.sum(ans) == 0:
            ans[-1] = 1
        return ans

    def check_finish(self, state):
        return self.get_valid_moves(state)[-1] == 1 and self.get_valid_moves(self.change_perspective(state, -1))[-1] == 1
    
    def check_winner(self, state):
        res = np.sum(state)
        if res > 0:
            return 1
        elif res < 0:
            return -1
        else:
            return 0

    def get_value_and_terminated(self, state, action):
        if self.check_finish(state):
            return self.check_winner(state), True
        return 0, False

    def change_perspective(self, state, player:int):
        return state*player

    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state

    def get_visualized_state(self, state):
        size = 16
        visualized_state = np.zeros((3, self.row_count*size, self.row_count*size))
        for i in range(self.row_count):
            for j in range(self.column_count):
                if state[i, j] == 1:
                    visualized_state[0, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[1, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[2, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                elif state[i, j] == 0:
                    visualized_state[0, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[1, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[2, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                elif state[i, j] == -1:
                    visualized_state[0, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[1, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[2, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 1

        return visualized_state

class GomokuNaive:
    def __init__(self):
        self.row_count = 10
        self.column_count = 10
        self.action_size = self.row_count * self.column_count
        self.interval = 16

    def __repr__(self):
        return "GomokuNaive"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state


    def get_valid_moves(self, state):
        valid_moves = (state.reshape(-1) == 0).astype(np.uint8)
        return valid_moves

    def get_max_continuous(self, state, r, c, player):
        max_c = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for direction in [1, -1]:
                nr, nc = r + dr * direction, c + dc * direction
                while 0 <= nr < self.row_count and 0 <= nc < self.column_count and state[nr, nc] == player:
                    count += 1
                    nr += dr * direction
                    nc += dc * direction
            max_c = max(max_c, count)
        return max_c

    def check_win(self, state, action):
        if action is None:
            return False

        row = action // self.column_count
        col = action % self.column_count
        player = state[row, col]
        count = self.get_max_continuous(state, row, col, player)

        return count >= 5

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
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    def get_visualized_state(self, state):
        size = 16
        visualized_state = np.zeros((3, self.row_count * size, self.row_count * size))

        for i in range(self.row_count):
            for j in range(self.column_count):
                r0, r1 = i * self.interval, (i + 1) * self.interval
                c0, c1 = j * self.interval, (j + 1) * self.interval

                if state[i, j] == 1:
                    visualized_state[0, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[1, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[2, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0

                elif state[i, j] == 0:
                    visualized_state[0, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[1, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[2, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0

                elif state[i, j] == -1:
                    visualized_state[0, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[1, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[2, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 1

        return visualized_state

class Quoridor:
    def __init__(self, size=7):
        self.size = size
        self.num_squares = size * size
        self.piece_action_size = self.num_squares
        self.walls_action_size = (size - 1) ** 2
        self.action_size = self.piece_action_size + 2 * self.walls_action_size
        self.initial_walls_left = 5

        self.move_masks = self._precompute_move_masks()
        self.h_wall_masks = self._precompute_h_wall_masks()
        self.v_wall_masks = self._precompute_v_wall_masks()
        
        self.goal_masks = [
            sum(1 << ((self.size - 1) * self.size + c) for c in range(self.size)), # P1 (Top -> Bottom)
            sum(1 << (0 * self.size + c) for c in range(self.size))                # P2 (Bottom -> Top)
        ]

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
    def __init__(self, size=7):
        self.size = size
        self.num_squares = size * size
        self.piece_action_size = self.num_squares
        self.walls_action_size = (size - 1) ** 2
        self.action_size = self.piece_action_size + 2 * self.walls_action_size
        self.initial_walls_left = 5


        self.move_masks = self._precompute_move_masks()
        self.h_wall_masks = self._precompute_h_wall_masks()
        self.v_wall_masks = self._precompute_v_wall_masks()
        
        self.goal_masks = [
            sum(1 << ((self.size - 1) * self.size + c) for c in range(self.size)), # P1 (Top -> Bottom)
            sum(1 << (0 * self.size + c) for c in range(self.size))                # P2 (Bottom -> Top)
        ]

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
                    m = (r * self.size + c + col_off, (r + 1) * self.size + c + col_off)
                    blocks[m] = blocks.get(m, 0) | wall_bit
                    blocks[(m[1], m[0])] = blocks.get((m[1], m[0]), 0) | wall_bit
        return blocks

    def _precompute_v_wall_masks(self):
        blocks = {}
        for r in range(self.size - 1):
            for c in range(self.size - 1):
                wall_bit = 1 << (r * (self.size - 1) + c)
                for row_off in [0, 1]:
                    m = ((r + row_off) * self.size + c, (r + row_off) * self.size + c + 1)
                    blocks[m] = blocks.get(m, 0) | wall_bit
                    blocks[(m[1], m[0])] = blocks.get((m[1], m[0]), 0) | wall_bit
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

    def get_next_state(self, state, action_idx, player):
        p_idx = 0 if player == 1 else 1
        next_state = {
            'p_bits': state['p_bits'].copy(),
            'walls_h': state['walls_h'],
            'walls_v': state['walls_v'],
            'walls_left': state['walls_left'].copy()
        }
        
        if action_idx < self.piece_action_size:
            next_state['p_bits'][p_idx] = 1 << action_idx
        elif action_idx < self.piece_action_size + self.walls_action_size:
            next_state['walls_h'] |= (1 << (action_idx - self.piece_action_size))
            next_state['walls_left'][p_idx] -= 1
        else:
            next_state['walls_v'] |= (1 << (action_idx - self.piece_action_size - self.walls_action_size))
            next_state['walls_left'][p_idx] -= 1
        return next_state
    
    def get_valid_moves(self, state, player):
        p_idx = 0 if player == 1 else 1
        opp_idx = 1 - p_idx
        valid_moves = np.zeros(self.action_size, dtype=int)
        
        curr_bit = state['p_bits'][p_idx]
        curr_idx = curr_bit.bit_length() - 1
        opp_bit = state['p_bits'][opp_idx]
        
        for direction, target_bit in self.move_masks[curr_idx].items():
            target_idx = target_bit.bit_length() - 1
            
            if not self.is_blocked(state, curr_idx, target_idx):
                if target_bit != opp_bit:
                    valid_moves[target_idx] = 1
                else:
                    jump_masks = self.move_masks[target_idx]
                    if direction in jump_masks:
                        j_bit = jump_masks[direction]
                        j_idx = j_bit.bit_length() - 1
                        if not self.is_blocked(state, target_idx, j_idx):
                            valid_moves[j_idx] = 1

        if state['walls_left'][p_idx] > 0:
            for r in range(self.size - 1):
                for c in range(self.size - 1):
                    wall_bit = 1 << (r * (self.size - 1) + c)
                    
                    if not (state['walls_h'] & wall_bit) and not (state['walls_v'] & wall_bit):
                        overlap = False
                        if c > 0 and (state['walls_h'] & (wall_bit >> 1)): overlap = True
                        if c < self.size - 2 and (state['walls_h'] & (wall_bit << 1)): overlap = True
                        
                        if not overlap:
                            state['walls_h'] |= wall_bit
                            if self.has_path(state, 0) and self.has_path(state, 1):
                                valid_moves[self.piece_action_size + r * (self.size-1) + c] = 1
                            state['walls_h'] &= ~wall_bit

                    if not (state['walls_v'] & wall_bit) and not (state['walls_h'] & wall_bit):
                        overlap = False
                        if r > 0 and (state['walls_v'] & (wall_bit >> (self.size - 1))): overlap = True
                        if r < self.size - 2 and (state['walls_v'] & (wall_bit << (self.size - 1))): overlap = True
                        
                        if not overlap:
                            state['walls_v'] |= wall_bit
                            if self.has_path(state, 0) and self.has_path(state, 1):
                                valid_moves[self.piece_action_size + self.walls_action_size + r * (self.size-1) + c] = 1
                            state['walls_v'] &= ~wall_bit
                            
        return valid_moves
    
    def check_win(self, state, player):
        p_idx = 0 if player == 1 else 1
        
        return bool(state['p_bits'][p_idx] & self.goal_masks[p_idx])
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

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