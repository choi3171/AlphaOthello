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
    def __init__(self):
        self.size = 7
        self.piece_action_size = self.size**2
        self.walls_action_size = (self.size - 1)**2
        self.action_size = self.piece_action_size + 2 * self.walls_action_size
        self.initial_walls_left = 5

    def get_initial_state(self):
        return {
            'pos': np.array([[0, self.size//2], [self.size-1, self.size//2]]),
            'walls_v': np.zeros((self.size-1, self.size-1), dtype=np.int8),
            'walls_h': np.zeros((self.size-1, self.size-1), dtype=np.int8),
            'walls_left': np.array([self.initial_walls_left, self.initial_walls_left]),
            'path_cache': [None, None]
        }
    
    def get_next_state(self, state, action_idx, player):
        # 배열 단위 copy로 독립성 보장 (MCTS 안전)
        next_state = {
            'pos': state['pos'].copy(),
            'walls_v': state['walls_v'].copy(),
            'walls_h': state['walls_h'].copy(),
            'walls_left': state['walls_left'].copy(),
            'path_cache': [None, None]
        }
        p_idx = 0 if player == 1 else 1

        if action_idx < self.piece_action_size:
            row, col = divmod(action_idx, self.size)
            next_state['pos'][p_idx] = [row, col]
        elif action_idx < self.piece_action_size + self.walls_action_size:
            idx = action_idx - self.piece_action_size
            row, col = divmod(idx, self.size - 1)
            next_state['walls_h'][row, col] = 1
            next_state['walls_left'][p_idx] -= 1
        else:
            idx = action_idx - self.piece_action_size - self.walls_action_size
            row, col = divmod(idx, self.size - 1)
            next_state['walls_v'][row, col] = 1
            next_state['walls_left'][p_idx] -= 1
        return next_state

    def get_valid_moves(self, state, player):
        valid_moves = np.zeros(self.action_size, dtype=int)
        p_idx = 0 if player == 1 else 1
        r, c = state['pos'][p_idx]

        # 1. 이동(Move) - O(1) 위치 참조
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if not self.is_blocked_by_wall(state, r, c, nr, nc):
                    # 상대방 위치 확인
                    opp_pos = state['pos'][1 - p_idx]
                    if not (nr == opp_pos[0] and nc == opp_pos[1]):
                        valid_moves[nr * self.size + nc] = 1
                    else: # 점프 로직
                        jnr, jnc = nr + dr, nc + dc
                        if 0 <= jnr < self.size and 0 <= jnc < self.size and not self.is_blocked_by_wall(state, nr, nc, jnr, jnc):
                            valid_moves[jnr * self.size + jnc] = 1

        # 2. 벽(Wall) - O(N) 경로 캐싱 기반 최적화
        if state['walls_left'][p_idx] > 0:
            path1 = self.get_any_path(state, 1)
            path2 = self.get_any_path(state, -1)

            for row in range(self.size - 1):
                for col in range(self.size - 1):
                    if state['walls_h'][row, col] == 0 and state['walls_v'][row, col] == 0:
                        # 가로 벽
                        if (col == 0 or state['walls_h'][row, col-1] == 0) and (col == self.size-2 or state['walls_h'][row, col+1] == 0):
                            if self.is_path_safe(state, row, col, 'H', path1, path2):
                                valid_moves[self.piece_action_size + row * (self.size-1) + col] = 1
                        # 세로 벽
                        if (row == 0 or state['walls_v'][row-1, col] == 0) and (row == self.size-2 or state['walls_v'][row+1, col] == 0):
                            if self.is_path_safe(state, row, col, 'V', path1, path2):
                                valid_moves[self.piece_action_size + self.walls_action_size + row * (self.size-1) + col] = 1
        return valid_moves

    def is_path_safe(self, state, r, c, orientation, path1, path2):
        # 1. 신규 벽이 막는 에지 정의
        if orientation == 'H':
            blocked = {((r, c), (r+1, c)), ((r, c+1), (r+1, c+1))}
            target_layer = 'walls_h'
        else:
            blocked = {((r, c), (r, c+1)), ((r+1, c), (r+1, c+1))}
            target_layer = 'walls_v'

        # 2. 캐시된 경로를 건드리는지 확인
        needs_check1 = any((u, v) in blocked or (v, u) in blocked for u, v in zip(path1, path1[1:]))
        needs_check2 = any((u, v) in blocked or (v, u) in blocked for u, v in zip(path2, path2[1:]))

        if not needs_check1 and not needs_check2: return True

        # 3. In-place 체크 (복사 방지 최적화)
        state[target_layer][r, c] = 1
        res = True
        if needs_check1 and not self.has_path_exists(state, 1): res = False
        if res and needs_check2 and not self.has_path_exists(state, -1): res = False
        state[target_layer][r, c] = 0 # 원복
        return res

    def has_path_exists(self, state, player):
        """A* 기반 빠른 경로 존재 확인"""
        p_idx = 0 if player == 1 else 1
        start = tuple(state['pos'][p_idx])
        goal_row = (self.size - 1) if player == 1 else 0
        
        pq = [(abs(start[0] - goal_row), 0, start)]
        visited = {start}
        while pq:
            f, g, (r, c) = heapq.heappop(pq)
            if r == goal_row: return True
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in visited:
                    if not self.is_blocked_by_wall(state, r, c, nr, nc):
                        visited.add((nr, nc))
                        heapq.heappush(pq, (g + 1 + abs(nr - goal_row), g + 1, (nr, nc)))
        return False

    def get_any_path(self, state, player):
        """A* 기반 최단 경로 반환"""
        p_idx = 0 if player == 1 else 1
        start = tuple(state['pos'][p_idx])
        goal_row = (self.size - 1) if player == 1 else 0
        pq = [(abs(start[0] - goal_row), 0, start, [start])]
        visited = {start: 0}
        while pq:
            f, g, (r, c), path = heapq.heappop(pq)
            if r == goal_row: return path
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) not in visited or visited[(nr, nc)] > g + 1:
                        if not self.is_blocked_by_wall(state, r, c, nr, nc):
                            visited[(nr, nc)] = g + 1
                            heapq.heappush(pq, (g + 1 + abs(nr - goal_row), g + 1, (nr, nc), path + [(nr, nc)]))
        return [start]

    def is_blocked_by_wall(self, state, r, c, nr, nc):
        if r == nr: # 가로 이동
            tc = c if nc > c else nc
            return (r < self.size-1 and state['walls_v'][r, tc]) or (r > 0 and state['walls_v'][r-1, tc])
        else: # 세로 이동
            tr = r if nr > r else nr
            return (c < self.size-1 and state['walls_h'][tr, c]) or (c > 0 and state['walls_h'][tr, c-1])

    def check_win(self, state, player):
        p_idx = 0 if player == 1 else 1
        return (player == 1 and state['pos'][p_idx, 0] == self.size - 1) or \
               (player == -1 and state['pos'][p_idx, 0] == 0)

    def change_perspective(self, state, player):
            if player == 1:
                return state

            old_p1_path = state['path_cache'][0]
            old_p2_path = state['path_cache'][1]
            
            new_p1_path = None
            new_p2_path = None

            if old_p2_path is not None:
                new_p1_path = [(self.size - 1 - r, c) for r, c in old_p2_path]
                
            if old_p1_path is not None:
                new_p2_path = [(self.size - 1 - r, c) for r, c in old_p1_path]
                
            new_state = {
                'pos': np.array([
                    [self.size - 1 - state['pos'][1][0], state['pos'][1][1]], 
                    [self.size - 1 - state['pos'][0][0], state['pos'][0][1]]
                ]),
                
                'walls_v': np.flipud(state['walls_v']),
                'walls_h': np.flipud(state['walls_h']),
                
                'walls_left': np.array([state['walls_left'][1], state['walls_left'][0]]),
                
                'path_cache': [None, None]
            }
            
            return new_state
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent_value(self, value):
        return -value

    def get_encoded_state(self, state):
        """
        Channel 구성:
        0: Player 1 위치 (1이면 존재, 0이면 없음)
        1: Player 2 위치
        2: 가로 벽(Horizontal Walls) 위치 (1이면 존재)
        3: 세로 벽(Vertical Walls) 위치
        4: Player 1 남은 벽 비율 (0.0 ~ 1.0)
        5: Player 2 남은 벽 비율 (0.0 ~ 1.0)
        """
        encoded_state = np.zeros((6, self.size, self.size), dtype=np.float32)
        
        # 1. [Channel 0, 1] 플레이어 위치 정보
        p1_r, p1_c = state['pos'][0]
        p2_r, p2_c = state['pos'][1]
        
        encoded_state[0, p1_r, p1_c] = 1.0
        encoded_state[1, p2_r, p2_c] = 1.0
        
        # 2. [Channel 2, 3] 벽 위치 정보, padding 존재
        encoded_state[2, :self.size-1, :self.size-1] = state['walls_h']
        encoded_state[3, :self.size-1, :self.size-1] = state['walls_v']
        
        # 3. [Channel 4, 5] 남은 벽 개수 (Scalar -> Plane Broadcasting)
        p1_wall_ratio = state['walls_left'][0] / self.initial_walls_left
        p2_wall_ratio = state['walls_left'][1] / self.initial_walls_left
        
        encoded_state[4].fill(p1_wall_ratio)
        encoded_state[5].fill(p2_wall_ratio)
        
        return encoded_state

    def render(self, state):
        # 상단 열 번호 표시
        header = "       " + "    ".join([str(i) for i in range(self.size)])
        print("\n" + header)
        print("    " + "-" * (self.size * 4 + 5))
        
        # 플레이어 위치 추출 (딕셔너리 참조)
        p1_pos = state['pos'][0]
        p2_pos = state['pos'][1]
        
        for r in range(self.size):
            # 1. 말(Piece)과 세로 벽(Vertical Wall) 행 출력
            row_str = f" {r} | "
            for c in range(self.size):
                # 말 표시
                if np.array_equal([r, c], p1_pos):
                    char = "P1"
                elif np.array_equal([r, c], p2_pos):
                    char = "P2"
                else:
                    char = " ."
                row_str += char
                
                # 세로 벽 표시 (walls_v 참조)
                if c < self.size - 1:
                    # 세로 벽은 (r, c) 또는 (r-1, c) 위치의 wall_v 값이 1일 때 존재
                    is_v_wall = (r < self.size - 1 and state['walls_v'][r, c] == 1) or \
                                (r > 0 and state['walls_v'][r-1, c] == 1)
                    row_str += " | " if is_v_wall else "   "
            row_str += " |"
            print(row_str)
            
            # 2. 가로 벽(Horizontal Wall) 및 교차점 행 출력
            if r < self.size - 1:
                wall_line = "   | "
                for c in range(self.size):
                    # 가로 벽 체크 (walls_h 참조)
                    is_h_wall = (c < self.size - 1 and state['walls_h'][r, c] == 1) or \
                                (c > 0 and state['walls_h'][r, c-1] == 1)
                    if is_h_wall:
                        wall_line += "=== "
                    else:
                        wall_line += "   "

                    # 벽을 놓을 수 있는 교차점(Index) 표시
                    if c < self.size - 1:
                        # 이미 벽이 있는 곳은 공백, 없는 곳은 인덱스 표시
                        if state['walls_h'][r, c] == 0 and state['walls_v'][r, c] == 0:
                            wall_line += f"{r}{c}"
                        else:
                            wall_line += " "
                print(wall_line)
        
        print("    " + "-" * (self.size * 4 + 5))
        print(f"남은 벽 -> Player 1: {state['walls_left'][0]}개 | Player 2: {state['walls_left'][1]}개")

#Bitboard 연산으로 속도 최대화(by. Gemini) → 이해하지 못함...
class BitboardQuoridor:
    def __init__(self, size=7):
        self.size = size
        self.num_squares = size * size
        self.piece_action_size = self.num_squares
        self.walls_action_size = (size - 1) ** 2
        self.action_size = self.piece_action_size + 2 * self.walls_action_size
        self.initial_walls_left = 5

        # 1. 이동 마스크 및 벽 충돌 마스크 사전 계산 (Pre-computation)
        self.move_masks = self._precompute_move_masks()
        self.h_wall_masks = self._precompute_h_wall_masks()
        self.v_wall_masks = self._precompute_v_wall_masks()
        
        # 2. 플레이어별 목표 행 비트마스크
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
        # 특정 (from, to) 이동을 막는 가로벽 비트들을 매핑
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
        # 특정 (from, to) 이동을 막는 세로벽 비트들을 매핑
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
            'walls_h': 0, # 64비트 정수로 가로벽 관리
            'walls_v': 0, # 64비트 정수로 세로벽 관리
            'walls_left': np.array([self.initial_walls_left, self.initial_walls_left])
        }

    def is_blocked(self, state, f_idx, t_idx):
        # 비트 AND 연산으로 즉시 충돌 확인 (O(1))
        if self.h_wall_masks.get((f_idx, t_idx), 0) & state['walls_h']: return True
        if self.v_wall_masks.get((f_idx, t_idx), 0) & state['walls_v']: return True
        return False

    def has_path(self, state, player_idx):
        # 비트 Flood Fill: 현재 도달 가능한 모든 칸을 비트 합집합으로 계산 (초고속)
        reachable = state['p_bits'][player_idx]
        goal_mask = self.goal_masks[player_idx]
        
        while True:
            next_reachable = reachable
            temp = reachable
            while temp:
                curr_bit = temp & -temp # 최하위 비트 추출
                curr_idx = curr_bit.bit_length() - 1
                for _, target_bit in self.move_masks[curr_idx].items():
                    if not self.is_blocked(state, curr_idx, target_bit.bit_length() - 1):
                        next_reachable |= target_bit
                temp &= temp - 1 # 처리한 비트 제거
            
            if next_reachable & goal_mask: return True
            if next_reachable == reachable: return False
            reachable = next_reachable

    def get_next_state(self, state, action_idx, player):
        p_idx = 0 if player == 1 else 1
        # 넘파이 배열 복사 최소화, 벽은 단순 정수 대입
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
        
        # 1. 말 이동(Move) - O(1) 비트 연산 기반
        for direction, target_bit in self.move_masks[curr_idx].items():
            target_idx = target_bit.bit_length() - 1
            
            # 벽에 막혀있는지 확인
            if not self.is_blocked(state, curr_idx, target_idx):
                # 상대방이 없다면 일반 이동
                if target_bit != opp_bit:
                    valid_moves[target_idx] = 1
                else:
                    # 점프 로직: 상대방이 있을 때 같은 방향으로 점프 가능한지 확인
                    jump_masks = self.move_masks[target_idx]
                    if direction in jump_masks:
                        j_bit = jump_masks[direction]
                        j_idx = j_bit.bit_length() - 1
                        if not self.is_blocked(state, target_idx, j_idx):
                            valid_moves[j_idx] = 1

        # 2. 벽 설치(Wall) - 비트 마스킹 기반
        if state['walls_left'][p_idx] > 0:
            for r in range(self.size - 1):
                for c in range(self.size - 1):
                    wall_bit = 1 << (r * (self.size - 1) + c)
                    
                    # 가로 벽(Horizontal) 설치 가능 여부
                    # 조건: 해당 지점에 벽이 없고, 양옆 가로벽과 겹치지 않아야 함
                    if not (state['walls_h'] & wall_bit) and not (state['walls_v'] & wall_bit):
                        overlap = False
                        # 좌우 인접 가로벽 체크 (비트 시프트로 O(1) 확인)
                        if c > 0 and (state['walls_h'] & (wall_bit >> 1)): overlap = True
                        if c < self.size - 2 and (state['walls_h'] & (wall_bit << 1)): overlap = True
                        
                        if not overlap:
                            # 경로 검사 (In-place 수정으로 성능 최적화)
                            state['walls_h'] |= wall_bit
                            if self.has_path(state, 0) and self.has_path(state, 1):
                                valid_moves[self.piece_action_size + r * (self.size-1) + c] = 1
                            state['walls_h'] &= ~wall_bit # 원복

                    # 세로 벽(Vertical) 설치 가능 여부
                    if not (state['walls_v'] & wall_bit) and not (state['walls_h'] & wall_bit):
                        overlap = False
                        # 상하 인접 세로벽 체크
                        if r > 0 and (state['walls_v'] & (wall_bit >> (self.size - 1))): overlap = True
                        if r < self.size - 2 and (state['walls_v'] & (wall_bit << (self.size - 1))): overlap = True
                        
                        if not overlap:
                            state['walls_v'] |= wall_bit
                            if self.has_path(state, 0) and self.has_path(state, 1):
                                valid_moves[self.piece_action_size + self.walls_action_size + r * (self.size-1) + c] = 1
                            state['walls_v'] &= ~wall_bit # 원복
                            
        return valid_moves
    
    def check_win(self, state, player):
        # player: 1 (Player 1, Top), -1 (Player 2, Bottom)
        # 플레이어 인덱스 설정 (1 -> 0, -1 -> 1)
        p_idx = 0 if player == 1 else 1
        
        # 플레이어의 현재 위치 비트와 해당 플레이어의 목표 지점 마스크를 AND 연산
        # 결과가 0이 아니면 목표 지점에 도달한 것임
        return bool(state['p_bits'][p_idx] & self.goal_masks[p_idx])
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

        def get_opponent_value(self, value):
        return -value

    def get_encoded_state(self, state):
        """
        Channel 구성:
        0: Player 1 위치 (1이면 존재, 0이면 없음)
        1: Player 2 위치
        2: 가로 벽(Horizontal Walls) 위치 (1이면 존재)
        3: 세로 벽(Vertical Walls) 위치
        4: Player 1 남은 벽 비율 (0.0 ~ 1.0)
        5: Player 2 남은 벽 비율 (0.0 ~ 1.0)
        """
        encoded_state = np.zeros((6, self.size, self.size), dtype=np.float32)
        
        # 1. [Channel 0, 1] 플레이어 위치 정보
        p1_r, p1_c = state['pos'][0]
        p2_r, p2_c = state['pos'][1]
        
        encoded_state[0, p1_r, p1_c] = 1.0
        encoded_state[1, p2_r, p2_c] = 1.0
        
        # 2. [Channel 2, 3] 벽 위치 정보, padding 존재
        encoded_state[2, :self.size-1, :self.size-1] = state['walls_h']
        encoded_state[3, :self.size-1, :self.size-1] = state['walls_v']
        
        # 3. [Channel 4, 5] 남은 벽 개수 (Scalar -> Plane Broadcasting)
        p1_wall_ratio = state['walls_left'][0] / self.initial_walls_left
        p2_wall_ratio = state['walls_left'][1] / self.initial_walls_left
        
        encoded_state[4].fill(p1_wall_ratio)
        encoded_state[5].fill(p2_wall_ratio)
        
        return encoded_state