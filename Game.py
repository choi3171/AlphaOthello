import numpy as np

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

import numpy as np
from collections import deque

class Quoridor:
    def __init__(self):
        self.size = 7
        self.piece_action_size = self.size**2
        self.walls_action_size = (self.size - 1)**2
        self.action_size = self.piece_action_size + 2 * self.walls_action_size
        self.initial_walls_left = 5

    def get_initial_state(self):
        BoardState = [np.zeros(shape=(self.size, self.size), dtype=int),
                      np.zeros(shape=(self.size - 1, self.size - 1), dtype=int),
                      np.zeros(shape=(self.size - 1, self.size - 1), dtype=int),
                      np.zeros(shape=(2, 1), dtype=int)]
        BoardState[0][0, self.size//2] = 1
        BoardState[0][-1, self.size//2] = -1
        BoardState[3][0, 0] = self.initial_walls_left
        BoardState[3][1, 0] = self.initial_walls_left
        return BoardState

    def get_next_state(self, state, action_idx, player):
        # MCTS 연산을 위해 깊은 복사 수행
        next_state = [s.copy() for s in state]
        p_idx = 0 if player == 1 else 1

        # 1. 이동 액션 처리
        if action_idx < self.piece_action_size:
            row, col = divmod(action_idx, self.size)
            next_state[0][next_state[0] == player] = 0 # 기존 위치 제거
            next_state[0][row, col] = player
            
        # 2. 가로 벽 설치 (action_sort = 1 역할)
        elif action_idx < self.piece_action_size + self.walls_action_size:
            idx = action_idx - self.piece_action_size
            row, col = divmod(idx, self.size - 1)
            next_state[1][row, col] = 1
            next_state[3][p_idx, 0] -= 1 # (2, 1) 구조이므로 [p_idx, 0] 접근
            
        # 3. 세로 벽 설치 (action_sort = 2 역할)
        else:
            idx = action_idx - self.piece_action_size - self.walls_action_size
            row, col = divmod(idx, self.size - 1)
            next_state[2][row, col] = 1
            next_state[3][p_idx, 0] -= 1
            
        return next_state

    def get_valid_moves(self, state, player):
        #valid_moves[0 : self.piece_action_size] : 말 놓을 수 있는 장소
        #valid_moves[self.piece_action_szie : self.piece_action_size + self.walls_action_size] : 가로 벽 놓을 수 있는 장소
        #valid_moves[self.walls_action_size : ] : 세로 벽 놓을 수 있는 장소
        valid_moves = np.zeros(self.action_size, dtype=int)
        
        # 현재 플레이어의 위치 찾기
        pos = np.argwhere(state[0] == player)[0]
        r, c = pos[0], pos[1]

        # --- 1. 이동(Move) 가능 여부 체크 ---
        # 상(0), 하(1), 좌(2), 우(3)
        dir = [(-1, 0), (1, 0), (0, -1), (0, 1)]


        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]
        
        for d in dir:
            nr, nc = r + d[0], c + d[1]
            if 0 <= nr < self.size and 0 <= nc < self.size:
                # 칸 사이에 벽이 있는지 체크하는 로직 (별도 함수 구현 권장)
                if not self.is_blocked_by_wall(state, r, c, nr, nc):
                    # 다른 플레이어가 있다면 점프 로직이 필요하지만, 우선 단순 이동만 구현
                    if state[0][nr, nc] == 0:
                        valid_moves[nr * self.size + nc] = 1
                    elif state[0][nr, nc] == -player:
                        jump_dir = [x for x in dir if x != (-d[0], -d[1])]
                        for j_d in jump_dir:
                            jnr, jnc = nr + j_d[0], nc + j_d[1]
                            if 0 <= jnr < self.size and 0 <= jnc < self.size:
                                if not self.is_blocked_by_wall(state, nr, nc, jnr, jnc):
                                    if state[0][jnr, jnc] == 0:
                                        valid_moves[jnr * self.size + jnc] = 1                

        # --- 2. 벽(Wall) 설치 가능 여부 체크 ---
        p_idx = 0 if player == 1 else 1
        if state[3][p_idx, 0] > 0:
            for row in range(self.size - 1):
                for col in range(self.size - 1):
                    if state[1][row, col] == 0 and state[2][row, col] == 0:
                        #가로 벽 놓을 수 있는 장소
                        if self.path_exists_after_wall(state, row, col, orientation = 'H'):
                            if (col > 0 and state[1][row, col-1] == 0 or col == 0) and (col+1 < self.size-1 and state[1][row, col+1] == 0 or col == self.size-1):
                                valid_moves[self.piece_action_size + row * (self.size-1) + col] = 1
                        #세로 벽 놓을 수 있는 장소
                        if self.path_exists_after_wall(state, row, col, orientation = 'V'):
                            if (row > 0 and state[2][row-1, col] == 0 or row == 0) and (row+1 < self.size-1 and state[2][row+1, col] == 0 or row == self.size-1):
                                valid_moves[self.piece_action_size + self.walls_action_size + row * (self.size-1) + col] = 1
        return valid_moves
    
    def path_exists_after_wall(self, state, r, c, orientation):
        # 임시로 벽을 설치해본 상태를 시뮬레이션
        temp_state = [s.copy() for s in state]
        if orientation == 'H': temp_state[1][r, c] = 1
        else: temp_state[2][r, c] = 1
        
        # 두 플레이어 각각에 대해 BFS 수행
        for p in [1, -1]:
            start_pos = np.argwhere(temp_state[0] == p)[0]
            goal_row = (self.size - 1) if p == 1 else 0 # 플레이어 1은 맨 아래, 2는 맨 위가 목표라고 가정
            
            if not self.bfs_check(temp_state, tuple(start_pos), goal_row):
                return False
        return True

    def bfs_check(self, state, start, goal_row):
        queue = deque([start])
        visited = {start}
        while queue:
            r, c = queue.popleft()
            if r == goal_row: return True
            
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) not in visited and not self.is_blocked_by_wall(state, r, c, nr, nc):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def is_blocked_by_wall(self, state, r, c, nr, nc):
        """두 칸(r,c)와 (nr,nc) 사이를 가로막는 벽이 있는지 확인"""
        # 가로 이동 시: 세로 벽이 막고 있는지 체크
        if r == nr: 
            target_c = min(c, nc)
            # 세로 벽(Channel 2)이 (r, target_c) 또는 (r-1, target_c)에 있는지 확인
            if (r < self.size - 1 and state[2][r, target_c] == 1) or (r > 0 and state[2][r-1, target_c] == 1):
                return True
        # 세로 이동 시: 가로 벽이 막고 있는지 체크
        elif c == nc:
            target_r = min(r, nr)
            # 가로 벽(Channel 1)이 (target_r, c) 또는 (target_r, c-1)에 있는지 확인
            if (c < self.size - 1 and state[1][target_r, c] == 1) or (c > 0 and state[1][target_r, c-1] == 1):
                return True
        return False
    
    def check_win(self, state, player):
        pos = np.argwhere(state[0] == player)[0]
        r, c = pos[0], pos[1]
        result = False

        if player == 1 and r == self.size - 1:
            result = True
        
        if player == -1 and r == 0:
            result = True
        
        return result 

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent_value(self, value):
        return -value

    def render(self, state):
        # 상단 열 번호 표시 (정렬을 위해 공백 조정)
        header = "       " + "    ".join([str(i) for i in range(self.size)])
        print("\n" + header)
        print("    " + "-" * (self.size * 4 + 5))
        
        for r in range(self.size):
            # 1. 말의 행 출력
            row_str = f" {r} | "
            for c in range(self.size):
                p = state[0][r, c]
                char = "P1" if p == 1 else "P2" if p == -1 else " ."
                row_str += char
                # 세로 벽 표시
                if c < self.size - 1:
                    is_v_wall = (r < self.size - 1 and state[2][r, c] == 1) or (r > 0 and state[2][r-1, c] == 1)
                    row_str += " | " if is_v_wall else "   "
            row_str += " |"
            print(row_str)
            
            # 2. 가로 벽 및 중점(벽 인덱스) 행 출력
            if r < self.size - 1:
                wall_line = "   | "
                for c in range(self.size):
                    # 가로 벽 체크
                    is_h_wall = (c < self.size - 1 and state[1][r, c] == 1) or (c > 0 and state[1][r, c-1] == 1)
                    if is_h_wall:
                        wall_line += "=== "
                    else:
                        wall_line += "   "

                    # 중점 인덱스 표시 (말과 말 사이 대각선 위치)
                    if c < self.size - 1:
                        if state[1][r, c] == 0 and state[2][r, c] == 0:
                            wall_line += f"{r}{c}"
                print(wall_line)
        
        print("    " + "-" * (self.size * 4 + 5))
        print(f"남은 벽 -> Player 1: {state[3][0, 0]}개 | Player 2: {state[3][1, 0]}개")  

class Play:
    def __init__(self):
        self.game = Othello()
        self.state = self.game.get_initial_state()
        self.player = 1

    def play(self):
        while not self.game.check_finish(self.game.change_perspective(self.state, self.player)):
            print(self.game.get_valid_moves(self.game.change_perspective(self.state, self.player)))
            print("{0}의 차례".format(self.player))
            self.print_state()
            row = int(input("행: ")) - 1
            col = int(input("열: ")) - 1
            if row == -1 and col == -1:
                action = self.game.action_size - 1
            else:
                action = row * self.game.row_count + col
            self.state = self.game.get_next_state(self.state, action, self.player)
            self.player = self.game.get_opponent(self.player)
        self.print_state()
        print("{0}의 승리!".format(self.game.check_winner(self.state)))
    
    def print_state(self):
        print(self.state)
