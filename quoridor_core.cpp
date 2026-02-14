#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

typedef uint64_t bitboard;

struct State {
  bitboard p_bits[2];  // 49비트 사용 (0~48)
  bitboard walls_h;    // 36비트 사용 (6x6 격자)
  bitboard walls_v;    // 36비트 사용
  int8_t walls_left[2];
  int8_t turn;  // 0 or 1
};

class Quoridor {
 public:
  static const int SIZE = 7;
  static const int NUM_SQUARES = 49;
  static const int WALL_SIZE = 6;
  static const int ACTION_SIZE = 49 + 2 * 36;

  // 0, 7, 14, 21, 28, 35, 42번째 비트
  const bitboard L_MASK = 0x40810204081ULL;
  // 6, 13, 20, 27, 34, 41, 48번째 비트
  const bitboard R_MASK = 0x1020408102040ULL;  // 마스크 수정 완료

  bitboard goal_masks[2];

  Quoridor() {
    // P1 목표: 마지막 행 (42~48)
    goal_masks[0] = 0x1FC0000000000ULL;
    // P2 목표: 첫 번째 행 (0~6)
    goal_masks[1] = 0x7FULL;
  }

  State get_initial_state() {
    State s;
    s.p_bits[0] = 1ULL << (SIZE / 2);                    // 상단 중앙
    s.p_bits[1] = 1ULL << (NUM_SQUARES - 1 - SIZE / 2);  // 하단 중앙
    s.walls_h = 0;
    s.walls_v = 0;
    s.walls_left[0] = 5;
    s.walls_left[1] = 5;
    s.turn = 0;
    return s;
  }

  State apply_action(const State& state, int action_idx) {
    State next_state = state;  // 현재 상태 복사
    int p_idx = state.turn;

    if (action_idx < 49) {
      // 1. 말 이동 (0~48)
      next_state.p_bits[p_idx] = 1ULL << action_idx;
    } else if (action_idx < 49 + 36) {
      // 2. 가로 벽 설치 (49~84)
      int wall_idx = action_idx - 49;
      next_state.walls_h |= (1ULL << wall_idx);
      next_state.walls_left[p_idx]--;
    } else {
      // 3. 세로 벽 설치 (85~120)
      int wall_idx = action_idx - 85;
      next_state.walls_v |= (1ULL << wall_idx);
      next_state.walls_left[p_idx]--;
    }

    // 턴 교체
    next_state.turn = 1 - state.turn;
    return next_state;
  }

  void render(const State& state) {
    // 상단 열 번호
    std::cout << "\n     ";
    for (int c = 0; c < 7; ++c) std::cout << c << "   ";
    std::cout << "\n   +" << std::string(27, '-') << "+\n";

    for (int r = 0; r < 7; ++r) {
      // 1. 말(Piece)과 세로 벽(Vertical Wall) 출력
      std::cout << " " << r << " | ";
      for (int c = 0; c < 7; ++c) {
        // 말 위치 확인
        uint64_t bit = 1ULL << (r * 7 + c);
        if (state.p_bits[0] & bit)
          std::cout << "P1";
        else if (state.p_bits[1] & bit)
          std::cout << "P2";
        else
          std::cout << " .";

        // 세로 벽 출력 (마지막 열 제외)
        if (c < 6) {
          // 쿼리도 규칙상 세로 벽 하나는 두 칸을 막음
          // 현재 칸(r, c) 또는 윗칸(r-1, c)에 벽이 설치되었는지 확인
          bool v_wall = false;
          if (r < 6 && (state.walls_v & (1ULL << (r * 6 + c)))) v_wall = true;
          if (r > 0 && (state.walls_v & (1ULL << ((r - 1) * 6 + c))))
            v_wall = true;

          std::cout << (v_wall ? " | " : "   ");
        }
      }
      std::cout << " |\n";

      // 2. 가로 벽(Horizontal Wall) 행 출력 (마지막 행 제외)
      if (r < 6) {
        std::cout << "   | ";
        for (int c = 0; c < 7; ++c) {
          // 가로 벽 하나는 두 칸을 막음
          // 현재 칸(r, c) 또는 왼쪽 칸(r, c-1)에 벽이 설치되었는지 확인
          bool h_wall = false;
          if (c < 6 && (state.walls_h & (1ULL << (r * 6 + c)))) h_wall = true;
          if (c > 0 && (state.walls_h & (1ULL << (r * 6 + c - 1))))
            h_wall = true;

          std::cout << (h_wall ? "===" : "   ");

          // 벽 사이의 교차점 처리
          if (c < 6) std::cout << " ";
        }
        std::cout << " |\n";
      }
    }
    std::cout << "   +" << std::string(27, '-') << "+\n";
    std::cout << "   [남은 벽] P1: " << (int)state.walls_left[0]
              << " | P2: " << (int)state.walls_left[1] << " | 턴: P"
              << (int)state.turn + 1 << "\n";
  }

  // [최적화 1] 비트 병렬 Flood Fill: 전체 보드를 한 번에 확장
  bool has_path(const State& state, int p_idx) const {
    bitboard reachable = state.p_bits[p_idx];
    bitboard goal = goal_masks[p_idx];

    // 벽에 의한 이동 차단 마스크 생성 (한 번의 연산으로 전체 보드 적용)
    bitboard h_block = expand_h(state.walls_h);
    bitboard v_block = expand_v(state.walls_v);

    while (true) {
      bitboard prev = reachable;
      // 위로 이동 (Up): 현재 위치 >> 7, 단 아래칸 가로벽(h_block << 7)에 막히지
      // 않아야 함
      bitboard up = (reachable >> SIZE) & ~(h_block);
      // 아래로 이동 (Down): 현재 위치 << 7, 단 현재칸 가로벽(h_block)에 막히지
      // 않아야 함
      bitboard down = (reachable << SIZE) & ~(h_block << SIZE);
      // 왼쪽 이동 (Left): 현재 위치 >> 1, 가장자리 및 세로벽 차단
      bitboard left = ((reachable & ~L_MASK) >> 1) & ~(v_block);
      // 오른쪽 이동 (Right): 현재 위치 << 1, 가장자리 및 세로벽 차단
      bitboard right = ((reachable & ~R_MASK) << 1) & ~(v_block << 1);

      reachable |= (up | down | left | right);

      if (reachable & goal) return true;
      if (reachable == prev) return false;
    }
  }

  // [최적화 2] 벽 비트 확장 로직 (36 -> 49 Mapping)
  // h_walls의 1비트는 두 칸의 상하 이동을 동시에 막음
  inline bitboard expand_h(bitboard w) const {
    bitboard res = 0;
    for (int r = 0; r < WALL_SIZE; r++) {
      bitboard row = (w >> (r * WALL_SIZE)) & 0x3FULL;
      res |= (row | (row << 1)) << (r * SIZE);
    }
    return res;
  }

  inline bitboard expand_v(bitboard w) const {
    bitboard res = 0;
    for (int r = 0; r < WALL_SIZE; r++) {
      bitboard row = (w >> (r * WALL_SIZE)) & 0x3FULL;
      res |= row << (r * SIZE);
      res |= row << ((r + 1) * SIZE);
    }
    return res;
  }

  // 유효 수 계산 (Python의 jump/diagonal 로직 포함)
  std::vector<int> get_valid_moves(State& state) {
    std::vector<int> moves;
    int p_idx = state.turn;
    int opp_idx = 1 - p_idx;
    bitboard my_pos = state.p_bits[p_idx];
    bitboard opp_pos = state.p_bits[opp_idx];
    int curr_idx = get_lsb_index(my_pos);

    // 1. 말 이동 로직
    static const int dr[] = {-1, 1, 0, 0};  // U, D, L, R
    static const int dc[] = {0, 0, -1, 1};

    for (int i = 0; i < 4; ++i) {
      int r = curr_idx / SIZE, c = curr_idx % SIZE;
      int nr = r + dr[i], nc = c + dc[i];
      if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;

      if (!is_move_blocked(state, curr_idx, nr * SIZE + nc)) {
        bitboard target_bit = 1ULL << (nr * SIZE + nc);
        if (target_bit != opp_pos) {
          moves.push_back(nr * SIZE + nc);
        } else {
          // 점프 로직
          int jnr = nr + dr[i], jnc = nc + dc[i];
          bool straight_jump = false;
          if (jnr >= 0 && jnr < SIZE && jnc >= 0 && jnc < SIZE &&
              !is_move_blocked(state, nr * SIZE + nc, jnr * SIZE + jnc)) {
            moves.push_back(jnr * SIZE + jnc);
            straight_jump = true;
          }
          if (!straight_jump) {
            // 대각선 점프
            for (int j = 0; j < 4; ++j) {
              if ((i < 2 && j >= 2) ||
                  (i >= 2 && j < 2)) {  // 수직->수평 or 수평->수직
                int dnr = nr + dr[j], dnc = nc + dc[j];
                if (dnr >= 0 && dnr < SIZE && dnc >= 0 && dnc < SIZE &&
                    !is_move_blocked(state, nr * SIZE + nc, dnr * SIZE + dnc)) {
                  moves.push_back(dnr * SIZE + dnc);
                }
              }
            }
          }
        }
      }
    }

    // 2. 벽 설치 로직
    if (state.walls_left[p_idx] > 0) {
      for (int r = 0; r < WALL_SIZE; ++r) {
        for (int c = 0; c < WALL_SIZE; ++c) {
          bitboard wall_bit = 1ULL << (r * WALL_SIZE + c);
          // 가로벽
          if (!(state.walls_h & wall_bit) && !(state.walls_v & wall_bit)) {
            bool overlap =
                (c > 0 && (state.walls_h & (wall_bit >> 1))) ||
                (c < WALL_SIZE - 1 && (state.walls_h & (wall_bit << 1)));
            if (!overlap) {
              state.walls_h |= wall_bit;
              if (has_path(state, 0) && has_path(state, 1))
                moves.push_back(49 + r * WALL_SIZE + c);
              state.walls_h &= ~wall_bit;
            }
          }
          // 세로벽
          if (!(state.walls_v & wall_bit) && !(state.walls_h & wall_bit)) {
            bool overlap =
                (r > 0 && (state.walls_v & (wall_bit >> WALL_SIZE))) ||
                (r < WALL_SIZE - 1 &&
                 (state.walls_v & (wall_bit << WALL_SIZE)));
            if (!overlap) {
              state.walls_v |= wall_bit;
              if (has_path(state, 0) && has_path(state, 1))
                moves.push_back(49 + 36 + r * WALL_SIZE + c);
              state.walls_v &= ~wall_bit;
            }
          }
        }
      }
    }
    return moves;
  }

  // 특정 이동이 벽에 막혔는지 확인
  bool is_move_blocked(const State& state, int from, int to) const {
    int r1 = from / SIZE, c1 = from % SIZE;
    int r2 = to / SIZE, c2 = to % SIZE;
    if (r1 == r2) {  // 좌우 이동
      int min_c = std::min(c1, c2);
      bitboard wall_bit = 1ULL << (r1 * WALL_SIZE + min_c);
      if (r1 > 0 && (state.walls_v & (1ULL << ((r1 - 1) * WALL_SIZE + min_c))))
        return true;
      if (r1 < WALL_SIZE && (state.walls_v & wall_bit)) return true;
    } else {  // 상하 이동
      int min_r = std::min(r1, r2);
      bitboard wall_bit = 1ULL << (min_r * WALL_SIZE + c1);
      if (c1 > 0 && (state.walls_h & (1ULL << (min_r * WALL_SIZE + c1 - 1))))
        return true;
      if (c1 < WALL_SIZE && (state.walls_h & wall_bit)) return true;
    }
    return false;
  }

  bool check_win(const State& state, int p_idx) const {
    // 플레이어의 위치 비트와 해당 플레이어의 목표 마스크를 AND 연산
    return (state.p_bits[p_idx] & goal_masks[p_idx]) != 0;
  }

  State get_next_state(const State& state, int action_idx) {
    return apply_action(state, action_idx);
  }

  State change_perspective(const State& state, int player) {
    if (player == 0) return state;

    State new_state;
    new_state.turn = 0;

    new_state.walls_left[0] = state.walls_left[1];
    new_state.walls_left[1] = state.walls_left[0];

    new_state.p_bits[0] = flip_bits(state.p_bits[1], 49);
    new_state.p_bits[1] = flip_bits(state.p_bits[0], 49);

    new_state.walls_h = flip_bits(state.walls_h, 36);
    new_state.walls_v = flip_bits(state.walls_v, 36);

    return new_state;
  }

  std::vector<float> get_encoded_state(const State& state) {
    // 채널 6개, 7x7 크기
    std::vector<float> encoded(6 * 49, 0.0f);

    int current_turn = state.turn;         // 0 or 1
    bool need_flip = (current_turn == 1);  // P2 차례면 보드 뒤집어서 저장

    int my_idx = current_turn;
    int opp_idx = 1 - current_turn;

    // Plane 0: My Position
    if (state.p_bits[my_idx]) {
      int idx = get_lsb_index(state.p_bits[my_idx]);
      int final_idx = need_flip ? (49 - 1 - idx) : idx;
      encoded[0 * 49 + final_idx] = 1.0f;
    }

    // Plane 1: Opponent Position
    if (state.p_bits[opp_idx]) {
      int idx = get_lsb_index(state.p_bits[opp_idx]);
      int final_idx = need_flip ? (49 - 1 - idx) : idx;
      encoded[1 * 49 + final_idx] = 1.0f;
    }

    // Helper Lambda for Walls
    // 벽 인덱스(0~35)를 7x7 그리드의 좌표로 매핑 후 1.0f 할당
    auto fill_wall_plane = [&](int plane_offset, bitboard walls) {
      bitboard temp = walls;
      while (temp) {
        int w_idx = get_lsb_index(temp);
        int final_w_idx = need_flip ? (36 - 1 - w_idx) : w_idx;

        // Wall Index(0~35) -> 7x7 Grid Index(0~48) 매핑
        // 6x6 격자는 7x7 격자에서 (0,0)~(5,5)에 해당하므로 행/열 계산 필요
        int r = final_w_idx / 6;
        int c = final_w_idx % 6;

        // 7x7 평면상의 인덱스 (stride = 7)
        int grid_idx = r * 7 + c;
        encoded[plane_offset * 49 + grid_idx] = 1.0f;

        temp &= (temp - 1);
      }
    };

    // Plane 2: Horizontal Walls
    fill_wall_plane(2, state.walls_h);

    // Plane 3: Vertical Walls
    fill_wall_plane(3, state.walls_v);

    // Plane 4: My Walls Left (Scalar plane)
    float my_val =
        (float)state.walls_left[my_idx] / 6.0f;  // 6.0f: Initial walls
    std::fill(encoded.begin() + 4 * 49, encoded.begin() + 5 * 49, my_val);

    // Plane 5: Opponent Walls Left
    float opp_val = (float)state.walls_left[opp_idx] / 6.0f;
    std::fill(encoded.begin() + 5 * 49, encoded.begin() + 6 * 49, opp_val);

    return encoded;
  }

  // 4. Utils
  int get_opponent(int player) { return 1 - player; }
  float get_opponent_value(float value) { return -value; }

  inline int get_lsb_index(uint64_t v) const {
    if (v == 0) return 64;  // 안전 장치
#ifdef _MSC_VER
    // Visual Studio
    unsigned long index;
    _BitScanForward64(&index, v);
    return (int)index;
#else
    // GCC, Clang
    return __builtin_ctzll(v);
#endif
  }

  inline uint64_t flip_bits(uint64_t val, int max_bits) const {
    uint64_t res = 0;
    while (val) {
      int idx = get_lsb_index(val);
      int new_idx = (max_bits - 1) - idx;
      res |= (1ULL << new_idx);
      val &= (val - 1);  // 최하위 비트 삭제
    }
    return res;
  }
};

int main() {
  srand(time(NULL));
  Quoridor engine;
  State state = engine.get_initial_state();

  while (true) {
    engine.render(state);  // 현재 보드 상태 출력

    // 승리 확인
    if (engine.check_win(state, 0)) {
      std::cout << "P1 승리!\n";
      break;
    }
    if (engine.check_win(state, 1)) {
      std::cout << "P2 승리!\n";
      break;
    }

    // 유효한 수 가져오기
    auto moves = engine.get_valid_moves(state);
    if (moves.empty()) {
      std::cout << "둘 수 있는 수가 없습니다.\n";
      break;
    }

    // 무작위로 한 수 선택 (검증용)
    int random_move = moves[rand() % moves.size()];

    // 상태 업데이트 (apply_action 함수가 구현되어 있어야 함)
    state = engine.apply_action(state, random_move);
  }

  return 0;
}
