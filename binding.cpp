#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // std::vector <-> python list 자동 변환

#include "quoridor_core.cpp"  // 작성하신 쿼리도 엔진 소스 (헤더 분리 안 했다면 직접 인클루드)

namespace py = pybind11;

PYBIND11_MODULE(quoridor_engine, m) {
  m.doc() = "Quoridor 7x7 Engine";

  // 1. State 구조체 바인딩
  py::class_<State>(m, "State")
      .def(py::init<>())
      .def_readwrite("walls_h", &State::walls_h)
      .def_readwrite("walls_v", &State::walls_v)
      .def_readwrite("turn", &State::turn)
      .def(py::pickle(
            [](const State &s) { // __getstate__: C++ -> Python Tuple
                return py::make_tuple(
                    s.p_bits[0], s.p_bits[1], 
                    s.walls_h, s.walls_v, 
                    s.walls_left[0], s.walls_left[1], 
                    s.turn
                );
            },
            [](py::tuple t) { // __setstate__: Python Tuple -> C++
                State s;
                s.p_bits[0] = t[0].cast<uint64_t>();
                s.p_bits[1] = t[1].cast<uint64_t>();
                s.walls_h   = t[2].cast<uint64_t>();
                s.walls_v   = t[3].cast<uint64_t>();
                s.walls_left[0] = t[4].cast<int8_t>();
                s.walls_left[1] = t[5].cast<int8_t>();
                s.turn      = t[6].cast<int8_t>();
                return s;
            }
        ))
      .def_property(
          "p_bits",
          [](State& s) {
            return std::vector<uint64_t>{s.p_bits[0], s.p_bits[1]};
          },
          [](State& s, const std::vector<uint64_t>& v) {
            s.p_bits[0] = v[0];
            s.p_bits[1] = v[1];
          })
      // int8_t는 파이썬에서 문자로 깨질 수 있어서 int로 캐스팅
      .def_property(
          "walls_left",
          [](State& s) {
            return std::vector<int>{s.walls_left[0], s.walls_left[1]};
          },
          [](State& s, const std::vector<int>& v) {
            s.walls_left[0] = (int8_t)v[0];
            s.walls_left[1] = (int8_t)v[1];
          })
      // 디버깅용 문자열 표현 (__repr__)
      .def("__repr__", [](const State& s) {
        return "<State turn=" + std::to_string(s.turn) +
               " P1=" + std::to_string(s.p_bits[0]) +
               " P2=" + std::to_string(s.p_bits[1]) + ">";
      });

  // 2. Quoridor 클래스 바인딩
  py::class_<Quoridor>(m, "Quoridor")
      .def(py::init<>())
      .def(py::pickle(
            [](const Quoridor &q) { 
                return py::make_tuple(); // 저장할 거 없음 (룰북이니까)
            },
            [](py::tuple t) { 
                return new Quoridor();   // 그냥 새 룰북 사서 줌
            }
        ))
      .def_property_readonly_static("SIZE", [](py::object) { return Quoridor::SIZE; })
      .def_property_readonly_static("ACTION_SIZE", [](py::object) { return Quoridor::ACTION_SIZE; })

      .def("get_initial_state", &Quoridor::get_initial_state)
      .def("get_valid_moves", &Quoridor::get_valid_moves)
      .def("apply_action", &Quoridor::apply_action)
      .def("check_win", &Quoridor::check_win)
      .def("render", &Quoridor::render)
      .def("get_next_state", &Quoridor::get_next_state,
           "행동 적용 후 다음 상태 반환")
      .def("change_perspective", &Quoridor::change_perspective,
           "P2 시점일 경우 보드 반전 (Canonical Form)")
      .def("get_encoded_state", &Quoridor::get_encoded_state,
           "CNN 입력용 텐서 (Flat Vector) 반환")
      .def("get_opponent", &Quoridor::get_opponent)
      .def("get_opponent_value", &Quoridor::get_opponent_value);
}