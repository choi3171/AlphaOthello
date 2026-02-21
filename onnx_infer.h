#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "gomoku.h"

class OnnxInfer {
 public:
  struct InferProfile {
    uint64_t query_count = 0;
    uint64_t queue_wait_ns = 0;
    uint64_t worker_batch_count = 0;
    uint64_t worker_item_count = 0;
    uint64_t worker_batch_collect_ns = 0;
    uint64_t worker_input_build_ns = 0;
    uint64_t worker_ort_run_ns = 0;
    uint64_t worker_postprocess_ns = 0;
  };

  OnnxInfer(
      const std::string& model_path,
      bool use_cuda,
      int cuda_device_id,
      int num_server_threads,
      int max_batch_size);
  ~OnnxInfer();

  std::pair<std::array<float, gomoku::kActionSize>, float> infer(
      const gomoku::Board& canonical_state);
  InferProfile profile_snapshot() const;

 private:
  struct PendingQuery {
    gomoku::Board state{};
    std::array<float, gomoku::kActionSize> policy{};
    float value = 0.0f;
    bool ready = false;
    std::mutex mutex;
    std::condition_variable cv;
  };

  void worker_loop();
  void apply_fallback(std::vector<std::shared_ptr<PendingQuery>>& batch) const;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  std::vector<std::string> input_names_holder_;
  std::vector<std::string> output_names_holder_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  int num_server_threads_;
  int max_batch_size_;
  int64_t fixed_batch_size_ = -1;
  std::deque<std::shared_ptr<PendingQuery>> query_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  bool stop_workers_ = false;
  std::vector<std::thread> workers_;

  std::atomic<uint64_t> query_count_{0};
  std::atomic<uint64_t> queue_wait_ns_{0};
  std::atomic<uint64_t> worker_batch_count_{0};
  std::atomic<uint64_t> worker_item_count_{0};
  std::atomic<uint64_t> worker_batch_collect_ns_{0};
  std::atomic<uint64_t> worker_input_build_ns_{0};
  std::atomic<uint64_t> worker_ort_run_ns_{0};
  std::atomic<uint64_t> worker_postprocess_ns_{0};
};
