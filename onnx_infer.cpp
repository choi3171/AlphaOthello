#include "onnx_infer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
std::array<float, gomoku::kActionSize> softmax_logits(const float* logits) {
  std::array<float, gomoku::kActionSize> probs{};
  float max_logit = logits[0];
  for (int i = 1; i < gomoku::kActionSize; i++) {
    max_logit = std::max(max_logit, logits[i]);
  }

  double sum = 0.0;
  for (int i = 0; i < gomoku::kActionSize; i++) {
    probs[i] = static_cast<float>(std::exp(static_cast<double>(logits[i] - max_logit)));
    sum += probs[i];
  }
  if (sum <= 0.0) {
    probs.fill(1.0f / static_cast<float>(gomoku::kActionSize));
    return probs;
  }
  for (int i = 0; i < gomoku::kActionSize; i++) {
    probs[i] = static_cast<float>(probs[i] / sum);
  }
  return probs;
}
}  // namespace

OnnxInfer::OnnxInfer(
    const std::string& model_path,
    bool use_cuda,
    int cuda_device_id,
    int num_server_threads,
    int max_batch_size)
    : env_(ORT_LOGGING_LEVEL_WARNING, "cpp_selfplay"),
      session_options_(),
      session_(nullptr),
      num_server_threads_(std::max(1, num_server_threads)),
      max_batch_size_(std::max(1, max_batch_size)) {
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (use_cuda) {
    try {
      OrtCUDAProviderOptionsV2* cuda_options = nullptr;
      Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));
      std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)> holder(
          cuda_options, [](OrtCUDAProviderOptionsV2* p) {
            if (p != nullptr) {
              Ort::GetApi().ReleaseCUDAProviderOptions(p);
            }
          });
      const char* keys[] = {"device_id"};
      const std::string device_id_str = std::to_string(cuda_device_id);
      const char* values[] = {device_id_str.c_str()};
      Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
          cuda_options, keys, values, 1));
      Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
          session_options_, cuda_options));
      std::cout << "OnnxInfer: CUDA provider enabled (device_id=" << cuda_device_id << ")\n";
    } catch (const std::exception& e) {
      std::cerr << "OnnxInfer: failed to enable CUDA provider, fallback to CPU: " << e.what()
                << "\n";
    }
  }

  session_ = Ort::Session(env_, model_path.c_str(), session_options_);

  {
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();
    if (!shape.empty() && shape[0] > 0) {
      fixed_batch_size_ = shape[0];
      if (fixed_batch_size_ == 1 && max_batch_size_ > 1) {
        max_batch_size_ = 1;
        std::cout << "OnnxInfer: fixed batch=1 model detected, forcing nn max batch size to 1\n";
      }
    }
  }

  Ort::AllocatorWithDefaultOptions allocator;
  const size_t input_count = session_.GetInputCount();
  const size_t output_count = session_.GetOutputCount();
  input_names_holder_.reserve(input_count);
  output_names_holder_.reserve(output_count);
  input_names_.reserve(input_count);
  output_names_.reserve(output_count);

  for (size_t i = 0; i < input_count; i++) {
    auto name = session_.GetInputNameAllocated(i, allocator);
    input_names_holder_.push_back(name.get());
  }
  for (size_t i = 0; i < output_count; i++) {
    auto name = session_.GetOutputNameAllocated(i, allocator);
    output_names_holder_.push_back(name.get());
  }
  for (const auto& s : input_names_holder_) {
    input_names_.push_back(s.c_str());
  }
  for (const auto& s : output_names_holder_) {
    output_names_.push_back(s.c_str());
  }

  workers_.reserve(num_server_threads_);
  for (int i = 0; i < num_server_threads_; i++) {
    workers_.emplace_back([this]() { worker_loop(); });
  }
}

OnnxInfer::~OnnxInfer() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_workers_ = true;
  }
  queue_cv_.notify_all();
  for (auto& w : workers_) {
    if (w.joinable()) {
      w.join();
    }
  }
}

void OnnxInfer::apply_fallback(std::vector<std::shared_ptr<PendingQuery>>& batch) const {
  for (auto& q : batch) {
    q->policy.fill(1.0f / static_cast<float>(gomoku::kActionSize));
    q->value = 0.0f;
    {
      std::lock_guard<std::mutex> lock(q->mutex);
      q->ready = true;
    }
    q->cv.notify_one();
  }
}

void OnnxInfer::worker_loop() {
  while (true) {
    std::vector<std::shared_ptr<PendingQuery>> batch;
    batch.reserve(static_cast<size_t>(max_batch_size_));
    const auto collect_start = std::chrono::steady_clock::now();

    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this]() { return stop_workers_ || !query_queue_.empty(); });
      if (stop_workers_ && query_queue_.empty()) {
        return;
      }

      batch.push_back(query_queue_.front());
      query_queue_.pop_front();

      const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(1000);
      while (static_cast<int>(batch.size()) < max_batch_size_) {
        if (query_queue_.empty()) {
          if (!queue_cv_.wait_until(lock, deadline, [this]() {
                return stop_workers_ || !query_queue_.empty();
              })) {
            break;
          }
          if (query_queue_.empty()) {
            break;
          }
        }
        batch.push_back(query_queue_.front());
        query_queue_.pop_front();
      }
    }
    const auto collect_end = std::chrono::steady_clock::now();
    worker_batch_collect_ns_.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(collect_end - collect_start)
                .count()),
        std::memory_order_relaxed);
    worker_batch_count_.fetch_add(1, std::memory_order_relaxed);
    worker_item_count_.fetch_add(
        static_cast<uint64_t>(batch.size()), std::memory_order_relaxed);

    try {
      const int batch_size = static_cast<int>(batch.size());
      const auto input_build_start = std::chrono::steady_clock::now();
      std::vector<float> input_data(
          static_cast<size_t>(batch_size) * 3 * gomoku::kActionSize, 0.0f);
      for (int b = 0; b < batch_size; b++) {
        const auto& s = batch[b]->state;
        for (int i = 0; i < gomoku::kActionSize; i++) {
          const int8_t v = s[i];
          input_data[static_cast<size_t>(b) * 3 * gomoku::kActionSize + i] =
              (v == -1) ? 1.0f : 0.0f;
          input_data[static_cast<size_t>(b) * 3 * gomoku::kActionSize + gomoku::kActionSize + i] =
              (v == 0) ? 1.0f : 0.0f;
              input_data[static_cast<size_t>(b) * 3 * gomoku::kActionSize +
                     2 * gomoku::kActionSize + i] = (v == 1) ? 1.0f : 0.0f;
        }
      }
      const auto input_build_end = std::chrono::steady_clock::now();
      worker_input_build_ns_.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  input_build_end - input_build_start)
                  .count()),
          std::memory_order_relaxed);

      std::array<int64_t, 4> input_shape = {batch_size, 3, gomoku::kBoardSize, gomoku::kBoardSize};
      Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info,
          input_data.data(),
          input_data.size(),
          input_shape.data(),
          input_shape.size());

      const auto ort_run_start = std::chrono::steady_clock::now();
      auto outputs = session_.Run(
          Ort::RunOptions{nullptr},
          input_names_.data(),
          &input_tensor,
          1,
          output_names_.data(),
          output_names_.size());
      const auto ort_run_end = std::chrono::steady_clock::now();
      worker_ort_run_ns_.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(ort_run_end - ort_run_start)
                  .count()),
          std::memory_order_relaxed);

      if (outputs.size() < 2) {
        throw std::runtime_error("ONNX model must have policy and value outputs");
      }

      const float* policy_logits = outputs[0].GetTensorData<float>();
      const float* value_ptr = outputs[1].GetTensorData<float>();

      const auto post_start = std::chrono::steady_clock::now();
      for (int b = 0; b < batch_size; b++) {
        const float* row_logits = policy_logits + static_cast<size_t>(b) * gomoku::kActionSize;
        batch[b]->policy = softmax_logits(row_logits);
        batch[b]->value = value_ptr[b];
        {
          std::lock_guard<std::mutex> lock(batch[b]->mutex);
          batch[b]->ready = true;
        }
        batch[b]->cv.notify_one();
      }
      const auto post_end = std::chrono::steady_clock::now();
      worker_postprocess_ns_.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(post_end - post_start).count()),
          std::memory_order_relaxed);
    } catch (const std::exception& e) {
      std::cerr << "OnnxInfer worker inference failure: " << e.what() << "\n";
      apply_fallback(batch);
    }
  }
}

std::pair<std::array<float, gomoku::kActionSize>, float> OnnxInfer::infer(
    const gomoku::Board& canonical_state) {
  const auto wait_start = std::chrono::steady_clock::now();
  auto q = std::make_shared<PendingQuery>();
  q->state = canonical_state;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    query_queue_.push_back(q);
  }
  queue_cv_.notify_one();

  std::unique_lock<std::mutex> q_lock(q->mutex);
  q->cv.wait(q_lock, [&q]() { return q->ready; });
  const auto wait_end = std::chrono::steady_clock::now();
  query_count_.fetch_add(1, std::memory_order_relaxed);
  queue_wait_ns_.fetch_add(
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(wait_end - wait_start).count()),
      std::memory_order_relaxed);
  return {q->policy, q->value};
}

OnnxInfer::InferProfile OnnxInfer::profile_snapshot() const {
  InferProfile p;
  p.query_count = query_count_.load(std::memory_order_relaxed);
  p.queue_wait_ns = queue_wait_ns_.load(std::memory_order_relaxed);
  p.worker_batch_count = worker_batch_count_.load(std::memory_order_relaxed);
  p.worker_item_count = worker_item_count_.load(std::memory_order_relaxed);
  p.worker_batch_collect_ns = worker_batch_collect_ns_.load(std::memory_order_relaxed);
  p.worker_input_build_ns = worker_input_build_ns_.load(std::memory_order_relaxed);
  p.worker_ort_run_ns = worker_ort_run_ns_.load(std::memory_order_relaxed);
  p.worker_postprocess_ns = worker_postprocess_ns_.load(std::memory_order_relaxed);
  return p;
}
