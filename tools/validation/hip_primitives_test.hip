// Small, independent checks of the production HIP warp and block primitives.
// Compile only unless the caller has assigned one GPU and its expected PCI ID.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define WS 32
#define TPB 512
#define CS 16384
#include "include/hip_warp32.h"
#include "include/prefix_sum.h"
#include "include/sum_reduction.h"
#include "include/max_scan.h"
#include "include/max_reduction.h"

namespace {

constexpr int kWidths[] = {0, 1, 2, 4, 8, 16, 32};
constexpr int kShuffleOps = 35;
constexpr int kVoteOps = 9;
constexpr int kSyncRounds = 16;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr unsigned kSparseMask = 0x00ff00ffu;
constexpr size_t kDeviceByteLimit = 256 * 1024;
const char* kTypes[] = {"int", "uint", "longlong", "ulonglong", "float", "double"};
const char* kShuffleNames[] = {"shfl", "up", "xor", "legacy_up", "legacy_xor"};
const char* kBlockNames[] = {"prefix_sum", "sum_reduction", "max_scan", "max_reduction"};
const char* kVoteNames[] = {"ballot_pattern", "any_pattern", "all_pattern", "ballot_false",
                          "ballot_true", "any_false", "any_true", "all_false", "all_true"};
const char* kVoteModes[] = {"all_warps", "high_half_only", "sparse_mask"};

std::string quoted(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
    else if (c < 0x20) out << "\\u" << std::hex << std::setw(4)
                            << std::setfill('0') << static_cast<unsigned>(c);
    else out << static_cast<char>(c);
  }
  out << '"';
  return out.str();
}

std::string pci_address(const std::string& value) {
  unsigned domain, bus, device, function;
  char trailing;
  if (std::sscanf(value.c_str(), "%x:%x:%x.%x%c", &domain, &bus, &device,
                  &function, &trailing) != 4 || domain > 0xffff || bus > 0xff ||
      device > 0x1f || function > 7) {
    throw std::runtime_error("Invalid PCI address: " + value);
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(4) << domain << ':'
      << std::setw(2) << bus << ':' << std::setw(2) << device << '.' << function;
  return out.str();
}

void checked(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status) +
                             " (" + std::to_string(static_cast<int>(status)) + ")");
  }
  return;
}

struct Case {
  std::string name;
  size_t expected = 0;
  size_t compared = 0;
  size_t failures = 0;
  std::string first_failure;
  bool passed() const { return expected > 0 && compared == expected && failures == 0; }
};

struct Report {
  std::vector<Case> cases;
  std::map<std::string, size_t> index;
  std::vector<std::string> errors;
  size_t device_bytes = 0;
  size_t peak_device_bytes = 0;
  size_t kernel_count = 0;

  void plan(const std::string& name, size_t expected) {
    if (expected == 0 || !index.emplace(name, cases.size()).second) {
      throw std::runtime_error("Invalid or duplicate case: " + name);
    }
    cases.push_back({name, expected, 0, 0, ""});
    return;
  }

  template <typename T>
  void compare(const std::string& name, size_t item, T actual, T expected) {
    Case& entry = cases.at(index.at(name));
    ++entry.compared;
    if (actual != expected) {
      ++entry.failures;
      if (entry.first_failure.empty()) {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<T>::max_digits10)
            << "item=" << item << ", expected=" << expected << ", actual=" << actual;
        entry.first_failure = out.str();
      }
    }
    return;
  }
};

struct DeviceBuffer {
  Report& report;
  void* pointer = nullptr;
  size_t bytes;
  DeviceBuffer(Report& state, size_t count) : report(state), bytes(count) {
    if (bytes == 0 || bytes + report.device_bytes > kDeviceByteLimit) {
      throw std::runtime_error("Device allocation exceeds the test memory limit");
    }
    checked(hipMalloc(&pointer, bytes), "hipMalloc");
    report.device_bytes += bytes;
    report.peak_device_bytes = std::max(report.peak_device_bytes, report.device_bytes);
  }
  ~DeviceBuffer() {
    if (pointer) {
      const hipError_t status = hipFree(pointer);
      if (status != hipSuccess) {
        report.errors.push_back(std::string("hipFree: ") + hipGetErrorString(status));
      }
      report.device_bytes -= bytes;
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  template <typename T> T* as() { return static_cast<T*>(pointer); }
};

std::string shuffle_case(const std::string& type, int width, int operation) {
  return type + "." + kShuffleNames[operation] + "." +
         (width == 0 ? "default32" : "width" + std::to_string(width));
}

void plan_cases(Report& report) {
  for (const char* type : kTypes) {
    for (int width : kWidths) {
      for (int op = 0; op < 5; ++op) report.plan(shuffle_case(type, width, op), TPB);
    }
    for (const char* op : kBlockNames) report.plan(std::string(type) + "." + op, TPB);
  }
  for (int mode = 0; mode < 3; ++mode) {
    const size_t active = mode == 0 ? TPB : mode == 1 ? WS : TPB / 2;
    for (const char* op : kVoteNames) {
      report.plan(std::string("votes.") + kVoteModes[mode] + "." + op, active);
    }
    if (mode != 0) {
      report.plan(std::string("votes.") + kVoteModes[mode] + ".inactive_untouched",
                  (TPB - active) * kVoteOps);
    }
  }
  report.plan("syncwarp.all_warps.shared_dependency", TPB * kSyncRounds);
  report.plan("syncwarp.high_half_only.shared_dependency", WS * kSyncRounds);
  report.plan("syncwarp.high_half_only.inactive_untouched", (TPB - WS) * kSyncRounds);
  return;
}

template <typename T>
__global__ __launch_bounds__(TPB) void shuffle_kernel(const T* input, T* output) {
  const int tid = threadIdx.x;
  const T value = input[tid];
  int column = 0;
  for (int width_index = 0; width_index < 7; ++width_index) {
    const int width = width_index == 0 ? 32 : 1 << (width_index - 1);
    const int source_lane = width - 1;
    const int delta = width > 1 ? width / 2 : 1;
    const int lane_mask = width > 1 ? width / 2 : 0;
    if (width_index == 0) {
      output[column++ * TPB + tid] = pfpl_shfl_sync(kFullMask, value, source_lane);
      output[column++ * TPB + tid] = pfpl_shfl_up_sync(kFullMask, value, delta);
      output[column++ * TPB + tid] = pfpl_shfl_xor_sync(kFullMask, value, lane_mask);
      output[column++ * TPB + tid] = pfpl_shfl_up(value, delta);
      output[column++ * TPB + tid] = pfpl_shfl_xor(value, lane_mask);
    } else {
      output[column++ * TPB + tid] = pfpl_shfl_sync(kFullMask, value, source_lane, width);
      output[column++ * TPB + tid] = pfpl_shfl_up_sync(kFullMask, value, delta, width);
      output[column++ * TPB + tid] = pfpl_shfl_xor_sync(kFullMask, value, lane_mask, width);
      output[column++ * TPB + tid] = pfpl_shfl_up(value, delta, width);
      output[column++ * TPB + tid] = pfpl_shfl_xor(value, lane_mask, width);
    }
  }
  return;
}

template <typename T, int Operation>
__global__ __launch_bounds__(TPB) void block_kernel(const T* input, T* output) {
  // The production routines use up to WS scratch elements, including unused
  // lanes of the first logical warp. Initialize every element deterministically.
  __shared__ T scratch[WS];
  if (threadIdx.x < WS) scratch[threadIdx.x] = static_cast<T>(17);
  __syncthreads();
  const T value = input[threadIdx.x];
  if constexpr (Operation == 0) output[threadIdx.x] = block_prefix_sum(value, scratch);
  if constexpr (Operation == 1) output[threadIdx.x] = block_sum_reduction(value, scratch);
  if constexpr (Operation == 2) output[threadIdx.x] = block_max_scan(value, scratch);
  if constexpr (Operation == 3) output[threadIdx.x] = block_max_reduction(value, scratch);
  return;
}

__global__ __launch_bounds__(TPB) void vote_kernel(const int* predicate, unsigned* output, int mode) {
  const int tid = threadIdx.x;
  const int lane = tid % WS;
  const unsigned mask = mode == 2 ? kSparseMask : kFullMask;
  if (mode == 1 && tid < TPB - WS) return;
  if ((mask & (1u << lane)) == 0) return;
  output[0 * TPB + tid] = pfpl_ballot_sync(mask, predicate[tid]);
  output[1 * TPB + tid] = pfpl_any_sync(mask, predicate[tid]);
  output[2 * TPB + tid] = pfpl_all_sync(mask, predicate[tid]);
  output[3 * TPB + tid] = pfpl_ballot_sync(mask, 0);
  output[4 * TPB + tid] = pfpl_ballot_sync(mask, 1);
  output[5 * TPB + tid] = pfpl_any_sync(mask, 0);
  output[6 * TPB + tid] = pfpl_any_sync(mask, 1);
  output[7 * TPB + tid] = pfpl_all_sync(mask, 0);
  output[8 * TPB + tid] = pfpl_all_sync(mask, 1);
  return;
}

__global__ __launch_bounds__(TPB) void sync_kernel(const int* input, int* output, bool high_half_only) {
  __shared__ volatile int values[TPB];
  const int tid = threadIdx.x;
  if (high_half_only && tid < TPB - WS) return;
  const int base = tid / WS * WS;
  const int lane = tid % WS;
  for (int round = 0; round < kSyncRounds; ++round) {
    values[tid] = input[tid] + round * 8192;
    pfpl_syncwarp();
    output[round * TPB + tid] = values[base + ((lane + 7 * round + 1) % WS)];
    // Prevent a fast lane from overwriting the previous round's shared value.
    pfpl_syncwarp();
  }
  return;
}

void finish_kernel(Report& report, const char* operation) {
  checked(hipGetLastError(), operation);
  checked(hipDeviceSynchronize(), "hipDeviceSynchronize");
  ++report.kernel_count;
  return;
}

template <typename T>
std::vector<T> shuffle_input() {
  std::vector<T> values(TPB);
  for (int tid = 0; tid < TPB; ++tid) {
    const long long code = 101 * (tid / WS + 1) + 3 * (tid % WS) + 1;
    if constexpr (std::is_same<T, long long>::value) {
      values[tid] = -(1ll << 45) - code * 1048577ll;
    } else if constexpr (std::is_same<T, unsigned long long>::value) {
      values[tid] = (1ull << 57) + static_cast<unsigned long long>(code) * 1048577ull;
    } else if constexpr (std::is_same<T, double>::value) {
      values[tid] = -static_cast<double>(code) - static_cast<double>(tid + 1) / 1048576.0;
    } else if constexpr (std::is_same<T, float>::value) {
      values[tid] = -static_cast<float>(code) - static_cast<float>(tid % 7) / 8.0f;
    } else if constexpr (std::is_signed<T>::value) {
      values[tid] = static_cast<T>(-code);
    } else {
      values[tid] = static_cast<T>(code + 0x80000000ull);
    }
  }
  return values;
}

template <typename T>
void test_shuffle(Report& report, const std::string& type) {
  const std::vector<T> input = shuffle_input<T>();
  std::vector<T> output(kShuffleOps * TPB);
  DeviceBuffer device_input(report, input.size() * sizeof(T));
  DeviceBuffer device_output(report, output.size() * sizeof(T));
  checked(hipMemcpy(device_input.pointer, input.data(), device_input.bytes, hipMemcpyHostToDevice),
          "shuffle input H2D");
  checked(hipMemset(device_output.pointer, 0xa5, device_output.bytes), "shuffle output sentinel");
  shuffle_kernel<T><<<1, TPB>>>(device_input.as<T>(), device_output.as<T>());
  finish_kernel(report, "shuffle_kernel launch");
  checked(hipMemcpy(output.data(), device_output.pointer, device_output.bytes, hipMemcpyDeviceToHost),
          "shuffle output D2H");
  int column = 0;
  for (int width_setting : kWidths) {
    const int width = width_setting == 0 ? WS : width_setting;
    const int delta = width > 1 ? width / 2 : 1;
    const int xor_mask = width > 1 ? width / 2 : 0;
    for (int operation = 0; operation < 5; ++operation, ++column) {
      for (int tid = 0; tid < TPB; ++tid) {
        const int base = tid / width * width;
        const int lane = tid % width;
        int source = base + width - 1;
        if (operation == 1 || operation == 3) source = lane >= delta ? tid - delta : tid;
        if (operation == 2 || operation == 4) source = base + (lane ^ xor_mask);
        report.compare(shuffle_case(type, width_setting, operation), tid,
                        output[column * TPB + tid], input[source]);
      }
    }
  }
  return;
}

template <typename T>
void test_blocks(Report& report, const std::string& type) {
  std::vector<T> input(TPB), expected(TPB), output(TPB);
  for (int tid = 0; tid < TPB; ++tid) {
    // Distinct adjacent logical warps, nonmonotone lanes and all-negative
    // signed/floating inputs expose accidental mixing or a zero identity.
    const int warp_rank = (tid / WS * 5) % (TPB / WS);
    const int scale = std::is_signed<T>::value ? TPB / WS - warp_rank : warp_rank + 1;
    const int value = 1 + scale * 23 + ((tid * 19 + 7) % 31);
    input[tid] = static_cast<T>(std::is_signed<T>::value ? -value : value);
  }
  DeviceBuffer device_input(report, input.size() * sizeof(T));
  DeviceBuffer device_output(report, output.size() * sizeof(T));
  checked(hipMemcpy(device_input.pointer, input.data(), device_input.bytes, hipMemcpyHostToDevice),
          "block input H2D");
  for (int operation = 0; operation < 4; ++operation) {
    T running = (operation < 2) ? static_cast<T>(0) : input.front();
    for (int tid = 0; tid < TPB; ++tid) {
      if (operation < 2) running = static_cast<T>(running + input[tid]);
      else running = std::max(running, input[tid]);
      expected[tid] = running;
    }
    if (operation == 1 || operation == 3) std::fill(expected.begin(), expected.end(), running);
    checked(hipMemset(device_output.pointer, 0xa5, device_output.bytes), "block output sentinel");
    if (operation == 0) block_kernel<T, 0><<<1, TPB>>>(device_input.as<T>(), device_output.as<T>());
    if (operation == 1) block_kernel<T, 1><<<1, TPB>>>(device_input.as<T>(), device_output.as<T>());
    if (operation == 2) block_kernel<T, 2><<<1, TPB>>>(device_input.as<T>(), device_output.as<T>());
    if (operation == 3) block_kernel<T, 3><<<1, TPB>>>(device_input.as<T>(), device_output.as<T>());
    finish_kernel(report, "block_kernel launch");
    checked(hipMemcpy(output.data(), device_output.pointer, device_output.bytes, hipMemcpyDeviceToHost),
            "block output D2H");
    for (int tid = 0; tid < TPB; ++tid) {
      report.compare(type + "." + kBlockNames[operation], tid, output[tid], expected[tid]);
    }
  }
  return;
}

void test_votes(Report& report) {
  std::vector<int> input(TPB);
  std::vector<unsigned> output(TPB * kVoteOps);
  for (int tid = 0; tid < TPB; ++tid) {
    input[tid] = ((tid % WS * 5 + tid / WS * 3) % 7) < (tid / WS % 5 + 1);
  }
  DeviceBuffer device_input(report, input.size() * sizeof(int));
  DeviceBuffer device_output(report, output.size() * sizeof(unsigned));
  checked(hipMemcpy(device_input.pointer, input.data(), device_input.bytes, hipMemcpyHostToDevice),
          "vote input H2D");
  for (int mode = 0; mode < 3; ++mode) {
    const unsigned mask = mode == 2 ? kSparseMask : kFullMask;
    checked(hipMemset(device_output.pointer, 0xa5, device_output.bytes), "vote output sentinel");
    vote_kernel<<<1, TPB>>>(device_input.as<int>(), device_output.as<unsigned>(), mode);
    finish_kernel(report, "vote_kernel launch");
    checked(hipMemcpy(output.data(), device_output.pointer, device_output.bytes, hipMemcpyDeviceToHost),
            "vote output D2H");
    const std::string prefix = std::string("votes.") + kVoteModes[mode] + ".";
    for (int warp = 0; warp < TPB / WS; ++warp) {
      unsigned ballot = 0;
      for (int lane = 0; lane < WS; ++lane) {
        if ((mask & (1u << lane)) && input[warp * WS + lane]) ballot |= 1u << lane;
      }
      const unsigned expected[] = {ballot, ballot != 0, ballot == mask, 0, mask, 0, 1, 0, 1};
      for (int lane = 0; lane < WS; ++lane) {
        const int tid = warp * WS + lane;
        const bool active = (mode != 1 || tid >= TPB - WS) && ((mask & (1u << lane)) != 0);
        for (int operation = 0; operation < kVoteOps; ++operation) {
          if (active) {
            report.compare(prefix + kVoteNames[operation], tid,
                            output[operation * TPB + tid], expected[operation]);
          } else {
            report.compare(prefix + "inactive_untouched", operation * TPB + tid,
                            output[operation * TPB + tid], 0xa5a5a5a5u);
          }
        }
      }
    }
  }
  return;
}

void test_sync(Report& report) {
  std::vector<int> input(TPB), output(TPB * kSyncRounds);
  for (int tid = 0; tid < TPB; ++tid) input[tid] = 1009 * (tid / WS + 1) + 13 * (tid % WS);
  DeviceBuffer device_input(report, input.size() * sizeof(int));
  DeviceBuffer device_output(report, output.size() * sizeof(int));
  checked(hipMemcpy(device_input.pointer, input.data(), device_input.bytes, hipMemcpyHostToDevice),
          "sync input H2D");
  for (int high_only = 0; high_only < 2; ++high_only) {
    // -1 is not generated by any dependency, and remains in inactive lanes.
    checked(hipMemset(device_output.pointer, 0xff, device_output.bytes), "sync output sentinel");
    sync_kernel<<<1, TPB>>>(device_input.as<int>(), device_output.as<int>(), high_only != 0);
    finish_kernel(report, "sync_kernel launch");
    checked(hipMemcpy(output.data(), device_output.pointer, device_output.bytes, hipMemcpyDeviceToHost),
            "sync output D2H");
    const std::string prefix = high_only ? "syncwarp.high_half_only." : "syncwarp.all_warps.";
    for (int round = 0; round < kSyncRounds; ++round) {
      for (int tid = 0; tid < TPB; ++tid) {
        const int item = round * TPB + tid;
        if (high_only && tid < TPB - WS) {
          report.compare(prefix + "inactive_untouched", item, output[item], -1);
        } else {
          const int source = tid / WS * WS + (tid % WS + 7 * round + 1) % WS;
          report.compare(prefix + "shared_dependency", item, output[item], input[source] + round * 8192);
        }
      }
    }
  }
  return;
}

template <typename T>
void test_type(Report& report, const char* name) {
  test_shuffle<T>(report, name);
  test_blocks<T>(report, name);
  return;
}

std::string environment(const char* name) {
  const char* value = std::getenv(name);
  return value ? quoted(value) : "null";
}

}  // namespace

int main(int argc, char** argv) {
  Report report;
  std::string expected_pci, actual_pci;
  int count = -1, device = -1;
  hipDeviceProp_t properties{};
  bool properties_read = false, pci_matches = false;
  try {
    plan_cases(report);
    if (argc != 3 || std::string(argv[1]) != "--expected-pci") {
      throw std::runtime_error("Usage: hip_primitives_test --expected-pci DDDD:BB:DD.F");
    }
    expected_pci = pci_address(argv[2]);
    checked(hipGetDeviceCount(&count), "hipGetDeviceCount");
    if (count != 1) throw std::runtime_error("Exactly one visible GPU is required; observed " + std::to_string(count));
    checked(hipGetDevice(&device), "hipGetDevice");
    if (device != 0) throw std::runtime_error("Selected logical device must be zero");
    checked(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    properties_read = true;
    char address[64]{};
    checked(hipDeviceGetPCIBusId(address, sizeof(address), device), "hipDeviceGetPCIBusId");
    actual_pci = pci_address(address);
    if (actual_pci != expected_pci) throw std::runtime_error("PCI mismatch: expected " + expected_pci + ", observed " + actual_pci);
    pci_matches = true;
    if (properties.warpSize != 64) throw std::runtime_error("This suite requires wave64 to exercise both physical half-waves");
    if (properties.maxThreadsPerBlock < TPB || properties.sharedMemPerBlock < TPB * sizeof(int)) {
      throw std::runtime_error("Device cannot support the 512-thread test block");
    }
    test_type<int>(report, "int");
    test_type<unsigned int>(report, "uint");
    test_type<long long>(report, "longlong");
    test_type<unsigned long long>(report, "ulonglong");
    test_type<float>(report, "float");
    test_type<double>(report, "double");
    test_votes(report);
    test_sync(report);
  } catch (const std::exception& error) {
    report.errors.push_back(error.what());
  }
  size_t failed_cases = 0, compared = 0, failed_comparisons = 0;
  for (const Case& entry : report.cases) {
    failed_cases += !entry.passed();
    compared += entry.compared;
    failed_comparisons += entry.failures;
  }
  const bool passed = report.errors.empty() && !report.cases.empty() && compared > 0 &&
                      failed_cases == 0 && report.kernel_count == 35;
  if (report.kernel_count != 35) report.errors.push_back("Expected 35 completed kernels; observed " + std::to_string(report.kernel_count));
  std::cout << std::boolalpha << "{\n  \"status\": " << quoted(passed ? "passed" : "failed")
            << ",\n  \"case_count\": " << report.cases.size()
            << ",\n  \"failures\": " << failed_cases
            << ",\n  \"comparison_count\": " << compared
            << ",\n  \"failed_comparisons\": " << failed_comparisons
            << ",\n  \"kernel_count\": " << report.kernel_count
            << ",\n  \"grid\": [1, " << TPB << ']'
            << ",\n  \"logical_warp_size\": " << WS
            << ",\n  \"logical_warps_per_block\": " << TPB / WS
            << ",\n  \"peak_device_buffer_bytes\": " << report.peak_device_bytes
            << ",\n  \"device_buffer_limit_bytes\": " << kDeviceByteLimit
            << ",\n  \"expected_pci\": " << quoted(expected_pci)
            << ",\n  \"actual_pci\": " << quoted(actual_pci)
            << ",\n  \"pci\": " << quoted(actual_pci)
            << ",\n  \"pci_matches\": " << pci_matches
            << ",\n  \"visible_device_count\": " << count
            << ",\n  \"logical_device\": " << device
            << ",\n  \"hip_visible_devices\": " << environment("HIP_VISIBLE_DEVICES")
            << ",\n  \"rocr_visible_devices\": " << environment("ROCR_VISIBLE_DEVICES")
            << ",\n  \"device\": ";
  if (properties_read) {
    std::cout << "{\"name\": " << quoted(properties.name)
              << ", \"gcn_arch_name\": " << quoted(properties.gcnArchName)
              << ", \"warp_size\": " << properties.warpSize << '}';
  } else std::cout << "null";
  std::cout << ",\n  \"cases\": [";
  for (size_t index = 0; index < report.cases.size(); ++index) {
    const Case& entry = report.cases[index];
    if (index) std::cout << ',';
    std::cout << "\n    {\"name\": " << quoted(entry.name)
              << ", \"status\": " << quoted(entry.passed() ? "passed" : entry.compared == 0 ? "not_run" : "failed")
              << ", \"expected_comparisons\": " << entry.expected
              << ", \"comparison_count\": " << entry.compared
              << ", \"failures\": " << entry.failures
              << ", \"first_failure\": " << quoted(entry.first_failure) << '}';
  }
  std::cout << "\n  ],\n  \"errors\": [";
  for (size_t index = 0; index < report.errors.size(); ++index) {
    if (index) std::cout << ", ";
    std::cout << quoted(report.errors[index]);
  }
  std::cout << "]\n}\n";
  return passed ? 0 : 1;
}
