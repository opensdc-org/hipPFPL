// Independent CPU/GPU component comparisons. This program launches GPU work
// only after verifying one visible device and the explicitly assigned PCI ID.
#define NDEBUG
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using byte = unsigned char;
static constexpr int CS = 16384;
static constexpr int TPB = 512;
#define WS 32

#include "include/macros.h"
#include "include/sum_reduction.h"
#include "include/max_scan.h"
#include "include/prefix_sum.h"
#include "components/h_DIFFNB_4.h"
#include "components/h_DIFFNB_8.h"
#include "components/h_BIT_4.h"
#include "components/h_BIT_8.h"
#include "components/h_RZE_1.h"
#include "components/d_DIFFNB_4.h"
#include "components/d_DIFFNB_8.h"
#include "components/d_BIT_4.h"
#include "components/d_BIT_8.h"
#include "components/d_RZE_1.h"

namespace {

constexpr int kGuard = 64;
constexpr int kStride = CS + 2 * kGuard;
constexpr byte kCanary = 0xa5;
constexpr byte kPoison = 0xcd;
constexpr std::uint32_t kSeed = 0x5046504c;
enum Component { DIFF4, DIFF8, BIT4, BIT8, RZE1 };
const char* const kComponents[] = {"DIFFNB_4", "DIFFNB_8", "BIT_4", "BIT_8", "RZE_1"};

struct alignas(16) HostBuffer {
  std::array<byte, kStride> bytes;
  HostBuffer() {
    bytes.fill(kCanary);
    std::fill(bytes.begin() + kGuard, bytes.begin() + kGuard + CS, kPoison);
  }
  byte* data() { return bytes.data() + kGuard; }
  const byte* data() const { return bytes.data() + kGuard; }
  bool guards_ok() const {
    for (int i = 0; i < kGuard; ++i) {
      if (bytes[i] != kCanary || bytes[kGuard + CS + i] != kCanary) return false;
    }
    return true;
  }
};

struct DeviceResult {
  int success[TPB];
  int size[TPB];
  int guard_errors[TPB];
};

struct DeviceBuffers {
  byte* input = nullptr;
  byte* output = nullptr;
  DeviceResult* result = nullptr;
  bool pending = false;
};

struct Operation {
  HostBuffer output;
  DeviceResult result{};
  bool guards_ok = false;
};

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

std::string pci_address(const std::string& text) {
  unsigned domain, bus, device, function;
  char trailing;
  if (std::sscanf(text.c_str(), "%x:%x:%x.%x%c", &domain, &bus, &device,
                  &function, &trailing) != 4 || domain > 0xffff || bus > 0xff ||
      device > 0x1f || function > 7) throw std::runtime_error("Invalid PCI address");
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(4) << domain << ':'
      << std::setw(2) << bus << ':' << std::setw(2) << device << '.' << function;
  return out.str();
}

std::string hip_error(hipError_t code, const char* operation) {
  return std::string(operation) + ": " + hipGetErrorString(code) +
         " (" + std::to_string(static_cast<int>(code)) + ")";
}

void checked(hipError_t code, const char* operation) {
  if (code != hipSuccess) throw std::runtime_error(hip_error(code, operation));
  return;
}

void cleanup_checked(hipError_t code, const char* operation,
                     std::vector<std::string>& errors) {
  if (code != hipSuccess) errors.push_back(hip_error(code, operation));
  return;
}

bool cpu_encode(Component component, int& size, HostBuffer& input, HostBuffer& output) {
  switch (component) {
    case DIFF4: return h_DIFFNB_4(size, input.data(), output.data());
    case DIFF8: return h_DIFFNB_8(size, input.data(), output.data());
    case BIT4: return h_BIT_4(size, input.data(), output.data());
    case BIT8: return h_BIT_8(size, input.data(), output.data());
    case RZE1: return h_RZE_1(size, input.data(), output.data());
  }
  throw std::runtime_error("Unknown CPU component");
}

void cpu_decode(Component component, int& size, HostBuffer& input, HostBuffer& output) {
  switch (component) {
    case DIFF4: h_iDIFFNB_4(size, input.data(), output.data()); break;
    case DIFF8: h_iDIFFNB_8(size, input.data(), output.data()); break;
    case BIT4: h_iBIT_4(size, input.data(), output.data()); break;
    case BIT8: h_iBIT_8(size, input.data(), output.data()); break;
    case RZE1: h_iRZE_1(size, input.data(), output.data()); break;
  }
  return;
}

__global__ __launch_bounds__(TPB) void exercise_component(Component component, bool inverse, int input_size,
                                   const byte* input, byte* output, DeviceResult* result) {
  // Match the production component address space and alignment. Guards are
  // outside each entire CS-byte allocation, never inside component scratch.
  __shared__ unsigned long long storage[3 * kStride / sizeof(unsigned long long)];
  byte* const raw = reinterpret_cast<byte*>(storage);
  byte* const in = raw + kGuard;
  byte* const out = raw + kStride + kGuard;
  byte* const temp = raw + 2 * kStride + kGuard;
  const int tid = threadIdx.x;
  for (int i = tid; i < 3 * kStride; i += TPB) {
    const int offset = i % kStride;
    raw[i] = (offset < kGuard || offset >= kGuard + CS) ? kCanary : kPoison;
  }
  __syncthreads();
  for (int i = tid; i < input_size; i += TPB) in[i] = input[kGuard + i];
  __syncthreads();

  int size = input_size;
  bool success = true;
  if (inverse) {
    switch (component) {
      case DIFF4: d_iDIFFNB_4(size, in, out, temp); break;
      case DIFF8: d_iDIFFNB_8(size, in, out, temp); break;
      case BIT4: d_iBIT_4(size, in, out, temp); break;
      case BIT8: d_iBIT_8(size, in, out, temp); break;
      case RZE1: d_iRZE_1(size, in, out, temp); break;
    }
  } else {
    switch (component) {
      case DIFF4: success = d_DIFFNB_4(size, in, out, temp); break;
      case DIFF8: success = d_DIFFNB_8(size, in, out, temp); break;
      case BIT4: success = d_BIT_4(size, in, out, temp); break;
      case BIT8: success = d_BIT_8(size, in, out, temp); break;
      case RZE1: success = d_RZE_1(size, in, out, temp); break;
    }
  }
  __syncthreads();
  result->success[tid] = success;
  result->size[tid] = success ? size : -1;
  int guard_errors = 0;
  for (int i = tid; i < 3 * kStride; i += TPB) {
    const int offset = i % kStride;
    if ((offset < kGuard || offset >= kGuard + CS) && raw[i] != kCanary) ++guard_errors;
  }
  result->guard_errors[tid] = guard_errors;
  // A false encoder return does not define output bytes or an output size.
  if (success && size >= 0 && size <= CS) {
    for (int i = tid; i < size; i += TPB) output[kGuard + i] = out[i];
  }
  return;
}

Operation gpu_operation(DeviceBuffers& device, Component component, bool inverse,
                        int input_size, const HostBuffer& input) {
  if (input_size < 0 || input_size > CS) throw std::runtime_error("Invalid component input size");
  Operation operation;
  checked(hipMemcpy(device.input, input.bytes.data(), kStride, hipMemcpyHostToDevice),
          "component input H2D");
  checked(hipMemcpy(device.output, operation.output.bytes.data(), kStride, hipMemcpyHostToDevice),
          "component output guards H2D");
  checked(hipMemset(device.result, 0xff, sizeof(DeviceResult)), "component result memset");
  device.pending = true;
  exercise_component<<<1, TPB>>>(component, inverse, input_size, device.input,
                                 device.output, device.result);
  checked(hipGetLastError(), "component kernel launch");
  checked(hipDeviceSynchronize(), "component kernel synchronize");
  device.pending = false;
  checked(hipMemcpy(&operation.result, device.result, sizeof(DeviceResult), hipMemcpyDeviceToHost),
          "component metadata D2H");
  // Only copy defined payload when every lane reported a valid successful size.
  bool defined = true;
  for (int lane = 0; lane < TPB; ++lane) {
    if (operation.result.success[lane] != 1 || operation.result.size[lane] < 0 ||
        operation.result.size[lane] > CS ||
        operation.result.size[lane] != operation.result.size[0]) defined = false;
  }
  if (defined) {
    checked(hipMemcpy(operation.output.data(), device.output + kGuard,
                      operation.result.size[0], hipMemcpyDeviceToHost), "component payload D2H");
  }
  checked(hipMemcpy(operation.output.bytes.data(), device.output, kGuard, hipMemcpyDeviceToHost),
          "component leading output guard D2H");
  checked(hipMemcpy(operation.output.bytes.data() + kGuard + CS,
                    device.output + kGuard + CS, kGuard, hipMemcpyDeviceToHost),
          "component trailing output guard D2H");
  operation.guards_ok = operation.output.guards_ok();
  for (int lane = 0; lane < TPB; ++lane) {
    if (operation.result.guard_errors[lane] != 0) operation.guards_ok = false;
  }
  return operation;
}

HostBuffer input_pattern(int length, int pattern) {
  HostBuffer input;
  static constexpr byte periodic[] = {0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 1, 1, 2, 2, 3, 3};
  std::uint32_t state = kSeed ^ static_cast<std::uint32_t>(length);
  for (int i = 0; i < length; ++i) {
    if (pattern == 0) input.data()[i] = 0;
    else if (pattern == 1) input.data()[i] = periodic[i % sizeof(periodic)];
    else {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      input.data()[i] = static_cast<byte>(state);
    }
  }
  return input;
}

int first_difference(const byte* a, const byte* b, int length) {
  for (int i = 0; i < length; ++i) if (a[i] != b[i]) return i;
  return -1;
}

bool lane_status_matches(const Operation& operation, bool success) {
  for (int lane = 0; lane < TPB; ++lane) {
    if (operation.result.success[lane] != static_cast<int>(success)) return false;
  }
  return true;
}

bool lane_sizes_match(const Operation& operation, int size) {
  for (int lane = 0; lane < TPB; ++lane) {
    if (operation.result.size[lane] != size) return false;
  }
  return true;
}

std::string run_case(DeviceBuffers& device, Component component, int length,
                     int pattern, bool& passed, int& decode_count) {
  const char* const patterns[] = {"zeros", "periodic", "seeded_bytes"};
  const HostBuffer original = input_pattern(length, pattern);
  HostBuffer cpu_input = original, encoded;
  int cpu_size = length;
  const bool cpu_success = cpu_encode(component, cpu_size, cpu_input, encoded);
  const Operation gpu = gpu_operation(device, component, false, length, original);
  const bool status_match = lane_status_matches(gpu, cpu_success);
  const bool cpu_guards = cpu_input.guards_ok() && encoded.guards_ok();
  passed = status_match && cpu_guards && gpu.guards_ok;
  const bool compare_payload = cpu_success && status_match && cpu_size >= 0 && cpu_size <= CS;
  bool size_match = false;
  int encode_difference = -1;
  if (compare_payload) {
    size_match = lane_sizes_match(gpu, cpu_size);
    if (size_match) encode_difference = first_difference(encoded.data(), gpu.output.data(), cpu_size);
    passed = passed && size_match && encode_difference < 0;
  } else if (cpu_success) {
    passed = false;
  }

  std::ostringstream decode;
  decode << "null";
  // Decode an independently CPU-encoded stream, never a GPU round-trip only.
  if (cpu_success && cpu_guards && cpu_size >= 0 && cpu_size <= CS) {
    ++decode_count;
    HostBuffer cpu_encoded = encoded, decoded;
    int decoded_size = cpu_size;
    cpu_decode(component, decoded_size, cpu_encoded, decoded);
    const Operation inverse = gpu_operation(device, component, true, cpu_size, encoded);
    const bool inverse_status = lane_status_matches(inverse, true);
    const bool inverse_size = decoded_size >= 0 && decoded_size <= CS &&
                              lane_sizes_match(inverse, decoded_size);
    int inverse_difference = -1, reference_difference = -1;
    if (inverse_status && inverse_size) {
      inverse_difference = first_difference(decoded.data(), inverse.output.data(), decoded_size);
    }
    if (decoded_size == length) {
      reference_difference = first_difference(original.data(), decoded.data(), length);
    }
    const bool reference_roundtrip = decoded_size == length && reference_difference < 0;
    const bool inverse_guards = inverse.guards_ok && cpu_encoded.guards_ok() && decoded.guards_ok();
    const bool inverse_ok = inverse_status && inverse_size && inverse_difference < 0 &&
                            reference_roundtrip && inverse_guards;
    passed = passed && inverse_ok;
    decode.str("");
    decode.clear();
    decode << std::boolalpha << "{\"status\":" << quoted(inverse_ok ? "passed" : "failed")
           << ",\"input_source\":\"cpu_encoded\",\"cpu_size\":" << decoded_size
           << ",\"gpu_size\":" << inverse.result.size[0]
           << ",\"all_lanes_success\":" << inverse_status
           << ",\"sizes_equal\":" << inverse_size
           << ",\"bytes_equal\":" << (inverse_status && inverse_size && inverse_difference < 0)
           << ",\"first_mismatch\":" << inverse_difference
           << ",\"cpu_roundtrip_exact\":" << reference_roundtrip
           << ",\"cpu_roundtrip_first_mismatch\":" << reference_difference
           << ",\"guards_ok\":" << inverse_guards << '}';
  }

  std::ostringstream out;
  out << std::boolalpha << "{\"component\":" << quoted(kComponents[component])
      << ",\"input_bytes\":" << length << ",\"pattern\":" << quoted(patterns[pattern])
      << ",\"status\":" << quoted(passed ? "passed" : "failed")
      << ",\"encode\":{\"cpu_success\":" << cpu_success
      << ",\"gpu_success\":" << (gpu.result.success[0] == 1)
      << ",\"all_lane_statuses_equal\":" << status_match
      << ",\"cpu_guards_ok\":" << cpu_guards << ",\"gpu_guards_ok\":" << gpu.guards_ok
      << ",\"cpu_size\":" << (cpu_success ? std::to_string(cpu_size) : "null")
      << ",\"gpu_size\":" << (gpu.result.success[0] == 1 ? std::to_string(gpu.result.size[0]) : "null")
      << ",\"sizes_equal\":" << (compare_payload ? (size_match ? "true" : "false") : "null")
      << ",\"bytes_equal\":" << (compare_payload ? (size_match && encode_difference < 0 ? "true" : "false") : "null")
      << ",\"first_mismatch\":" << (compare_payload && size_match ? std::to_string(encode_difference) : "null")
      << "},\"decode\":" << decode.str()
      << ",\"decode_skipped_reason\":" << (cpu_success ? "null" : "\"CPU encoder returned false; output is undefined\"")
      << '}';
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  DeviceBuffers device;
  std::vector<std::string> cases, errors;
  std::string expected_pci, actual_pci;
  int visible_count = -1, logical_device = -1, failed_cases = 0, decode_count = 0;
  hipDeviceProp_t properties{};
  bool properties_read = false;
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "{\"usage\":\"hip_components_test --expected-pci DDDD:BB:DD.F; expose exactly one assigned GPU\"}\n";
      return 0;
    }
    if (argc != 3 || std::string(argv[1]) != "--expected-pci") {
      throw std::runtime_error("Usage: hip_components_test --expected-pci DDDD:BB:DD.F");
    }
    expected_pci = pci_address(argv[2]);
    checked(hipGetDeviceCount(&visible_count), "hipGetDeviceCount");
    if (visible_count != 1) throw std::runtime_error("Exactly one visible GPU is required");
    checked(hipGetDevice(&logical_device), "hipGetDevice");
    if (logical_device != 0) throw std::runtime_error("Logical device must be zero");
    char pci[64]{};
    checked(hipDeviceGetPCIBusId(pci, sizeof(pci), logical_device), "hipDeviceGetPCIBusId");
    actual_pci = pci_address(pci);
    if (actual_pci != expected_pci) throw std::runtime_error("Assigned PCI address mismatch");
    checked(hipGetDeviceProperties(&properties, logical_device), "hipGetDeviceProperties");
    properties_read = true;
    if (properties.maxThreadsPerBlock < TPB || properties.sharedMemPerBlock < 3 * kStride) {
      throw std::runtime_error("Device cannot support the fixed component test launch");
    }
    checked(hipMalloc(reinterpret_cast<void**>(&device.input), kStride), "hipMalloc input");
    checked(hipMalloc(reinterpret_cast<void**>(&device.output), kStride), "hipMalloc output");
    checked(hipMalloc(reinterpret_cast<void**>(&device.result), sizeof(DeviceResult)), "hipMalloc result");

    const int lengths[] = {1, 7, 8, 31, 32, 33, 127, 128, 129, 511, 512, 513,
                           4095, 4096, 4097, CS - 1, CS};
    for (int component = DIFF4; component <= RZE1; ++component) {
      for (int length : lengths) {
        for (int pattern = 0; pattern < 3; ++pattern) {
          bool passed = false;
          cases.push_back(run_case(device, static_cast<Component>(component), length,
                                   pattern, passed, decode_count));
          if (!passed) ++failed_cases;
        }
      }
      std::cerr << "checked " << kComponents[component] << '\n';
    }
  } catch (const std::exception& error) {
    errors.push_back(error.what());
  }
  if (device.pending) cleanup_checked(hipDeviceSynchronize(), "cleanup synchronize", errors);
  if (device.result) cleanup_checked(hipFree(device.result), "hipFree result", errors);
  if (device.output) cleanup_checked(hipFree(device.output), "hipFree output", errors);
  if (device.input) cleanup_checked(hipFree(device.input), "hipFree input", errors);

  const bool passed = errors.empty() && failed_cases == 0 && cases.size() == 5 * 17 * 3;
  std::cout << std::boolalpha << "{\"status\":" << quoted(passed ? "passed" : "failed")
            << ",\"case_count\":" << cases.size()
            << ",\"failures\":" << failed_cases + errors.size()
            << ",\"decode_case_count\":" << decode_count
            << ",\"expected_pci\":" << quoted(expected_pci)
            << ",\"pci\":" << quoted(actual_pci)
            << ",\"visible_device_count\":" << visible_count
            << ",\"logical_device\":" << logical_device << ",\"device\":";
  if (properties_read) {
    std::cout << "{\"name\":" << quoted(properties.name)
              << ",\"gcn_arch_name\":" << quoted(properties.gcnArchName)
              << ",\"warp_size\":" << properties.warpSize << '}';
  } else std::cout << "null";
  std::cout << ",\"chunk_bytes\":" << CS << ",\"threads_per_block\":" << TPB
            << ",\"logical_warp_size\":" << WS << ",\"seed\":" << kSeed
            << ",\"device_allocation_bytes\":" << 2 * kStride + sizeof(DeviceResult)
            << ",\"shared_memory_bytes\":" << 3 * kStride
            << ",\"guard_policy\":\"64 bytes outside each full CS allocation; scratch contents are not compared\""
            << ",\"reference\":\"unchanged h_DIFFNB_4/8, h_BIT_4/8, h_RZE_1 from 36f5aae\""
            << ",\"performance_measurement\":false,\"cases\":[";
  for (size_t i = 0; i < cases.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << cases[i];
  }
  std::cout << "],\"errors\":[";
  for (size_t i = 0; i < errors.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << quoted(errors[i]);
  }
  std::cout << "]}\n";
  return passed ? 0 : 1;
}
