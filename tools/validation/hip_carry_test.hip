// Exercise the production carry function directly, without a second protocol.
// The renamed original CLI is compiled but is never called by this test.
#define main pfpl_unused_cli_main
#include "../../src/f32_abs_comp_gpu.hip"
#undef main

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

static_assert(TPB == 512 && CS == 16384 && WS == 32,
              "Review carry test geometry when production geometry changes");
constexpr int kGuard = 0x5a17cc3d;
constexpr int kMaxChunks = 8193;
constexpr size_t kSharedBytes = 3 * CS;
constexpr std::array<int, 9> kChunkCounts{{1, 2, 31, 32, 33, 65, 129, 1025, 8193}};
constexpr std::array<int, 5> kGridCounts{{1, 2, 3, 80, 160}};
constexpr std::array<const char*, 3> kPatterns{{"ones", "alternating", "permuted"}};

struct CaseSpec {
  int chunks;
  int blocks;
  std::string pattern;
  int repeat;
};

struct CaseResult {
  CaseSpec spec;
  bool launched = false;
  bool completed = false;
  bool passed = false;
  int mismatches = 0;
  int observed_counter = -1;
  int expected_total = 0;
  int observed_total = -1;
  std::vector<std::string> details;
};

struct DeviceBuffers {
  int* sizes = nullptr;
  int* carry = nullptr;
  int* offsets = nullptr;
  int* visits = nullptr;
};

std::string quote_json(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
    else if (c < 0x20) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<unsigned>(c);
    } else out << static_cast<char>(c);
  }
  out << '"';
  return out.str();
}

std::string normalize_pci(const std::string& text) {
  unsigned domain, bus, device, function;
  char trailing;
  if (std::sscanf(text.c_str(), "%x:%x:%x.%x%c", &domain, &bus, &device,
                  &function, &trailing) != 4 || domain > 0xffff || bus > 0xff ||
      device > 0x1f || function > 7) {
    throw std::runtime_error("Invalid PCI address: " + text);
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(4) << domain << ':'
      << std::setw(2) << bus << ':' << std::setw(2) << device << '.' << function;
  return out.str();
}

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status) + " (" +
                             std::to_string(static_cast<int>(status)) + ")");
  }
  return;
}

void free_checked(int* pointer, const char* label, std::vector<std::string>& errors) {
  if (pointer) {
    const hipError_t status = hipFree(pointer);
    if (status != hipSuccess) {
      errors.push_back(std::string(label) + ": " + hipGetErrorString(status));
    }
  }
  return;
}

int positive_integer(const std::string& text) {
  size_t used = 0;
  const int value = std::stoi(text, &used);
  if (used != text.size() || value <= 0) {
    throw std::runtime_error("Expected a positive integer: " + text);
  }
  return value;
}

int input_size(int index, const std::string& pattern) {
  if (pattern == "ones") return 1;
  if (pattern == "alternating") return (index & 1) ? CS : 1;
  if (pattern == "permuted") return (index * 7919 + 17) % CS + 1;
  throw std::runtime_error("Unknown pattern: " + pattern);
}

// Match production's 512-thread blocks and 48-KiB shared-memory footprint.
// Volatile initialization keeps the full scratch allocation observable; the
// carry protocol itself is the included production propagate_carry function.
static __global__ __launch_bounds__(TPB, 2)
void run_production_carry(const int* sizes, int count, int* fullcarry,
                          int* exclusive_offsets, int* visits) {
  __shared__ long long chunk[3 * (CS / sizeof(long long))];
  volatile long long* const initialize_chunk = chunk;
  const int tid = threadIdx.x;
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;
  int* const shared_carry = (int*)&chunk[2 * (CS / sizeof(long long))];
  for (int index = tid; index < 3 * (CS / sizeof(long long)); index += TPB) {
    initialize_chunk[index] = 0;
  }
  __syncthreads();

  do {
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();
    const int chunk_id = static_cast<int>(chunk[last]);
    if (chunk_id >= count) break;

    // Production reaches carry after block-wide codec barriers. Poison the
    // carry output before that barrier so a missing publication is visible.
    if (tid == 0) *shared_carry = -1;
    __syncthreads();
    propagate_carry(sizes[chunk_id], chunk_id, fullcarry, shared_carry);
    __syncthreads();
    if (tid == 0) {
      exclusive_offsets[chunk_id] = (chunk_id == 0) ? 0 : *shared_carry;
      atomicAdd(&visits[chunk_id], 1);
    }
    // Finish test-only result recording before the block claims another ID.
    // This also keeps the instrumented loop on a common block-wide backedge.
    __syncthreads();
  } while (true);
  return;
}

void mismatch(CaseResult& result, const std::string& field, int index,
               int expected, int observed) {
  if (expected != observed) {
    ++result.mismatches;
    if (result.details.size() < 12) {
      result.details.push_back(field + "[" + std::to_string(index) + "] expected " +
                               std::to_string(expected) + ", observed " +
                               std::to_string(observed));
    }
  }
  return;
}

void run_case(const DeviceBuffers& device, CaseResult& result) {
  const CaseSpec& spec = result.spec;
  const size_t entries = static_cast<size_t>(spec.chunks) + 2;
  const size_t bytes = entries * sizeof(int);
  std::vector<int> sizes(entries, kGuard), carry(entries, kGuard);
  std::vector<int> offsets(entries, kGuard), visits(entries, kGuard);
  std::vector<int> expected_offsets(spec.chunks), expected_carry(spec.chunks);
  long long prefix = 0;
  for (int index = 0; index < spec.chunks; ++index) {
    const int value = input_size(index, spec.pattern);
    if (value <= 0 || value > CS) throw std::runtime_error("Invalid synthetic chunk size");
    sizes[index + 1] = value;
    carry[index + 1] = 0;
    offsets[index + 1] = -1;
    visits[index + 1] = 0;
    expected_offsets[index] = static_cast<int>(prefix);
    prefix += value;
    if (prefix > std::numeric_limits<int>::max()) {
      throw std::runtime_error("CPU reference prefix does not fit the production int type");
    }
    expected_carry[index] = static_cast<int>(prefix);
  }
  result.expected_total = static_cast<int>(prefix);
  check_hip(hipMemcpy(device.sizes, sizes.data(), bytes, hipMemcpyHostToDevice), "sizes H2D");
  check_hip(hipMemcpy(device.carry, carry.data(), bytes, hipMemcpyHostToDevice), "carry H2D");
  check_hip(hipMemcpy(device.offsets, offsets.data(), bytes, hipMemcpyHostToDevice), "offsets H2D");
  check_hip(hipMemcpy(device.visits, visits.data(), bytes, hipMemcpyHostToDevice), "visits H2D");

  d_reset<<<1, 1>>>();
  check_hip(hipGetLastError(), "production d_reset launch");
  run_production_carry<<<spec.blocks, TPB>>>(device.sizes + 1, spec.chunks,
                                          device.carry + 1, device.offsets + 1,
                                          device.visits + 1);
  check_hip(hipGetLastError(), "production carry test launch");
  result.launched = true;
  check_hip(hipDeviceSynchronize(), "production carry test synchronize");
  check_hip(hipMemcpy(carry.data(), device.carry, bytes, hipMemcpyDeviceToHost), "carry D2H");
  check_hip(hipMemcpy(offsets.data(), device.offsets, bytes, hipMemcpyDeviceToHost), "offsets D2H");
  check_hip(hipMemcpy(visits.data(), device.visits, bytes, hipMemcpyDeviceToHost), "visits D2H");
  std::vector<int> observed_sizes(entries);
  check_hip(hipMemcpy(observed_sizes.data(), device.sizes, bytes, hipMemcpyDeviceToHost),
            "sizes immutable D2H");
  check_hip(hipMemcpyFromSymbol(&result.observed_counter, HIP_SYMBOL(g_chunk_counter),
                                sizeof(int), 0, hipMemcpyDeviceToHost), "chunk counter D2H");

  for (int index = 0; index < spec.chunks; ++index) {
    mismatch(result, "exclusive_offset", index, expected_offsets[index], offsets[index + 1]);
    mismatch(result, "inclusive_fullcarry", index, expected_carry[index], carry[index + 1]);
    mismatch(result, "visit_count", index, 1, visits[index + 1]);
    mismatch(result, "input_size", index, sizes[index + 1], observed_sizes[index + 1]);
  }
  for (int index : {0, spec.chunks + 1}) {
    mismatch(result, "carry_guard", index, kGuard, carry[index]);
    mismatch(result, "offset_guard", index, kGuard, offsets[index]);
    mismatch(result, "visit_guard", index, kGuard, visits[index]);
    mismatch(result, "input_guard", index, kGuard, observed_sizes[index]);
  }
  // Every chunk is claimed once and every launched block claims one exit ID.
  mismatch(result, "chunk_counter", 0, spec.chunks + spec.blocks, result.observed_counter);
  result.observed_total = carry[spec.chunks];
  result.completed = true;
  result.passed = result.mismatches == 0;
  return;
}

void print_strings(const std::vector<std::string>& strings) {
  std::cout << '[';
  for (size_t index = 0; index < strings.size(); ++index) {
    if (index) std::cout << ", ";
    std::cout << quote_json(strings[index]);
  }
  std::cout << ']';
  return;
}

}  // namespace

int main(int argc, char** argv) {
  std::string expected_pci, actual_pci;
  int selected_chunks = 0, selected_blocks = 0, repeats = 2;
  std::string selected_pattern;
  std::vector<std::string> errors;
  std::vector<CaseSpec> cases;
  std::vector<CaseResult> results;
  DeviceBuffers device;
  int visible_devices = -1, logical_device = -1;
  hipDeviceProp_t properties{};
  bool properties_read = false;

  try {
    for (int arg = 1; arg < argc; ++arg) {
      const std::string option = argv[arg];
      if (arg + 1 >= argc) throw std::runtime_error("Missing option value for " + option);
      const std::string value = argv[++arg];
      if (option == "--expected-pci") expected_pci = normalize_pci(value);
      else if (option == "--chunks") selected_chunks = positive_integer(value);
      else if (option == "--blocks") selected_blocks = positive_integer(value);
      else if (option == "--repeats") repeats = positive_integer(value);
      else if (option == "--pattern") selected_pattern = value;
      else throw std::runtime_error("Unknown option: " + option);
    }
    if (expected_pci.empty()) throw std::runtime_error("--expected-pci DDDD:BB:DD.F is required");
    if (repeats > 3) throw std::runtime_error("--repeats must be between 1 and 3");
    for (int chunks : kChunkCounts) {
      if (selected_chunks && chunks != selected_chunks) continue;
      for (int blocks : kGridCounts) {
        if (selected_blocks && blocks != selected_blocks) continue;
        for (const char* pattern : kPatterns) {
          if (!selected_pattern.empty() && selected_pattern != pattern) continue;
          for (int repeat = 0; repeat < repeats; ++repeat) {
            cases.push_back({chunks, blocks, pattern, repeat});
          }
        }
      }
    }
    if (cases.empty()) throw std::runtime_error("Case filters do not select any supported case");
    check_hip(hipGetDeviceCount(&visible_devices), "hipGetDeviceCount");
    if (visible_devices != 1) throw std::runtime_error("Exactly one visible device is required");
    check_hip(hipGetDevice(&logical_device), "hipGetDevice");
    if (logical_device != 0) throw std::runtime_error("Logical device zero is required");
    check_hip(hipGetDeviceProperties(&properties, logical_device), "hipGetDeviceProperties");
    properties_read = true;
    char pci[64]{};
    check_hip(hipDeviceGetPCIBusId(pci, sizeof(pci), logical_device), "hipDeviceGetPCIBusId");
    actual_pci = normalize_pci(pci);
    if (actual_pci != expected_pci) throw std::runtime_error("PCI mapping does not match --expected-pci");
    if (properties.maxThreadsPerBlock < TPB || properties.sharedMemPerBlock < kSharedBytes) {
      throw std::runtime_error("Device cannot support the production block/shared-memory geometry");
    }

    const size_t allocation_bytes = (kMaxChunks + 2) * sizeof(int);
    check_hip(hipMalloc(reinterpret_cast<void**>(&device.sizes), allocation_bytes), "hipMalloc sizes");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device.carry), allocation_bytes), "hipMalloc carry");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device.offsets), allocation_bytes), "hipMalloc offsets");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device.visits), allocation_bytes), "hipMalloc visits");

    for (const CaseSpec& spec : cases) {
      std::cerr << "BEGIN carry chunks=" << spec.chunks << " blocks=" << spec.blocks
                << " pattern=" << spec.pattern << " repeat=" << spec.repeat << std::endl;
      results.push_back(CaseResult{spec});
      run_case(device, results.back());
      const CaseResult& result = results.back();
      std::cerr << (result.passed ? "PASS" : "FAIL") << " carry chunks=" << spec.chunks
                << " blocks=" << spec.blocks << " pattern=" << spec.pattern
                << " repeat=" << spec.repeat << " mismatches=" << result.mismatches << std::endl;
      if (!result.passed) break;
    }
  } catch (const std::exception& error) {
    errors.push_back(error.what());
    std::cerr << "ERROR carry: " << error.what() << std::endl;
  }

  free_checked(device.visits, "hipFree visits", errors);
  free_checked(device.offsets, "hipFree offsets", errors);
  free_checked(device.carry, "hipFree carry", errors);
  free_checked(device.sizes, "hipFree sizes", errors);
  size_t failures = errors.size();
  for (const CaseResult& result : results) if (!result.passed) ++failures;
  const bool passed = failures == 0 && !results.empty() && results.size() == cases.size();

  std::cout << std::boolalpha
            << "{\n  \"status\": " << quote_json(passed ? "passed" : "failed")
            << ",\n  \"case_count\": " << results.size()
            << ",\n  \"expected_case_count\": " << cases.size()
            << ",\n  \"failures\": " << failures
            << ",\n  \"visible_devices\": " << visible_devices
            << ",\n  \"logical_device\": " << logical_device
            << ",\n  \"expected_pci\": " << quote_json(expected_pci)
            << ",\n  \"pci\": " << quote_json(actual_pci)
            << ",\n  \"device\": ";
  if (properties_read) {
    std::cout << "{\"name\": " << quote_json(properties.name)
              << ", \"gcn_arch_name\": " << quote_json(properties.gcnArchName)
              << ", \"warp_size\": " << properties.warpSize
              << ", \"multiprocessors\": " << properties.multiProcessorCount << '}';
  } else std::cout << "null";
  std::cout << ",\n  \"production_function\": \"src/f32_abs_comp_gpu.hip::propagate_carry\""
            << ",\n  \"threads_per_block\": " << TPB
            << ",\n  \"shared_bytes_per_block\": " << kSharedBytes
            << ",\n  \"device_buffer_bytes\": " << 4 * (kMaxChunks + 2) * sizeof(int)
            << ",\n  \"cases\": [";
  for (size_t index = 0; index < results.size(); ++index) {
    if (index) std::cout << ',';
    const CaseResult& result = results[index];
    std::cout << "\n    {\"chunks\": " << result.spec.chunks
              << ", \"blocks\": " << result.spec.blocks
              << ", \"pattern\": " << quote_json(result.spec.pattern)
              << ", \"repeat\": " << result.spec.repeat
              << ", \"status\": " << quote_json(result.passed ? "passed" : "failed")
              << ", \"launched\": " << result.launched
              << ", \"completed\": " << result.completed
              << ", \"mismatches\": " << result.mismatches
              << ", \"exclusive_offsets_checked\": " << (result.completed ? result.spec.chunks : 0)
              << ", \"inclusive_carries_checked\": " << (result.completed ? result.spec.chunks : 0)
              << ", \"expected_total\": " << result.expected_total
              << ", \"observed_total\": " << result.observed_total
              << ", \"expected_counter\": " << result.spec.chunks + result.spec.blocks
              << ", \"observed_counter\": " << result.observed_counter
              << ", \"first_mismatches\": ";
    print_strings(result.details);
    std::cout << '}';
  }
  std::cout << "\n  ],\n  \"errors\": ";
  print_strings(errors);
  std::cout << "\n}\n";
  return passed ? 0 : 1;
}
