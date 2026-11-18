#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kBytes = 64 * 1024;

std::string quoted(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      out << '\\' << static_cast<char>(c);
    } else if (c < 0x20) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<unsigned>(c);
    } else {
      out << static_cast<char>(c);
    }
  }
  out << '"';
  return out.str();
}

std::string pci_address(const std::string& text) {
  unsigned domain, bus, device, function;
  char trailing;
  if (std::sscanf(text.c_str(), "%x:%x:%x.%x%c", &domain, &bus, &device,
                  &function, &trailing) != 4 ||
      domain > 0xffff || bus > 0xff || device > 0x1f || function > 7) {
    throw std::runtime_error("Invalid PCI address: " + text);
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(4) << domain << ':'
      << std::setw(2) << bus << ':' << std::setw(2) << device << '.' << function;
  return out.str();
}

std::string hip_error(hipError_t status, const char* operation) {
  return std::string(operation) + ": " + hipGetErrorString(status) +
         " (" + std::to_string(static_cast<int>(status)) + ")";
}

void checked(hipError_t status, const char* operation) {
  if (status != hipSuccess) throw std::runtime_error(hip_error(status, operation));
  return;
}

void cleanup_checked(hipError_t status, const char* operation,
                     std::vector<std::string>& errors) {
  if (status != hipSuccess) errors.push_back(hip_error(status, operation));
  return;
}

unsigned char pattern(size_t index) {
  return static_cast<unsigned char>((index * 37 + (index >> 8) + 11) & 0xff);
}

__global__ void xor_pattern(unsigned char* data, size_t count) {
  for (size_t index = threadIdx.x; index < count; index += blockDim.x) {
    data[index] ^= 0xa5;
  }
  return;
}

std::string environment(const char* name) {
  const char* value = std::getenv(name);
  return value ? quoted(value) : "null";
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> errors;
  std::string expected_pci;
  std::string actual_pci;
  int count = -1, device = -1, runtime_version = 0, driver_version = 0;
  hipDeviceProp_t properties{};
  bool properties_read = false;
  bool pci_matches = false, copy_matches = false, kernel_matches = false;
  bool event_synchronized = false, device_synchronized = false;
  bool work_pending = false;
  float event_ms = 0;
  unsigned char* device_data = nullptr;
  hipEvent_t start = nullptr, stop = nullptr;

  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "{\"usage\":\"hip_runtime_probe --expected-pci DDDD:BB:DD.F; "
                   "set HIP_VISIBLE_DEVICES to exactly one assigned device\"}\n";
      return 0;
    }
    if (argc != 3 || std::string(argv[1]) != "--expected-pci") {
      throw std::runtime_error("Usage: hip_runtime_probe --expected-pci DDDD:BB:DD.F");
    }
    expected_pci = pci_address(argv[2]);
    checked(hipGetDeviceCount(&count), "hipGetDeviceCount");
    if (count != 1) {
      throw std::runtime_error("Exactly one visible device is required; observed " +
                               std::to_string(count));
    }
    checked(hipGetDevice(&device), "hipGetDevice");
    if (device != 0) throw std::runtime_error("The selected logical device must be zero");
    checked(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    properties_read = true;
    char pci[64]{};
    checked(hipDeviceGetPCIBusId(pci, sizeof(pci), device), "hipDeviceGetPCIBusId");
    actual_pci = pci_address(pci);
    if (actual_pci != expected_pci) {
      throw std::runtime_error("PCI mapping mismatch: expected " + expected_pci +
                               ", observed " + actual_pci);
    }
    pci_matches = true;
    checked(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion");
    checked(hipDriverGetVersion(&driver_version), "hipDriverGetVersion");
    if (properties.maxThreadsPerBlock < 64) {
      throw std::runtime_error("The device cannot launch the 64-thread smoke kernel");
    }

    std::array<unsigned char, kBytes> host;
    for (size_t index = 0; index < kBytes; ++index) host[index] = pattern(index);
    checked(hipMalloc(reinterpret_cast<void**>(&device_data), kBytes), "hipMalloc");
    checked(hipMemcpy(device_data, host.data(), kBytes, hipMemcpyHostToDevice),
            "hipMemcpy H2D");
    std::fill(host.begin(), host.end(), 0);
    checked(hipMemcpy(host.data(), device_data, kBytes, hipMemcpyDeviceToHost),
            "hipMemcpy D2H copy verification");
    for (size_t index = 0; index < kBytes; ++index) {
      if (host[index] != pattern(index)) {
        throw std::runtime_error("H2D/D2H mismatch at byte " + std::to_string(index));
      }
    }
    copy_matches = true;

    checked(hipEventCreate(&start), "hipEventCreate start");
    checked(hipEventCreate(&stop), "hipEventCreate stop");
    checked(hipEventRecord(start, nullptr), "hipEventRecord start");
    work_pending = true;
    xor_pattern<<<1, 64>>>(device_data, kBytes);
    checked(hipGetLastError(), "xor_pattern launch");
    checked(hipEventRecord(stop, nullptr), "hipEventRecord stop");
    checked(hipEventSynchronize(stop), "hipEventSynchronize stop");
    event_synchronized = true;
    checked(hipDeviceSynchronize(), "hipDeviceSynchronize");
    work_pending = false;
    device_synchronized = true;
    checked(hipEventElapsedTime(&event_ms, start, stop), "hipEventElapsedTime");
    std::fill(host.begin(), host.end(), 0);
    checked(hipMemcpy(host.data(), device_data, kBytes, hipMemcpyDeviceToHost),
            "hipMemcpy D2H kernel verification");
    for (size_t index = 0; index < kBytes; ++index) {
      if (host[index] != static_cast<unsigned char>(pattern(index) ^ 0xa5)) {
        throw std::runtime_error("Kernel output mismatch at byte " + std::to_string(index));
      }
    }
    kernel_matches = true;
  } catch (const std::exception& error) {
    errors.push_back(error.what());
  }

  if (work_pending) {
    cleanup_checked(hipDeviceSynchronize(), "cleanup hipDeviceSynchronize", errors);
  }
  if (stop) cleanup_checked(hipEventDestroy(stop), "hipEventDestroy stop", errors);
  if (start) cleanup_checked(hipEventDestroy(start), "hipEventDestroy start", errors);
  if (device_data) cleanup_checked(hipFree(device_data), "hipFree", errors);

  std::cout << std::boolalpha
            << "{\n  \"status\": " << quoted(errors.empty() ? "passed" : "failed")
            << ",\n  \"visible_device_count\": " << count
            << ",\n  \"logical_device\": " << device
            << ",\n  \"expected_pci\": " << quoted(expected_pci)
            << ",\n  \"actual_pci\": " << quoted(actual_pci)
            << ",\n  \"hip_visible_devices\": " << environment("HIP_VISIBLE_DEVICES")
            << ",\n  \"rocr_visible_devices\": " << environment("ROCR_VISIBLE_DEVICES")
            << ",\n  \"runtime_version\": " << runtime_version
            << ",\n  \"driver_version\": " << driver_version
            << ",\n  \"device_properties\": ";
  if (properties_read) {
    std::cout << "{\"name\": " << quoted(properties.name)
              << ", \"gcn_arch_name\": " << quoted(properties.gcnArchName)
              << ", \"warp_size\": " << properties.warpSize
              << ", \"max_threads_per_block\": " << properties.maxThreadsPerBlock
              << ", \"shared_memory_per_block_bytes\": " << properties.sharedMemPerBlock
              << ", \"multiprocessor_count\": " << properties.multiProcessorCount
              << ", \"pci_domain_id\": " << properties.pciDomainID
              << ", \"pci_bus_id\": " << properties.pciBusID
              << ", \"pci_device_id\": " << properties.pciDeviceID << '}';
  } else {
    std::cout << "null";
  }
  std::cout << ",\n  \"host_buffer_bytes\": " << kBytes
            << ",\n  \"device_buffer_bytes\": " << kBytes
            << ",\n  \"kernel_grid\": [1, 64]"
            << ",\n  \"checks\": {\"pci_matches\": " << pci_matches
            << ", \"copy_exact\": " << copy_matches
            << ", \"kernel_exact\": " << kernel_matches
            << ", \"event_synchronized\": " << event_synchronized
            << ", \"device_synchronized\": " << device_synchronized << '}'
            << ",\n  \"event_elapsed_ms_smoke_only\": " << event_ms
            << ",\n  \"performance_measurement\": false"
            << ",\n  \"errors\": [";
  for (size_t index = 0; index < errors.size(); ++index) {
    if (index) std::cout << ", ";
    std::cout << quoted(errors[index]);
  }
  std::cout << "]\n}\n";
  return errors.empty() ? 0 : 1;
}
