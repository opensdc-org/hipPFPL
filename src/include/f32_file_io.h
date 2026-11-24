#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>


static inline std::unique_ptr<byte[]> read_f32_bytes(std::FILE* file, int size)
{
  std::unique_ptr<byte[]> data(new byte[size]);
  if (std::fseek(file, 0, SEEK_SET) != 0 ||
      std::fread(data.get(), 1, size, file) != static_cast<std::size_t>(size))
    throw std::runtime_error("Cannot read complete input file");
  return data;
}


static inline std::unique_ptr<byte[]> read_f32_input(const char* filename, int& size)
{
  const std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::fopen(filename, "rb"), std::fclose);
  if (!file || std::fseek(file.get(), 0, SEEK_END) != 0)
    throw std::runtime_error("Cannot open or seek input file");
  const long length = std::ftell(file.get());
  if (length <= 0 || length >= 2147221529 || length % sizeof(float) != 0)
    throw std::runtime_error("Expected a nonempty array of complete FP32 values within the size limit");
  size = static_cast<int>(length);
  return read_f32_bytes(file.get(), size);
}


static inline int read_f32_u16(const byte* field)
{
  return static_cast<int>(field[0]) | (static_cast<int>(field[1]) << 8);
}


static inline std::unique_ptr<byte[]> read_f32_packet(const char* filename, int& decoded_size,
    int& chunks, int& packet_size)
{
  const std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::fopen(filename, "rb"), std::fclose);
  if (!file) throw std::runtime_error("Cannot open encoded file");

  std::int64_t original_size = 0;
  if (std::fread(&original_size, sizeof(original_size), 1, file.get()) != 1 ||
      original_size <= 0 || original_size > std::numeric_limits<int>::max() ||
      std::fseek(file.get(), 0, SEEK_END) != 0)
    throw std::runtime_error("Invalid encoded file size");

  decoded_size = static_cast<int>(original_size);
  chunks = 1 + (decoded_size - 1) / CS;
  const int directory_bytes = 16 + 2 * chunks;
  const long file_size = std::ftell(file.get());
  if (file_size < directory_bytes || file_size > std::numeric_limits<int>::max() ||
      file_size - directory_bytes > decoded_size)
    throw std::runtime_error("Invalid encoded file size");

  packet_size = static_cast<int>(file_size);
  auto packet = read_f32_bytes(file.get(), packet_size);
  if (std::memcmp(packet.get(), &original_size, sizeof(original_size)) != 0)
    throw std::runtime_error("Encoded file header changed during read");
  int offset = directory_bytes;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int stored = read_f32_u16(packet.get() + 16 + 2 * chunk);
    const int raw = std::min(CS, decoded_size - chunk * CS);
    if (stored <= 0 || stored > raw || stored > packet_size - offset)
      throw std::runtime_error("Invalid encoded chunk length");
    offset += stored;
    if (stored < raw) {
      // A zero block has just a two-byte footer. Other blocks also carry a four-byte top bitmap.
      if (stored != 2 && stored < 6)
        throw std::runtime_error("Encoded chunk is too short for its RZE metadata");
      if (read_f32_u16(packet.get() + offset - 2) != raw)
        throw std::runtime_error("Invalid encoded chunk footer");
    }
  }
  if (offset != packet_size) throw std::runtime_error("Encoded packet size does not match chunk lengths");
  return packet;
}


static inline void write_f32_output(const char* filename, const byte* data, int size)
{
  std::FILE* const file = std::fopen(filename, "wb");
  if (!file) throw std::runtime_error("Cannot open output file");
  const auto written = std::fwrite(data, 1, size, file);
  const int status = std::fclose(file);
  if (written != static_cast<std::size_t>(size) || status != 0)
    throw std::runtime_error("Cannot write complete output file");
}
