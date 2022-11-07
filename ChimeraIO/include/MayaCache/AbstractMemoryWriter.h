#pragma once

#include "InterfaceWriter.h"
#include <string>
#include <vector>

namespace Chimera
{
	namespace nCache
	{

		class AbstractMemoryWriter : public InterfaceWriter
		{
		public:
			typedef unsigned char DataBufferType;
			typedef DataBufferType* DataBufferPtr;
			typedef std::vector<DataBufferType> DataBufferContainer;

			AbstractMemoryWriter(const std::string& i_ncache_filename, const nCache::Header& o_header);
			virtual ~AbstractMemoryWriter();

			bool write(nCache::Header& o_header);
		protected:
			// HEADER
			bool write_header_tag(std::string& o_tag);
			bool write_header_int32(int32_t& o_value);
			bool write_header_int64(int64_t& o_value);
			bool write_header_pascal_string_32(std::string& o_string, int32_t o_bytes_to_write);
			bool write_header_pascal_string_64(std::string& o_string, int32_t o_bytes_to_write);
			bool can_write_more_header_data(size_t i_bytes_to_read) const;

			// CHANNEL
			bool write_channel_tag(std::string& o_tag);
			bool write_channel_int8(int8_t& o_value);
			bool write_channel_int16(int16_t& o_value);
			bool write_channel_int32(int32_t& o_value);
			bool write_channel_int64(int64_t& o_value);
			bool write_channel_pascal_string_32(std::string& o_string);
			bool write_channel_pascal_string_64(std::string& o_string);
			bool write_channel_blob(size_t i_bytes_to_read, void* o_blob);
			bool can_write_more_channel_data(size_t i_bytes_to_read) const;

			template<typename T>
			bool write_blob(FILE* i_fp, size_t i_blob_size, T* o_blob) const;
			float reverse_float(const float inFloat) const;
			double reverse_double(const double inDouble) const;

		protected:
			DataBufferContainer _header_data_unsigned_char_buffer;
			DataBufferContainer _channel_data_unsigned_char_buffer;
			DataBufferPtr _header_data_current_ptr;
			DataBufferPtr _channel_data_current_ptr;
			DataBufferPtr _header_data_end_ptr;
			DataBufferPtr _channel_data_end_ptr;
			std::string _cache_filename;

			FILE *_fp;
		};

	} // namespace nCache
}
