#include "MayaCache/AbstractMemoryWriter.h"
#include <iostream>
#include <stdexcept>
#include <stdio.h>

namespace Chimera
{
#pragma comment(lib, "Ws2_32.lib")
	using namespace nCache;

	AbstractMemoryWriter::AbstractMemoryWriter(const std::string& i_ncache_filename, const nCache::Header& o_header)
		: _header_data_current_ptr(0)
		, _channel_data_current_ptr(0)
		, _header_data_end_ptr(0)
		, _channel_data_end_ptr(0)
		, _cache_filename(i_ncache_filename)
	{
		_header_data_unsigned_char_buffer.resize(o_header.header_blob_size);
		_header_data_current_ptr = _header_data_unsigned_char_buffer.data();
		_header_data_end_ptr = _header_data_current_ptr + _header_data_unsigned_char_buffer.size();

		_channel_data_unsigned_char_buffer.resize(o_header.channels_blob_size);
		_channel_data_current_ptr = _channel_data_unsigned_char_buffer.data();
		_channel_data_end_ptr = _channel_data_current_ptr + _channel_data_unsigned_char_buffer.size();
	}

	AbstractMemoryWriter::~AbstractMemoryWriter()
	{
	}

	bool AbstractMemoryWriter::write(nCache::Header& o_header)
	{
		fopen_s(&_fp, _cache_filename.c_str(), "wb");
		if (_fp == 0)
			throw std::runtime_error((boost::format("Failed to open file '%1%'") % _cache_filename).str());
		else {
			std::string tag;
			size_t blob_size;
			int32_t tag_value_int32;
			int64_t tag_value_int64;
			
			// Header
			tag = "FOR8";
			tag_value_int64 = 0;
			InterfaceWriter::write_tag(_fp, tag);
			if (tag.compare("FOR4") == 0)
			{
				InterfaceWriter::write_int32t(_fp, tag_value_int32);
				blob_size = tag_value_int32;
			}
			else if (tag.compare("FOR8") == 0)
			{
				InterfaceWriter::write_int64t(_fp, tag_value_int64); // some 64bit data
				tag_value_int32 = o_header.header_blob_size;
				InterfaceWriter::write_int32t(_fp, tag_value_int32);
			}
			else
				throw std::runtime_error((boost::format("Failed to open file '%1%', unknown format '%2%'") % _cache_filename%tag).str());
			write_blob<DataBufferType>(_fp, o_header.header_blob_size, _header_data_unsigned_char_buffer.data());

			// Channels
			tag = "FOR8";
			tag_value_int64 = 493;
			InterfaceWriter::write_tag(_fp, tag);
			if (tag.compare("FOR4") == 0)
			{
				InterfaceWriter::write_int32t(_fp, tag_value_int32);
				blob_size = tag_value_int32;
			}
			else if (tag.compare("FOR8") == 0)
			{
				InterfaceWriter::write_int64t(_fp, tag_value_int64); // some 64bit data
				tag_value_int32 = o_header.channels_blob_size;
				InterfaceWriter::write_int32t(_fp, tag_value_int32);
			}
			else
				throw std::runtime_error((boost::format("Failed to open file '%1%', unknown format '%2%'") % _cache_filename%tag).str());
			write_blob<DataBufferType>(_fp, o_header.channels_blob_size, _channel_data_unsigned_char_buffer.data());
			fclose(_fp);
		}
		return true;
	}

	// HEADER
	bool AbstractMemoryWriter::write_header_tag(std::string& o_tag)
	{
		const size_t bytes_to_read = 4;
		if (!can_write_more_header_data(bytes_to_read))
			return false;
		char tag_string[bytes_to_read + 1];
		strcpy_s(tag_string, o_tag.c_str());

		memcpy(_header_data_current_ptr, tag_string, bytes_to_read);
		_header_data_current_ptr += bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_header_int32(int32_t& o_value)
	{
		const size_t bytes_to_read = sizeof(o_value);
		if (!can_write_more_header_data(bytes_to_read))
			return false;
		int32_t dummy_value;
		dummy_value = ntohl(o_value);
		memcpy(_header_data_current_ptr, &dummy_value, bytes_to_read);
		_header_data_current_ptr += bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_header_int64(int64_t& o_value)
	{
		const size_t bytes_to_read = sizeof(o_value);
		if (!can_write_more_header_data(bytes_to_read))
			return false;
		int64_t dummy_value;
		dummy_value = o_value;
		memcpy(_header_data_current_ptr, &dummy_value, bytes_to_read);
		_header_data_current_ptr += bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_header_pascal_string_32(std::string& o_string, int32_t o_bytes_to_write)
	{
		int32_t bytes_to_write;
		bytes_to_write = o_bytes_to_write;
		write_header_int32(bytes_to_write);
		char pascal_string_buffer[4096];
		strcpy_s(pascal_string_buffer, o_string.c_str());

		memcpy(_header_data_current_ptr, pascal_string_buffer, bytes_to_write);
		_header_data_current_ptr += bytes_to_write;

		return true;
	}

	bool AbstractMemoryWriter::write_header_pascal_string_64(std::string& o_string, int32_t o_bytes_to_write)
	{
		int64_t dummy_value;
		int32_t bytes_to_write;
		dummy_value = 493;
		write_header_int64(dummy_value);
		bytes_to_write = o_bytes_to_write;
		write_header_int32(bytes_to_write);
		char pascal_string_buffer[4096];
		strcpy_s(pascal_string_buffer, o_string.c_str());
		memcpy(_header_data_current_ptr, pascal_string_buffer, bytes_to_write);

		_header_data_current_ptr += bytes_to_write;

		return true;
	}
	bool AbstractMemoryWriter::can_write_more_header_data(size_t i_bytes_to_read) const
	{
		return ((_header_data_current_ptr + i_bytes_to_read) <= _header_data_end_ptr);
	}

	bool AbstractMemoryWriter::can_write_more_channel_data(size_t i_bytes_to_read) const
	{
		return ((_channel_data_current_ptr + i_bytes_to_read) <= _channel_data_end_ptr);
	}

	// CHANNEL
	bool AbstractMemoryWriter::write_channel_tag(std::string& o_tag)
	{
		const size_t bytes_to_read = 4;
		char tag_string[bytes_to_read + 1];

		strcpy_s(tag_string, o_tag.c_str());
		memcpy(_channel_data_current_ptr, tag_string, bytes_to_read);
		_channel_data_current_ptr += bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_channel_int8(int8_t& o_value)
	{
		const size_t bytes_to_read = sizeof(o_value);
		int8_t dummy_value;

		dummy_value = o_value;
		memcpy(_channel_data_current_ptr, &dummy_value, bytes_to_read);
		_channel_data_current_ptr += bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_channel_int16(int16_t& o_value)
	{
		const size_t bytes_to_read = sizeof(o_value);
		int16_t dummy_value;
		dummy_value = o_value;
		memcpy(_channel_data_current_ptr, &dummy_value, bytes_to_read);
		_channel_data_current_ptr += bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_channel_int32(int32_t& o_value)
	{
		const size_t bytes_to_read = sizeof(o_value);
		int32_t dummy_value;
		dummy_value = ntohl(o_value);
		memcpy(_channel_data_current_ptr, &dummy_value, bytes_to_read);
		_channel_data_current_ptr += bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_channel_int64(int64_t& o_value)
	{
		const size_t bytes_to_read = sizeof(o_value);
		int64_t dummy_value;
		dummy_value = htonll(o_value);
		memcpy(_channel_data_current_ptr, &dummy_value, bytes_to_read);
		_channel_data_current_ptr += bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_channel_pascal_string_32(std::string& o_string)
	{
		const int32_t modulo = 4;
		int32_t bytes_to_read;
		int32_t padded_bytes_to_read;
		write_channel_int32(bytes_to_read);

		int32_t  bytes_to_read_modulus = bytes_to_read%modulo;
		if (bytes_to_read_modulus)
			padded_bytes_to_read = bytes_to_read + (modulo - bytes_to_read_modulus);
		else
			padded_bytes_to_read = bytes_to_read;

		char pascal_string_buffer[4096];
		strcpy_s(pascal_string_buffer, o_string.c_str());
		memcpy(_channel_data_current_ptr, pascal_string_buffer, padded_bytes_to_read);

		_channel_data_current_ptr += padded_bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_channel_pascal_string_64(std::string& o_string)
	{
		const int32_t modulo = 8;
		int32_t bytes_to_read;
		int32_t padded_bytes_to_read;
		bytes_to_read = o_string.length() + 1;
		write_channel_int32(bytes_to_read);

		int32_t  bytes_to_read_modulus = bytes_to_read%modulo;

		if (bytes_to_read_modulus)
			padded_bytes_to_read = bytes_to_read + (modulo - bytes_to_read_modulus);
		else
			padded_bytes_to_read = bytes_to_read;

		char pascal_string_buffer[4096];
		strcpy_s(pascal_string_buffer, o_string.c_str());
		//pascal_string_buffer[bytes_to_read] = '\0';
		memcpy(_channel_data_current_ptr, pascal_string_buffer, padded_bytes_to_read);

		_channel_data_current_ptr += padded_bytes_to_read;

		return true;
	}

	bool AbstractMemoryWriter::write_channel_blob(size_t i_bytes_to_read, void* o_blob)
	{
		memcpy(_channel_data_current_ptr, o_blob, i_bytes_to_read);

		_channel_data_current_ptr += i_bytes_to_read;
		return true;
	}

	template<typename T>
	bool AbstractMemoryWriter::write_blob(FILE* i_fp, size_t i_blob_size, T* o_blob) const
	{
		fwrite(o_blob, sizeof(T), i_blob_size, i_fp);
		return true;
	}

	float AbstractMemoryWriter::reverse_float(const float inFloat) const
	{
		float retVal;
		char *floatToConvert = (char*)& inFloat;
		char *returnFloat = (char*)& retVal;

		// swap the bytes into a temporary buffer
		returnFloat[0] = floatToConvert[3];
		returnFloat[1] = floatToConvert[2];
		returnFloat[2] = floatToConvert[1];
		returnFloat[3] = floatToConvert[0];

		return retVal;
	}

	double AbstractMemoryWriter::reverse_double(const double inDouble) const
	{
		double retVal;
		char *doubleToConvert = (char*)& inDouble;
		char *returnDouble = (char*)& retVal;

		// swap the bytes into a temporary buffer
		returnDouble[0] = doubleToConvert[7];
		returnDouble[1] = doubleToConvert[6];
		returnDouble[2] = doubleToConvert[5];
		returnDouble[3] = doubleToConvert[4];
		returnDouble[4] = doubleToConvert[3];
		returnDouble[5] = doubleToConvert[2];
		returnDouble[6] = doubleToConvert[1];
		returnDouble[7] = doubleToConvert[0];

		return retVal;
	}
}
