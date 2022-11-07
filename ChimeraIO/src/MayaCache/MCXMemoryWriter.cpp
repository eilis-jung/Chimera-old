#include "MayaCache/MCXMemoryWriter.h"
#include <iostream>
#include <boost/format.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <stdio.h>

namespace Chimera
{
	using namespace nCache;

	//  |---CACH (Group)	// Header
	//  |     |---VRSN		// Version Number (char*)
	//  |     |---STIM		// Start Time of the Cache File (int)
	//  |     |---ETIM		// End Time of the Cache File (int)
	//  |
	//  |---MYCH (Group)	// 1st Time
	//  |     |---TIME		// Time (int)
	//  |     |---CHNM		// 1st Channel Name (char*)
	//  |     |---SIZE		// 1st Channel Size
	//  |     |---DVCA		// 1st Channel Data (Double Vector Array)
	//  |     |---CHNM		// n-th Channel Name
	//  |     |---SIZE		// n-th Channel Size
	//  |     |---DVCA		// n-th Channel Data (Double Vector Array)
	//  |     |..
	//  |
	//  |---MYCH (Group)	// 2nd Time
	//  |     |---TIME		// Time
	//  |     |---CHNM		// 1st Channel Name
	//  |     |---SIZE		// 1st Channel Size
	//  |     |---DVCA		// 1st Channel Data (Double Vector Array)
	//  |     |---CHNM		// n-th Channel Name
	//  |     |---SIZE		// n-th Channel Size
	//  |     |---DVCA		// n-th Channel Data (Double Vector Array)
	//  |     |..
	//  |
	//  |---..
	//	|
	//

	MCXMemoryWriter::MCXMemoryWriter(const std::string& i_mcx_filename, const nCache::Header& o_header, const ChannelInfoContainer* i_channels_info)
		: AbstractMemoryWriter(i_mcx_filename, o_header)
	{
	}

	MCXMemoryWriter::~MCXMemoryWriter()
	{
	}

	// std::string& o_VRSN, int& o_STIM, int& o_ETIM
	bool MCXMemoryWriter::write_header(Header& o_header)
	{
		//o_header.ETIM
		//o_header.STIM
		//o_header.VRSN

		std::string tag;
		size_t blob_size;
		int32_t value_int32, bytes_to_write;
		int64_t value_int64;
		tag = "CACH";
		write_header_tag(tag);
		//version
		//DLOG(INFO) << boost::format("HEADER : 01 tag '%1%'") % tag << std::endl;
		tag = "VRSN";
		write_header_tag(tag);
		//DLOG(INFO) << boost::format("HEADER : 02 tag '%1%'") % tag << std::endl;
		tag = o_header.VRSN;//mcx file version
		bytes_to_write = 4;
		write_header_pascal_string_64(tag, bytes_to_write);
		//DLOG(INFO) << boost::format("HEADER : 03 version '%1%'") % tag << std::endl;
		value_int32 = 0;
		write_header_int32(value_int32);
		//start
		tag = "STIM";
		write_header_tag(tag);
		//DLOG(INFO) << boost::format("HEADER : 04 tag '%1%'") % tag << std::endl;
		value_int64 = 0;
		write_header_int64(value_int64);
		value_int32 = 4;
		write_header_int32(value_int32);
		value_int32 = o_header.STIM;//to be changed
		write_header_int32(value_int32);
		value_int32 = 0;
		write_header_int32(value_int32);
		//end
		tag = "ETIM";
		write_header_tag(tag);
		//DLOG(INFO) << boost::format("HEADER : 04 tag '%1%'") % tag << std::endl;
		value_int64 = 0;
		write_header_int64(value_int64);
		value_int32 = 4;
		write_header_int32(value_int32);
		value_int32 = o_header.ETIM;//to be changed
		write_header_int32(value_int32);
		value_int32 = 0;
		write_header_int32(value_int32);
		//DLOG(INFO) << boost::format("HEADER : 05 o_header.STIM %1%") % o_header.STIM << std::endl;
		return true;
	}

	bool MCXMemoryWriter::write_channel(nCache::ChannelDataContainer _channels_data)
	{
		std::string tag;
		int8_t value_int8;
		int16_t value_int16;
		int32_t value_int32;
		int64_t value_int64;
		tag = "MYCH";
		write_channel_tag(tag);
		if (!_channels_data.empty())
		{
			nCache::ChannelDataContainer::const_iterator iter;
			nCache::ChannelDataContainer::const_iterator eIter = _channels_data.end();
			for (iter = _channels_data.begin(); iter != eIter; ++iter)
			{
				tag = "CHNM";
				write_channel_tag(tag);
				//dummy value
				value_int64 = 0;
				write_channel_int64(value_int64);
				tag = iter->first;
				write_channel_pascal_string_64(tag);
				tag = "SIZE";
				write_channel_tag(tag);
				//dummy value
				value_int64 = 0;
				write_channel_int64(value_int64);
				//dummy value
				value_int32 = 4;
				write_channel_int32(value_int32);
				if (iter->second._type == nCache::ChannelDataType::DBLA)
				{
					value_int32 = iter->second._real_size;
					write_channel_int32(value_int32);
					//dummy value
					value_int32 = 0;
					write_channel_int32(value_int32);
					tag = "DBLA";
					write_channel_tag(tag);
					//dummy value
					value_int64 = 0;
					write_channel_int64(value_int64);
					value_int32 = sizeof(double)*iter->second._real_size;
					write_channel_int32(value_int32);
					std::vector<double> dbla_buffer(iter->second._dbla.size());
					for (size_t i = 0; i<iter->second._dbla.size(); i++)
					{
						dbla_buffer[i] = reverse_double(iter->second._dbla[i]);
					}
					write_channel_blob(value_int32, dbla_buffer.data());
				}
				else if (iter->second._type == nCache::ChannelDataType::FVCA)
				{
					const int32_t modulo = 8;
					value_int32 = iter->second._real_size;
					write_channel_int32(value_int32);
					//dummy value
					value_int32 = 0;
					write_channel_int32(value_int32);
					tag = "FVCA";
					write_channel_tag(tag);
					//dummy value
					value_int64 = 0;
					write_channel_int64(value_int64);
					value_int32 = sizeof(float)*iter->second._real_size * 3;
					write_channel_int32(value_int32);
					int32_t  array_buffer_size_modulus = value_int32%modulo;
					int32_t padded_array_buffer_size = value_int32;

					if (array_buffer_size_modulus)
						padded_array_buffer_size = value_int32 + (modulo - array_buffer_size_modulus);

					std::vector<char> padded_fvca_buffer(padded_array_buffer_size);
					float* fvca_buffer_ptr = reinterpret_cast<float*>(padded_fvca_buffer.data());
					for (size_t i = 0; i<iter->second._fvca.size(); i++)
					{
						fvca_buffer_ptr[i * 3 + 0] = reverse_float(iter->second._fvca[i].x);
						fvca_buffer_ptr[i * 3 + 1] = reverse_float(iter->second._fvca[i].y);
						fvca_buffer_ptr[i * 3 + 2] = reverse_float(iter->second._fvca[i].z);
					}
					write_channel_blob(padded_array_buffer_size, padded_fvca_buffer.data());
				}
				else if (iter->second._type == nCache::ChannelDataType::FBCA)
				{
					const int32_t modulo = 8;
					value_int32 = iter->second._real_size;
					write_channel_int32(value_int32);
					//dummy value
					value_int32 = 0;
					write_channel_int32(value_int32);
					tag = "FBCA";
					write_channel_tag(tag);
					//dummy value
					value_int64 = 0;
					write_channel_int64(value_int64);
					value_int32 = sizeof(float)*iter->second._real_size;
					write_channel_int32(value_int32);
					int32_t  array_buffer_size_modulus = value_int32%modulo;
					int32_t padded_array_buffer_size = value_int32;

					if (array_buffer_size_modulus)
						padded_array_buffer_size = value_int32 + (modulo - array_buffer_size_modulus);

					std::vector<float> fbca_buffer(padded_array_buffer_size);
					for (size_t i = 0; i<iter->second._fbca.size(); i++)
					{
						fbca_buffer[i] = reverse_float(iter->second._fbca[i]);
					}
					write_channel_blob(padded_array_buffer_size, fbca_buffer.data());
				}
			}
		}
		return true;
	}
}
