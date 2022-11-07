#pragma once

#include <string>
#include <stdexcept>
#include <iostream>
#include <map>
#include <boost/format.hpp>

namespace Chimera
{
	namespace nCache
	{


		struct Header
		{
			std::string VRSN; // Version Number (char*)
			int STIM;		  // Start Time of the Cache File (int)
			int ETIM;		  // End Time of the Cache File (int)
			int header_blob_size;
			int channels_blob_size;
		};
		struct FloatVector
		{
			float x;
			float y;
			float z;
		};
		enum ChannelDataType
		{
			DBLA,
			FVCA,
			FBCA,
			UNKNOWN
		};
		ChannelDataType string2ChannelDataType(const std::string& i_channel_data_type_string);
		std::string ChannelDataType2string(const ChannelDataType& i_channel_data_type);
		struct ChannelData {
			// std::string _type;
			ChannelDataType _type;
			std::vector <double> _dbla;
			std::vector <float> _fbca;
			std::vector <FloatVector> _fvca;
			size_t _real_size;

			void clear()
			{
				_dbla.clear();
				_fbca.clear();
				_fvca.clear();
			}
		};
		typedef std::map<std::string, ChannelData> ChannelDataContainer;


		struct ChannelInfo
		{
			ChannelInfo()
				: _sampling_type(REGULAR)
				, _sampling_rate(0)
				, _start_time(0)
				, _end_time(0)
			{

			};
			enum SamplingType { REGULAR, IRREGULAR };
			std::string  _channel_name;
			std::string  _channel_type;
			std::string  _channel_interpretation;
			SamplingType _sampling_type;
			size_t       _sampling_rate;
			size_t       _start_time;
			size_t       _end_time;
		};
		// typedef std::vector<ChannelInfo> ChannelInfoContainer;
		typedef std::map<std::string, ChannelInfo> ChannelInfoContainer;
	} // namespace nCache
}