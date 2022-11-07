#pragma once

#include "ChannelInfo.h"
//#include <glog/logging.h>
#include <boost/format.hpp>
#include <string>
#include <vector>
#include <map>
#include <Winsock2.h>
#include <stdio.h>

namespace Chimera
{
	namespace nCache
	{
		class InterfaceWriter {
		public:
			InterfaceWriter()
			{
			};
			virtual ~InterfaceWriter() {};

			virtual bool write_header(Header& o_header) = 0;
			virtual bool write_channel(nCache::ChannelDataContainer channel) = 0;
		protected:

			static bool write_tag(FILE* i_fp, std::string& o_tag)
			{
				size_t string_buffer_index = 0;
				char c;
				do {
					c = o_tag[string_buffer_index];
					if (c != '\0')
						fwrite(&c, sizeof(c), 1, i_fp);
					string_buffer_index++;
				} while (c != '\0');
				return true;
			}
			static bool write_int32t(FILE* i_fp, int32_t& o_int32_value)
			{
				int32_t _tempvalue;
				_tempvalue = htonl(o_int32_value);
				fwrite(&_tempvalue, sizeof(_tempvalue), 1, i_fp);
				return true;
			}
			bool write_int64t(FILE* i_fp, int64_t& o_int64_value) const
			{
				int64_t _tempvalue;
				_tempvalue = htonll(o_int64_value);
				fwrite(&_tempvalue, sizeof(_tempvalue), 1, i_fp);
				return true;
			}
		};
	} // namespace nCache
}

