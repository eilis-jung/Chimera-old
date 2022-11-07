#pragma once

#include "ChannelInfo.h"

#include <string>
#include <stdexcept>
#include <set>
#include <iostream>

#include <boost/format.hpp>

namespace Chimera
{
	namespace nCache
	{
		// Xerces Error codes

		enum {
			ERROR_ARGS = 1,
			ERROR_XERCES_INIT,
			ERROR_PARSE,
			ERROR_EMPTY_DOCUMENT
		};

		class XMLWriter
		{
		public:
			XMLWriter();
			~XMLWriter();
			void init(std::string cache_type, std::string cache_format, size_t start, size_t end, size_t timePerFrame, std::string version);
			void write(const std::string& i_ncache_xml_filename);
			void addExtra(std::string extra) { _extras.insert(extra); }
			void addChannel(std::string name, ChannelInfo channel) { _channels.insert(std::pair<std::string, ChannelInfo>(name, channel)); }
			void set_cache_type(std::string cache_type) { _cache_type = cache_type; }
			void set_cache_format(std::string cache_format) { _cache_format = cache_format; }
			void set_cache_version(std::string cache_version) { _cacheVersion_Version = cache_version; }
			const ChannelInfoContainer& get_channels() const { return _channels; }
			std::string get_base_cache_name() const { return _base_cache_name; };
			std::string get_cache_directory() const { return _cache_directory; };
			// size_t get_num_frames() const { return _num_frames;};

			void set_time_range_start(size_t start) { _time_range_start = start; };
			void set_time_range_end(size_t end) { _time_range_end = end; };
			void set_cacheTimePerFrame_TimePerFrame(size_t timePerFrame) { _cacheTimePerFrame_TimePerFrame = timePerFrame; };
			void set_particle_count_sampling_rate(size_t particle) { _particle_count_sampling_rate = particle; };

		protected:
			std::string _cache_type;
			std::string _cache_format;
			size_t _time_range_start;
			size_t _time_range_end;
			size_t _cacheTimePerFrame_TimePerFrame;
			std::string _cacheVersion_Version;

			size_t _particle_count_sampling_rate;

			std::set<std::string> _extras;
			ChannelInfoContainer _channels;
			// size_t _num_frames;
			/*!
			* \brief Extracted from input *.xml and used for generating Frame/Subframe *.mcx file names
			*/
			std::string _base_cache_name;
			std::string _cache_directory;
		private:
		};
	} // namespace nCache
}

