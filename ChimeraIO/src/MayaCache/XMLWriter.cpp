#include <string>
#include <iostream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

#include "MayaCache/XMLWriter.h"

namespace Chimera
{
	using namespace nCache;

	XMLWriter::XMLWriter()
		: _time_range_start(0)
		, _time_range_end(0)
		, _cacheTimePerFrame_TimePerFrame(0)
		, _particle_count_sampling_rate(0)
	{
	}

	XMLWriter::~XMLWriter()
	{
	}

	void XMLWriter::init(std::string cache_type, std::string cache_format, size_t start, size_t end, size_t timePerFrame, std::string version)
	{
		set_cache_type(cache_type);
		set_cache_format(cache_format);
		set_time_range_start(start);
		set_time_range_end(end);
		set_cacheTimePerFrame_TimePerFrame(timePerFrame);
		set_cache_version(version);
	}

	void XMLWriter::write(const std::string& i_ncache_xml_filename)
	{
		try {
			std::cout << boost::format("i_ncache_xml_filename '%1%'") % i_ncache_xml_filename << std::endl;
			boost::property_tree::ptree pt;

			// This is Autodesk convention
			boost::filesystem::path p(i_ncache_xml_filename);
			boost::filesystem::directory_entry d(p);

			_base_cache_name = p.stem().string();
			_cache_directory = p.parent_path().string();
			std::cout << boost::format("_base_cache_name '%1%', _cache_directory '%2%'") % _base_cache_name % _cache_directory << std::endl;

			// Write non Channel tags first
			pt.put("Autodesk_Cache_File.cacheType.<xmlattr>.Type", _cache_type);
			pt.put("Autodesk_Cache_File.cacheType.<xmlattr>.Format", _cache_format);
			pt.put("Autodesk_Cache_File.time.<xmlattr>.Range", (boost::format("%1%-%2%") % _time_range_start %_time_range_end).str());
			pt.put("Autodesk_Cache_File.cacheTimePerFrame.<xmlattr>.TimePerFrame", std::to_string(_cacheTimePerFrame_TimePerFrame));
			pt.put("Autodesk_Cache_File.cacheVersion.<xmlattr>.Version", _cacheVersion_Version);
			BOOST_FOREACH(const std::string &name, _extras)
				pt.put("Autodesk_Cache_File.extra", name);

			// Now, just focus on the channel information
			for (std::pair<const std::string, ChannelInfo> element : _channels)
			{
				pt.put((boost::format("Autodesk_Cache_File.Channels.%1%.<xmlattr>.ChannelName") % element.first).str(), element.second._channel_name);
				pt.put((boost::format("Autodesk_Cache_File.Channels.%1%.<xmlattr>.ChannelType") % element.first).str(), element.second._channel_type);
				pt.put((boost::format("Autodesk_Cache_File.Channels.%1%.<xmlattr>.ChannelInterpretation") % element.first).str(), element.second._channel_interpretation);
				if (element.second._sampling_type == ChannelInfo::SamplingType::REGULAR)
					pt.put((boost::format("Autodesk_Cache_File.Channels.%1%.<xmlattr>.SamplingType") % element.first).str(), "Regular");
				else if (element.second._sampling_type == ChannelInfo::SamplingType::IRREGULAR)
					pt.put((boost::format("Autodesk_Cache_File.Channels.%1%.<xmlattr>.SamplingType") % element.first).str(), "Iregular");
				pt.put((boost::format("Autodesk_Cache_File.Channels.%1%.<xmlattr>.SamplingRate") % element.first).str(), std::to_string(element.second._sampling_rate));
				pt.put((boost::format("Autodesk_Cache_File.Channels.%1%.<xmlattr>.StartTime") % element.first).str(), std::to_string(element.second._start_time));
				pt.put((boost::format("Autodesk_Cache_File.Channels.%1%.<xmlattr>.EndTime") % element.first).str(), std::to_string(element.second._end_time));
			}

			boost::property_tree::xml_writer_settings<std::string> settings('\t', 1);
			write_xml(i_ncache_xml_filename, pt, std::locale(), settings);
		}
		catch (const boost::property_tree::ptree_error& e)
		{
			std::cerr << e.what() << std::endl;
		}
	}
}
