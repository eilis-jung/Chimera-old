#pragma once

#include "AbstractMemoryWriter.h"
#include <string>

namespace Chimera
{
	namespace nCache
	{

		class MCXMemoryWriter : public AbstractMemoryWriter
		{
		public:
			MCXMemoryWriter(const std::string& i_mcx_filename, const nCache::Header& o_header, const ChannelInfoContainer* i_channels_info = 0);
			virtual ~MCXMemoryWriter();

			bool write_header(Header& o_header);
			bool write_channel(nCache::ChannelDataContainer channels);
		};

	} // namespace nCache
}
