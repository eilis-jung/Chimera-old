#include "MayaCache/NParticleExporter.h"
#include "MayaCache/XMLWriter.h"
#include "MayaCache/MCXMemoryWriter.h"

namespace Chimera {

#pragma region Constructors
	NParticleExporter::NParticleExporter(ParticleSystem3D* particle_system3, Scalar endtime, Scalar samplingrate):
	m_pParticleSystem3D(particle_system3),
	m_endtime(endtime),
	m_samplingrate(samplingrate){	
		m_currFrame = 1;
	}
#pragma region DumpingFunctions
	void NParticleExporter::dumpXML(){

		string xmlExportName("Flow Logs/3D/Particle Cache/NPARTICLESHAPE" + intToStr(1) + ".xml");
		//std::string xml_filename("D:\\nCacheWriter\\nCacheWriter\\Debug\\test.xml");
		nCache::XMLWriter xml_writer;
		xml_writer.init("OneFilePerFrame", "mcx", 0, m_endtime, m_samplingrate, "2.0");
		//xml_writer.addExtra("Particle Info for nParticleShape1:");
		nCache::ChannelInfo channel_info;
		channel_info._channel_name = "nParticleShape1_id";
		channel_info._channel_interpretation = "id";
		channel_info._sampling_type = nCache::ChannelInfo::SamplingType::REGULAR;
		channel_info._channel_type = "DoubleArray";
		channel_info._sampling_rate = m_samplingrate;
		channel_info._start_time = 250;
		channel_info._end_time = m_endtime;
		xml_writer.addChannel("channel0", channel_info);
		channel_info._channel_name = "nParticleShape1_count";
		channel_info._channel_interpretation = "count";
		channel_info._sampling_type = nCache::ChannelInfo::SamplingType::REGULAR;
		channel_info._channel_type = "DoubleArray";
		channel_info._sampling_rate = m_samplingrate;
		channel_info._start_time = 250;
		channel_info._end_time = m_endtime;
		xml_writer.addChannel("channel1", channel_info);
		channel_info._channel_name = "nParticleShape1_position";
		channel_info._channel_interpretation = "position";
		channel_info._sampling_type = nCache::ChannelInfo::SamplingType::REGULAR;
		channel_info._channel_type = "FloatVectorArray";
		channel_info._sampling_rate = m_samplingrate;
		channel_info._start_time = 250;
		channel_info._end_time = m_endtime;
		xml_writer.addChannel("channel2", channel_info);
		//initialize writer
		xml_writer.write(xmlExportName); 
	}

	void NParticleExporter::dumpFrame()
	{
		if (m_currFrame == 1 || m_currFrame == 180)
		{
			dumpXML();
			m_currFrame = 1;
		}
		
		nCache::Header o_header;
		size_t array_size = m_pParticleSystem3D->getRealNumberOfParticles();
		m_pPositions = m_pParticleSystem3D->getParticlePositionsVectorPtr();
		o_header.header_blob_size = 76;
		//  |---MYCH (Group)			// 4
		//  |     |---CHNM				// 4
		//  |     |---Dummy	number		// 8
		//  |     |---Channel name size	// 4
		//  |     |---Channel name		// flexible(24)
		//  |     |---SIZE				// 4
		//  |     |---Dummy	number		// 8
		//  |     |---Dummy	number		// 4
		//  |     |---Channel data size	// 4
		//  |     |---Dummy	number		// 4
		//  |     |---Channel data type	// 4
		//  |     |---Dummy	number		// 8
		//  |     |---Buffer size		// 4
		//  |     |---DVCA				// flexible (Double Vector Array)
		//  |     |..
		o_header.channels_blob_size = 4 + 56 + 24 + array_size * 8
			+ 56 + 24 + 1 * 8
			+ 56 + 32 + array_size * 12;
		o_header.VRSN = "0.1";
		o_header.STIM = m_samplingrate * m_currFrame;
		o_header.ETIM = m_samplingrate * m_currFrame;
		string framExportName("Flow Logs/3D/Particle Cache/NPARTICLESHAPE" + intToStr(1) + "Frame" + intToStr(m_currFrame) + ".mcx");
		nCache::MCXMemoryWriter mcx_writer(framExportName, o_header);
		mcx_writer.write_header(o_header);


		nCache::ChannelDataContainer channels_data;
		nCache::ChannelData o_id, o_count, o_position;
		o_id._type = nCache::ChannelDataType::DBLA;
		o_count._type = nCache::ChannelDataType::DBLA;
		o_position._type = nCache::ChannelDataType::FVCA;
		
		o_id._dbla.resize(array_size);
		o_id._real_size = array_size;

		o_count._dbla.resize(1);
		o_count._real_size = 1;

		o_position._fvca.resize(array_size);
		o_position._real_size = array_size;

		o_count._dbla[0] = array_size;

		for (int i = 0; i < array_size; i++)
		{
			o_id._dbla[i] = i;
			o_position._fvca[i].x = m_pPositions->at(i).x;
			o_position._fvca[i].y = m_pPositions->at(i).y;
			o_position._fvca[i].z = m_pPositions->at(i).z;
		}
		//velocity
		channels_data.insert(nCache::ChannelDataContainer::value_type("nParticleShape1_id", o_id));
		channels_data.insert(nCache::ChannelDataContainer::value_type("nParticleShape1_count", o_count));
		channels_data.insert(nCache::ChannelDataContainer::value_type("nParticleShape1_position", o_position));

		mcx_writer.write_channel(channels_data);
		mcx_writer.write(o_header);


		m_currFrame++;
	}

#pragma endregion
}