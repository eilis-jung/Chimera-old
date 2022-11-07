//This class is used to export particles of the simulation. The Exporter
//can help to load particles data into maya with nParticle tool

#ifndef __CHIMERA_NPARTICLE_EXPORTER_
#define __CHIMERA_NPARTICLE_EXPORTER_

#pragma once

#include "ChimeraRendering.h"
#include "ChimeraCore.h"


namespace Chimera {
	using namespace Core;

	//template <class VectorType, template <class> class ArrayType>
	class NParticleExporter {
	public:
		#pragma region Constructors
		NParticleExporter(ParticleSystem3D* particle_system3, Scalar endtime, Scalar samplingrate);
		~NParticleExporter() {
		}
		#pragma endregion
	
		void dumpFrame();
	private:
		void dumpXML();
		ParticleSystem3D* m_pParticleSystem3D;
		std::vector<Vector3>* m_pPositions;
		Scalar m_endtime;
		Scalar m_samplingrate;
		unsigned int m_currFrame;
	};
}

#endif