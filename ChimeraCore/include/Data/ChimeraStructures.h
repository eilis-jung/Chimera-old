//  Copyright (c) 2013, Vinicius Costa Azevedo
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met: 
//
//1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer. 
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution. 
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//	The views and conclusions contained in the software and documentation are those
//	of the authors and should not be interpreted as representing official policies, 
//	either expressed or implied, of the FreeBSD Project.

#pragma once
#ifndef __CHIMERA_STRUCTURES_H__
#define __CHIMERA_STRUCTURES_H__

namespace Chimera {

	/************************************************************************/
	/* Fundamental structures                                               */
	/************************************************************************/

	/**Defines the platforms on which the may algorithms run.*/
	typedef enum plataform_t {
		PlataformCPU,
		PlataformGPU
	} plataform_t;

	/** Struct for (maximum) three dimensions indexes. Convention adopted:
			** i - iterates along the X direction (m_dimX).
			** j - iterates along the Y direction (m_dimY).
			** k - iterates along the Z direction (m_dimZ). */
	typedef struct dimensions_t {
		int x, y, z;

		dimensions_t() {
			x = y = z = 0;
		}

		dimensions_t(int gX, int gY) {
			x = gX; y = gY; z = 0;
		}

		dimensions_t(int gX, int gY, int gZ) {
			x = gX; y = gY; z = gZ;
		}

		dimensions_t(const dimensions_t &rhs) {
			x = rhs.x; y = rhs.y; z = rhs.z;
		}

		/************************************************************************/
		/* Operators                                                            */
		/************************************************************************/
		// Operators
		// Array indexing
		FORCE_INLINE int &operator [] (unsigned int i) {
			assert(i < 3);
			return *(&x + i);
		}

		// Array indexing
		FORCE_INLINE const int &operator [] (unsigned int i) const {
			assert(i < 3);
			return *(&x + i);
		}

		dimensions_t operator +(dimensions_t rhs) {
			return dimensions_t(x + rhs.x, y + rhs.y, z + rhs.z);
		}

		FORCE_INLINE dimensions_t & operator+=(const dimensions_t& rhs) {
			x += rhs.x; y += rhs.y; z += rhs.z;
			return *this;
		}
	
		FORCE_INLINE dimensions_t friend operator +(const dimensions_t &lhs, const dimensions_t &rhs) {
			dimensions_t dim(lhs);
			dim += rhs;
			return dim;
		}

		FORCE_INLINE bool operator==(const dimensions_t &rhs) {
			return(x == rhs.x && y == rhs.y && z == rhs.z);
		}

		FORCE_INLINE bool friend operator==(const dimensions_t &lhs, const dimensions_t &rhs) {
			return(lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
		}

		FORCE_INLINE bool operator!=(const dimensions_t &rhs) {
			return !(*this == rhs);
		}

		FORCE_INLINE bool friend operator!=(const dimensions_t &lhs, const dimensions_t &rhs) {
			return !(lhs == rhs);
		}

		FORCE_INLINE bool operator<(const dimensions_t &rhs) {
			if (this->y < rhs.y)
				return true;
			else if (this->y == rhs.y && this->x < rhs.x)
				return true;
			else
				return false;
		}

		FORCE_INLINE bool friend operator<(const dimensions_t &lhs, const dimensions_t &rhs) {
			if (lhs.y < rhs.y)
				return true;
			else if (lhs.y == rhs.y && lhs.x < rhs.x)
				return true;
			else
				return false;
		}
		FORCE_INLINE bool operator>(const dimensions_t &rhs) {
			if (this->y > rhs.y)
				return true;
			else if (this->y == rhs.y && this->x > rhs.x)
				return true;
			else
				return false;
		}

		FORCE_INLINE bool friend operator>(const dimensions_t &lhs, const dimensions_t &rhs) {
			if (lhs.y > rhs.y)
				return true;
			else if (lhs.y == rhs.y && lhs.x > rhs.x)
				return true;
			else
				return false;
		}
	} dimensions_t;


	typedef enum velocityComponent_t {
		xComponent,
		yComponent,
		zComponent,
		fullVector
	}velocityComponent_t;

	typedef enum solidBoundaryType_t {
		Solid_FreeSlip,
		Solid_NoSlip,
		Solid_Interpolation,
		Solid_Extrapolate
	}solidBoundaryType_t;

	/** Possible locations of the boundary conditions */
	typedef enum boundaryLocation_t {
		North,
		South,
		West,
		East,
		Front,
		Back,
		None
	} boundaryLocation_t;


	typedef enum solverType_t {
		finiteDifferenceMethod,		//Only for regular grids
		cutCellMethod,
		cutCellSOMethod,
		raycastMethod,
		streamfunctionTurbulenceMethod,
		sharpLiquids,
		ghostLiquids,
		streamfunctionVorticity,
		finiteVolumeMethod			//Regular and non-regular grids
	} solverType_t;

	typedef enum pressureMethod_t {
		CPU_CG,
		GPU_CG,
		MultigridMethod,
		SOR,
		GaussSeidelMethod,
		EigenCG
	} pressureMethod_t;

	typedef enum gridBasedAdvectionMethod_t {
		SemiLagrangian,
		MacCormack,
		USCIP
	} gridBasedAdvectionMethod_t;

	typedef enum particleBasedAdvectionMethod_t {
		FLIP,
		RPIC,
		APIC
	} particleBasedAdvectionMethod_t;

	typedef enum advectionCategory_t {
		LagrangianAdvection, //ParticleBasedAdvection
		EulerianAdvection //GridBasedAdvection
	};

	typedef enum mixedNodeInterpolationType_t {
		Unweighted,
		WeightedExtraDimensions,
		WeightedNoExtraDimensions,
		FaceVelocity
	} mixedNodeInterpolationType_t;

	typedef enum integrationMethod_t {
		forwardEuler,
		RungeKutta_2,
		RungeKutta_4,
		RungeKutta_Adaptive
	} integrationMethod_t;

	typedef enum collisionDetectionMethod_t {
		continuousCollisionBrochu,
		continuousCollisionWang,
		cgalSegmentIntersection,
		noCollisionDetection
	} collisionDetectionMethod_t;

	typedef enum interpolationMethod_t {
		NotDefined,
		Linear,
		MeanValueCoordinates,
		Turbulence,
		LinearStreamfunction,
		SubdivisStreamfunction,
		QuadraticStreamfunction,
		CubicStreamfunction
	} interpolationMethod_t;


	typedef enum gridArrangement_t {
		staggeredArrangement,
		nodalArrangement
	};

	typedef enum kernelTypes_t {
		SPHkernel,
		bilinearKernel,
		inverseDistance
	};

	typedef enum particlesSampling_t {
		stratifiedSampling,
		poissonSampling
	};
}
#endif