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
#ifndef _CHIMERA_CORE_H__
#define _CHIMERA_CORE_H__

#if defined(_WIN32) 
	#define WIN32_LEAN_AND_MEAN
	#define NOMINMAX
	#include <Windows.h>
	#undef max //causes numerical limits to fail
#else
#endif

#include "Config/ChimeraConfig.h"

#define TW_NO_LIB_PRAGMA

/** Cuda & Cusp configs */
#include "Config/CudaConfig.h"
#include "Config/CuspConfig.h"

#include "Utils/Singleton.h"
#include "Utils/Logger.h"

#include "Data/Array2D.h"
#include "Data/Array3D.h"
#include "Data/ChimeraStructures.h"
#include "Data/DoubleBuffer.h"
#include "Data/ReferenceWrapper.h"

#include "Utils/Utils.h"
#include "Utils/Times.h"
#include "Utils/Timer.h"

#include "Math/DoubleScalar.h"
#include "Math/Integer.h"
#include "Math/MathUtils.h"
#include "Math/MathUtilsCore.h"
#include "Math/Scalar.h"
#include "Math/DoubleScalar.h"
#include "Math/Vector2.h"
#include "Math/Vector2d.h"
#include "Math/Vector3.h"
#include "Math/Vector3d.h"
#include "Math/Intersection.h"

#include "Math/Matrix/Matrix2x2.h"
#include "Math/Matrix/Matrix3x3.h"
#include "Math/Matrix/Matrix3x3d.h"
#include "Math/Matrix/MatrixNxN.h"
#include "Math/Matrix/MatrixNxNd.h"
#include "Math/Matrix/Quaternion.h"

#include "Physics/PhysicsCore.h"
#include "Physics/PhysicalObject.h"

#include "GL/glew.h"
#include "GL/glut.h"
#include "Rendering/Color.h"





#endif