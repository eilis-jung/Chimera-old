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

#ifndef __CHIMERA_RENDERING_H__
#define __CHIMERA_RENDERING_H__

#pragma once

#include "Primitives/Camera.h"

/************************************************************************/
/* Resources                                                            */
/************************************************************************/
#include "Resources/ResourceManager.h"

/************************************************************************/
/* Visualization                                                        */
/************************************************************************/
#include "Visualization/QuadGridRenderer.h"
#include "Visualization/HexaGridRenderer.h"
#include "Visualization/ScalarFieldRenderer.h"
//#include "Visualization/CutCellsRenderer2D.h"
#include "Visualization/CutCellsRenderer3D.h"
#include "Visualization/MeshRenderer3D.h"
#include "Visualization/MeshRenderer.h"
#include "Visualization/VolumeMeshRenderer.h"
#include "Visualization/PolygonMeshRenderer.h"
#include "Visualization/LineMeshRenderer.h"
#include "Visualization/CutCellsVelocityRenderer2D.h"
#include "Visualization/IsocontourRenderer.h"
//#include "Visualization/RaycastRenderer2D.h"


#include "GLRenderer2D.h"
#include "GLRenderer3D.h"
#include "RenderingUtils.h"
#include "Particles/ParticleSystem2D.h"
#include "Particles/ParticleSystem3D.h"
#include "Particles/ParticlesRenderer.h"

#include "Primitives/Circle.h"
#include "Primitives/Line.h"

#endif