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

#ifndef __CHIMERA_ADVECTION_H_
#define __CHIMERA_ADVECTION_H_

#pragma once

/************************************************************************/
/* 2D                                                                   */
/************************************************************************/
#include "ParticleBased/GridToParticlesFLIP2D.h"
#include "ParticleBased/GridToParticlesRPIC2D.h"
#include "ParticleBased/GridToParticlesAPIC2D.h"
#include "ParticleBased/GridToParticlesTurbulent2D.h"
#include "ParticleBased/ParticlesToStaggeredGrid2D.h"
#include "ParticleBased/ParticlesToNodalGrid2D.h"
#include "ParticleBased/ParticlesToStaggeredGridRPIC2D.h"
#include "ParticleBased/ParticlesToStaggeredGridAPIC2D.h"
#include "ParticleBased/TurbulentParticlesToGrid2D.h"

/************************************************************************/
/* 3-D                                                                  */
/************************************************************************/
#include "ParticleBased/GridToParticlesFLIP3D.h"
#include "ParticleBased/ParticlesToStaggeredGrid3D.h"
#include "ParticleBased/ParticlesToNodalGrid3D.h"

/************************************************************************/
/* Base                                                                 */
/************************************************************************/
#include "ParticleBased/ParticleBasedAdvection.h"

/************************************************************************/
/* Kernels                                                              */
/************************************************************************/
#include "Kernels/BilinearKernel.h"
#include "Kernels/SPHKernel.h"

/************************************************************************/
/* Integration                                                          */
/************************************************************************/
#include "Integration/ForwardEulerIntegrator.h"
#include "Integration/RungeKutta2Integrator.h"

/************************************************************************/
/* Grid-based advection                                                 */
/************************************************************************/
#include "GridBased/SemiLagrangianAdvection.h"
#include "GridBased/MacCormackAdvection.h"

#endif