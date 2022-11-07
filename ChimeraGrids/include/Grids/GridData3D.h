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

#ifndef __CHIMERA_GRID_DATA_3D__
#define __CHIMERA_GRID_DATA_3D__
#pragma once

#include "ChimeraCore.h"
#include "GridData.h"

namespace Chimera {

	using namespace Core;

	namespace Grids { 

		/** GridData3D class. Abstracts data for both 2D and 3D grids.
		** IMPORTANT: GridData3D class is only a data storage class. It is NOT RESPONSIBLE for initialization of variables,
		** grid loading, metrics or density points/values.*/

		class GridData3D : public GridData<Vector3> {

		public:

			/** Default ctor*/
			FORCE_INLINE GridData3D(Array3D<Vector3> *gridPoints, dimensions_t dimensions) : GridData<Vector3>(dimensions),
				m_gridCenters(dimensions),
				m_velocities(dimensions), m_auxVelocities(dimensions),
				m_pressures(dimensions), m_divergents(dimensions), m_vorticities(dimensions), m_levelSetData(dimensions),
				m_kineticEnergy(dimensions), m_kineticEnergyChange(dimensions),
				m_temperatureBuffer(dimensions), m_densityBuffer(dimensions),
				m_scaleFactors(dimensions), m_volumes(dimensions),
				m_leftFaceAreas(dimensions), m_bottomFaceAreas(dimensions), m_backFaceAreas(dimensions),
				m_transformMatrices(dimensions), m_inverseTransformMatrices(dimensions),
				m_xiBaseNormals(dimensions), m_etaBaseNormals(dimensions), m_talBaseNormals(dimensions) {
				m_gridPoints = gridPoints;
				int vectorSize;
				vectorSize = m_dimensions.x*m_dimensions.y;

				initData();
				initGridCenters();
				initGridBoundaries();
			}

			/************************************************************************/
			/* 2D Access functions - Velocities and pressures                       */
			/************************************************************************/
			/** Velocity */
			FORCE_INLINE const Vector3 & getVelocity(int x, int y, int z) const {
				return m_velocities(x, y, z);
			}
			FORCE_INLINE void setVelocity(const Vector3 &velocity, int x, int y, int z) {
				m_velocities(x, y, z) = velocity;
			}

			FORCE_INLINE const Vector3 & getAuxiliaryVelocity(int x, int y, int z) const {
				return m_auxVelocities(x, y, z);
			}
			FORCE_INLINE void setAuxiliaryVelocity(const Vector3 &velocity, int x, int y, int z) {
				m_auxVelocities(x, y, z) = velocity;
			}

			FORCE_INLINE void setAuxiliaryVelocity(const Scalar velocity, int x, int y, int z, velocityComponent_t velocityComponent) {
				m_auxVelocities(x, y, z)[velocityComponent] = velocity;
			}

			/** Pressure */
			FORCE_INLINE Scalar getPressure(int x, int y, int z) const {
				return m_pressures(x, y, z);
			}
			FORCE_INLINE void setPressure(Scalar pressure, int x, int y, int z) {
				m_pressures(x, y, z) = pressure;
			}

			/** Divergents */
			FORCE_INLINE Scalar getDivergent(int x, int y, int z) const {
				return m_divergents(x, y, z);
			}

			FORCE_INLINE void setDivergent(Scalar divergent, int x, int y, int z) {
				m_divergents(x, y, z) = divergent;
			}

			/** Vorticity */
			FORCE_INLINE Scalar getVorticity(int x, int y, int z) const {
				return m_vorticities(x, y, z);
			}

			FORCE_INLINE void setVorticity(Scalar vorticity, int x, int y, int z) {
				m_vorticities(x, y, z) = vorticity;
			}

			/** Level set */
			FORCE_INLINE Scalar getLevelSetValue(int x, int y, int z) const {
				return m_levelSetData(x, y, z);
			}

			FORCE_INLINE void setLevelSetValue(Scalar lsValue, int x, int y, int z) {
				m_levelSetData(x, y, z) = lsValue;
			}

			/** Kinetic Energy */
			FORCE_INLINE Scalar getKineticEnergyValue(int x, int y, int z) const {
				return m_kineticEnergy(x, y, z);
			}

			FORCE_INLINE void setKineticEnergyValue(Scalar kineticEnergy, int x, int y, int z) {
				m_kineticEnergy(x, y, z) = kineticEnergy;
			}

			/** Kinetic Energy Difference */
			FORCE_INLINE Scalar getKineticEnergyChangeValue(int x, int y, int z) const {
				return m_kineticEnergyChange(x, y, z);
			}

			FORCE_INLINE void setKineticEnergyChangeValue(Scalar kineticEnergyChange, int x, int y, int z) {
				m_kineticEnergyChange(x, y, z) = kineticEnergyChange;
			}

			/************************************************************************/
			/* 3D Access functions - Grid points			                        */
			/************************************************************************/
			/** Returns the cell 2D origin point */
			FORCE_INLINE const Vector3 & getPoint(int x, int y, int z) const {
				return (*m_gridPoints)(x, y, z);
			}

			/** Returns the cell 2D center point */
			FORCE_INLINE const Vector3 & getCenterPoint(int x, int y, int z) const {
				return m_gridCenters(x, y, z);
			}

			FORCE_INLINE void setCenterPoint(const Vector3 & centerPoint, int x, int y, int z) {
				m_gridCenters(x, y, z) = centerPoint;
			}

			/************************************************************************/
			/* 2D Access functions - Metrics and volumes							*/
			/************************************************************************/
			FORCE_INLINE const Vector3 & getScaleFactor(int x, int y, int z) const {
				return m_scaleFactors(x, y, z);
			}
			FORCE_INLINE void setScaleFactor(const Vector3 & scaleFactor, int x, int y, int z) {
				m_scaleFactors(x, y, z) = scaleFactor;
			}

			FORCE_INLINE Scalar getVolume(int x, int y, int z) const {
				return m_volumes(x, y, z);
			}
			FORCE_INLINE void setVolume(Scalar volume, int x, int y, int z) {
				m_volumes(x, y, z) = volume;
			}
			
			FORCE_INLINE Scalar getLeftFaceArea(int x, int y, int z) const {
				return m_leftFaceAreas(x, y, z);
			}
			FORCE_INLINE void setLeftFaceArea(Scalar leftFaceArea, int x, int y, int z) {
				m_leftFaceAreas(x, y, z) = leftFaceArea;
			}

			FORCE_INLINE Scalar getBottomFaceArea(int x, int y, int z) const {
				return m_bottomFaceAreas(x, y, z);
			}
			FORCE_INLINE void setBottomFaceArea(Scalar bottomFaceArea, int x, int y, int z) {
				m_bottomFaceAreas(x, y, z) = bottomFaceArea;
			}

			FORCE_INLINE Scalar getBackFaceArea(int x, int y, int z) const {
				return m_backFaceAreas(x, y, z);
			}
			FORCE_INLINE void setBackFaceArea(Scalar backFaceArea, int x, int y, int z) {
				m_backFaceAreas(x, y, z) = backFaceArea;
			}

			/************************************************************************/
			/* 2D Access functions - Normals										*/
			/************************************************************************/
			FORCE_INLINE const Vector3 & getXiBaseNormal(int x, int y, int z) const {
				return m_xiBaseNormals(x, y, z);
			}
			FORCE_INLINE void setXiBaseNormal(const Vector3 &xiBaseNormal, int x, int y, int z) {
				m_xiBaseNormals(x, y, z) = xiBaseNormal;
			}

			FORCE_INLINE const Vector3 & getEtaBaseNormal(int x, int y, int z) const {
				return m_etaBaseNormals(x, y, z);
			}
			FORCE_INLINE void setEtaBaseNormal(const Vector3 &etaBaseNormal, int x, int y, int z) {
				m_etaBaseNormals(x, y, z) = etaBaseNormal;
			}

			FORCE_INLINE const Vector3 & getTalBaseNormal(int x, int y, int z) const {
				return m_talBaseNormals(x, y, z);
			}
			FORCE_INLINE void setTalBaseNormal(const Vector3 &talBaseNormal, int x, int y, int z) {
				m_talBaseNormals(x, y, z) = talBaseNormal;
			}

			/************************************************************************/
			/* Transformation Matrices                                              */
			/************************************************************************/
			FORCE_INLINE const Matrix3x3 & getTransformationMatrix(int x, int y, int z) {
				return m_transformMatrices(x, y, z);
			}
			FORCE_INLINE void setTransformationMatrix(const Matrix3x3 &transformationMatrix, int x, int y, int z) {
				m_transformMatrices(x, y, z) = transformationMatrix; 
			}

			FORCE_INLINE const Matrix3x3 & getInverseTransformationMatrix(int x, int y, int z) {
				return m_inverseTransformMatrices(x, y, z);
			}
			FORCE_INLINE void setInverseTransformationMatrix(const Matrix3x3 &invTransformationMatrix, int x, int y, int z) {
				m_inverseTransformMatrices(x, y, z) = invTransformationMatrix; 
			}

			FORCE_INLINE const Array3D<Matrix3x3> & getTransformationMatrices() const {
				return m_transformMatrices;
			}

			FORCE_INLINE const Array3D<Matrix3x3> & getInverseTransformationMatrices() const {
				return m_inverseTransformMatrices;
			}
			
			/************************************************************************/
			/* Arrays Access Functions												*/
			/************************************************************************/
			const Array3D<Vector3> & getVelocityArray() const {
				return m_velocities;
			}
			Array3D<Vector3> * getVelocityArrayPtr() {
				return &m_velocities;
			}
			const Array3D<Vector3> & getAuxVelocityArray() const {
				return m_auxVelocities;
			}
			Array3D<Vector3> * getAuxVelocityArrayPtr() {
				return &m_auxVelocities;
			}
			const Array3D<Scalar> & getVolumeArray() const {
				return m_volumes;
			}
			const Array3D<Vector3> & getScaleFactorsArray() const {
				return m_scaleFactors;
			}
			const Array3D<Scalar> & getDivergentArray() const {
				return m_divergents;
			}
			const Array3D<Scalar> & getPressureArray() const {
				return m_pressures;
			}
			const Array3D<Scalar> & getLevelSetArray() const {
				return m_levelSetData;
			}
			const Array3D<Vector3> & getGridCentersArray() const {
				return m_gridCenters;
			}
			const Array3D<Vector3> & getGridPointsArray() const {
				return *m_gridPoints;
			}
			const Array3D<Scalar> & getVorticityArray() const {
				return m_vorticities;
			}
			const Array3D<Scalar> & getKineticEnergyArray() const {
				return m_kineticEnergy;
			}
			const Array3D<Scalar> & getKineticEnergyChangeArray() const {
				return m_kineticEnergyChange;
			}

			/************************************************************************/
			/* Buffer Access Functions												*/
			/************************************************************************/
			/** Temperature buffer. */
			FORCE_INLINE const DoubleBuffer<Scalar, Array3D> & getTemperatureBuffer() const {
				return m_temperatureBuffer;
			}
			FORCE_INLINE DoubleBuffer<Scalar, Array3D> & getTemperatureBuffer() {
				return m_temperatureBuffer;
			}
			/** Density buffer. */
			FORCE_INLINE const DoubleBuffer<Scalar, Array3D> & getDensityBuffer() const {
				return m_densityBuffer;
			}
			FORCE_INLINE DoubleBuffer<Scalar, Array3D> & getDensityBuffer() {
				return m_densityBuffer;
			}



			/************************************************************************/
			/* Utils                                                                */
			/************************************************************************/
			FORCE_INLINE const Vector3 & getMinBoundary() const {
				return m_minGridBoundary;
			}

			FORCE_INLINE const Vector3 & getMaxBoundary() const {
				return m_maxGridBoundary;
			}

			FORCE_INLINE dimensions_t getDimensions() const {
				return m_dimensions;
			}

		private:

			/************************************************************************/
			/* Initialization functions                                             */
			/************************************************************************/
			void initData() {
				Vector3 zeroVec;
				for(int i = 0; i < m_dimensions.x; i++) {
					for(int j = 0; j < m_dimensions.y; j++) {
						for(int k = 0; k < m_dimensions.z; k++) {
							setVelocity(zeroVec, i, j, k);
							setAuxiliaryVelocity(zeroVec, i, j, k);

							m_pressures(i, j, k) = 0;
							m_divergents(i, j, k) = 0;
							m_vorticities(i, j, k) = 0;
							m_levelSetData(i, j, k) = 0;
							m_kineticEnergy(i, j, k) = 0;
							m_kineticEnergyChange(i, j, k) = 0;

							m_densityBuffer.setValueBothBuffers(0, i, j, k);
							m_temperatureBuffer.setValueBothBuffers(0, i, j, k);
						}
					}
				}
			}

			void initGridCenters()  {
				Vector3 centerPoint;
				for(int k = 0; k < m_dimensions.z; k++) {
					for(int j = 0; j < m_dimensions.y; j++) {
						for(int i = 0; i < m_dimensions.x; i++) {
							centerPoint = (getPoint(i, j, k) + getPoint(i + 1, j, k) + getPoint(i + 1, j, k + 1) + getPoint(i, j, k + 1) 
								+ getPoint(i, j + 1, k) + getPoint(i + 1, j + 1, k) + getPoint(i + 1, j + 1, k + 1) + getPoint(i, j + 1, k + 1))*0.125f;
							setCenterPoint(centerPoint, i, j, k); 
						}
					}
				}	
			}

			void initGridBoundaries() {
				m_minGridBoundary = m_maxGridBoundary = getPoint(0, 0, 0);
				for(int k = 0; k <= m_dimensions.z; k++) {
					for(int j = 0; j <= m_dimensions.y; j++) {
						for(int i = 0; i <= m_dimensions.x; i++) {
							if(getPoint(i, j, k) < m_minGridBoundary) {
								m_minGridBoundary = getPoint(i, j, k);							
							} else if(getPoint(i, j, k) > m_maxGridBoundary) {
								m_maxGridBoundary = getPoint(i, j, k);
							}
						}
					}
				}
			}

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			/** Grid points array pointer. Since this structure is already initialized by an external loader, this class
			 ** is going to receive the points pointer.*/
			Array3D<Vector3> *m_gridPoints;
			/** Grid centers*/
			Array3D<Vector3> m_gridCenters;

			/** Grid velocity and auxiliary velocity array. */
			Array3D<Vector3> m_velocities, m_auxVelocities;

			/** Scalar fields*/
			Array3D<Scalar>	m_pressures, m_divergents, m_vorticities, m_levelSetData, m_kineticEnergy, m_kineticEnergyChange;
			
			/** Double buffers*/
			DoubleBuffer<Scalar, Array3D> m_temperatureBuffer, m_densityBuffer;

			/** Grid scale factors and volumes. In the discrete approximation, the components of the vector 
			are the lengths of the correspondent edges of the grid. */
			Array3D<Vector3> m_scaleFactors;
			Array3D<Scalar>	 m_volumes;
			Array3D<Scalar>  m_leftFaceAreas, m_bottomFaceAreas, m_backFaceAreas;

			/** Transformation matrices used on transformToCoordinateSystem function */
			Array3D<Matrix3x3> m_transformMatrices, m_inverseTransformMatrices;

			/** Base normals. Used only in non-Cartesian grids.*/
			Array3D<Vector3> m_xiBaseNormals, m_etaBaseNormals, m_talBaseNormals;

			/** Grid boundaries */
			Vector3			m_minGridBoundary, m_maxGridBoundary;

		};
	}
}

#endif