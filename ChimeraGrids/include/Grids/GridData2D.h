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

#ifndef __CHIMERA_GRID_DATA_2D__
#define __CHIMERA_GRID_DATA_2D__
#pragma once

#include "ChimeraCore.h"
#include "GridData.h"

namespace Chimera {

	using namespace Core;

	namespace Grids { 

		/** GridData2D class. Abstracts data for both 2D and 3D grids.
		** IMPORTANT: GridData2D class is only a data storage class. It is NOT RESPONSIBLE for initialization of variables,
		** grid loading, metrics or density points/values.*/

		class GridData2D : public GridData<Vector2> {

		public:

			/** Default ctor*/
			FORCE_INLINE GridData2D(Array2D<Vector2> *gridPoints, dimensions_t dimensions) : GridData<Vector2>(dimensions),
				m_gridCenters(dimensions),
				m_velocities(dimensions), m_auxVelocities(dimensions), m_gradientField(dimensions),
				m_pressures(dimensions), m_divergents(dimensions), m_vorticities(dimensions), m_levelSetData(dimensions),
				m_kineticEnergy(dimensions), m_kineticEnergyChange(dimensions), m_streamfunctions(dimensions),
				m_temperatureBuffer(dimensions), m_densityBuffer(dimensions),
				m_scaleFactors(dimensions), m_volumes(dimensions),
				m_transformMatrices(dimensions), m_inverseTransformMatrices(dimensions),
				m_xiBaseNormals(dimensions), m_etaBaseNormals(dimensions) {
				m_gridPoints = gridPoints;
				int vectorSize;
				vectorSize = m_dimensions.x*m_dimensions.y;

				m_fineGridDx = 0.0f;
				m_pFineGridScalarField = nullptr;

				initData();
				initGridCenters();
				initGridBoundaries();
			}

			/************************************************************************/
			/* 2D Access functions - Velocities and pressures                       */
			/************************************************************************/
			/** Velocity */
			FORCE_INLINE const Vector2 & getVelocity(int x, int y) const {
				return m_velocities(x, y);
			}
			FORCE_INLINE void setVelocity(const Vector2 &velocity, int x, int y) {
				m_velocities(x, y) = velocity;
			}
			FORCE_INLINE void setVelocity(const Scalar velocity, int x, int y, velocityComponent_t velocityComponent) {
				m_velocities(x, y)[velocityComponent] = velocity;
			}
			FORCE_INLINE void addVelocity(const Vector2 &velocity, int x, int y) {
				m_velocities(x, y) += velocity;
			}
			FORCE_INLINE void addVelocityComponent(Scalar &velocity, int x, int y, velocityComponent_t velocityComponent) {
				switch (velocityComponent) {
					case xComponent:
						m_velocities(x, y).x = m_velocities(x, y).x + velocity;
					break;
					case yComponent:
						m_velocities(x, y).y = m_velocities(x, y).y + velocity;
					break;
				}
			}
			FORCE_INLINE void setVelocityArray(const Array2D<Vector2> &velocityArray) {
				m_velocities = velocityArray;
			}

			FORCE_INLINE const Vector2 & getAuxiliaryVelocity(int x, int y) const {
				return m_auxVelocities(x, y);
			}
			FORCE_INLINE void setAuxiliaryVelocity(const Vector2 &velocity, int x, int y) {
				m_auxVelocities(x, y) = velocity;
			}
			FORCE_INLINE void setAuxiliaryVelocity(const Scalar velocity, int x, int y, velocityComponent_t velocityComponent) {
				m_auxVelocities(x, y)[velocityComponent] = velocity;
			}
			FORCE_INLINE void addAuxiliaryVelocity(const Vector2 &velocity, int x, int y) {
				m_auxVelocities(x, y) += velocity;
			}
			FORCE_INLINE void addAuxiliaryVelocityComponent(Scalar velocity, int x, int y, velocityComponent_t velocityComponent) {
				switch (velocityComponent) {
				case xComponent:
					m_auxVelocities(x, y).x = m_auxVelocities(x, y).x + velocity;
					break;
				case yComponent:
					m_auxVelocities(x, y).y = m_auxVelocities(x, y).y + velocity;
					break;
				}
			}

			/** Pressure */
			FORCE_INLINE Scalar getPressure(int x, int y) const {
				return m_pressures(x, y);
			}
			FORCE_INLINE void setPressure(Scalar pressure, int x, int y) {
				m_pressures(x, y) = pressure;
			}

			/** Divergents */
			FORCE_INLINE Scalar getDivergent(int x, int y) const {
				return m_divergents(x, y);
			}

			FORCE_INLINE void setDivergent(Scalar divergent, int x, int y) {
				m_divergents(x, y) = divergent;
			}

			/** Vorticity */
			FORCE_INLINE Scalar getVorticity(int x, int y) const {
				return m_vorticities(x, y);
			}

			FORCE_INLINE void setVorticity(Scalar vorticity, int x, int y) {
				m_vorticities(x, y) = vorticity;
			}

			/** Level set */
			FORCE_INLINE Scalar getLevelSetValue(int x, int y) const {
				return m_levelSetData(x, y);
			}

			FORCE_INLINE void setLevelSetValue(Scalar lsValue, int x, int y) {
				m_levelSetData(x, y) = lsValue;
			}

			/** Kinetic Energy */
			FORCE_INLINE Scalar getKineticEnergyValue(int x, int y) const {
				return m_kineticEnergy(x, y);
			}

			FORCE_INLINE void setKineticEnergyValue(Scalar kineticEnergy, int x, int y) {
				m_kineticEnergy(x, y) = kineticEnergy;
			}

			/** Kinetic Energy Difference */
			FORCE_INLINE Scalar getKineticEnergyChangeValue(int x, int y) const {
				return m_kineticEnergyChange(x, y);
			}

			FORCE_INLINE void setKineticEnergyChangeValue(Scalar kineticEnergyChange, int x, int y) {
				m_kineticEnergyChange(x, y) = kineticEnergyChange;
			}

			/** Streamfunctions */
			FORCE_INLINE Scalar getStreamfunction(int x, int y) const {
				return m_streamfunctions(x, y);
			}

			FORCE_INLINE void setStreamfunction(Scalar streamfunction, int x, int y) {
				m_streamfunctions(x, y) = streamfunction;
			}

			/************************************************************************/
			/* 2D Access functions - Grid points			                        */
			/************************************************************************/
			FORCE_INLINE void setPoint(const Vector2 &point, int x, int y) {
				(*m_gridPoints)(x, y) = point;
			}
			/** Returns the cell 2D origin point */
			FORCE_INLINE const Vector2 & getPoint(int x, int y) const {
				return (*m_gridPoints)(x, y);
			}

			/** Returns the cell 2D center point */
			FORCE_INLINE const Vector2 & getCenterPoint(int x, int y) const {
				return m_gridCenters(x, y);
			}

			FORCE_INLINE void setCenterPoint(const Vector2 & centerPoint, int x, int y) {
				m_gridCenters(x, y) = centerPoint;
			}

			/************************************************************************/
			/* 2D Access functions - Metrics and volumes							*/
			/************************************************************************/
			FORCE_INLINE const Vector2 & getScaleFactor(int x, int y) const {
				return m_scaleFactors(x, y);
			}
			FORCE_INLINE void setScaleFactor(const Vector2 & scaleFactor, int x, int y) {
				m_scaleFactors(x, y) = scaleFactor;
			}

			FORCE_INLINE Scalar getVolume(int x, int y) const {
				return m_volumes(x, y);
			}
			FORCE_INLINE void setVolume(Scalar volume, int x, int y) {
				m_volumes(x, y) = volume;
			}
			/************************************************************************/
			/* 2D Access functions - Normals										*/
			/************************************************************************/
			FORCE_INLINE const Vector2 & getXiBaseNormal(int x, int y) const {
				return m_xiBaseNormals(x, y);
			}
			FORCE_INLINE void setXiBaseNormal(const Vector2 &xiBaseNormal, int x, int y) {
				m_xiBaseNormals(x, y) = xiBaseNormal;
			}

			FORCE_INLINE const Vector2 & getEtaBaseNormal(int x, int y) const {
				return m_etaBaseNormals(x, y);
			}
			FORCE_INLINE void setEtaBaseNormal(const Vector2 &etaBaseNormal, int x, int y) {
				m_etaBaseNormals(x, y) = etaBaseNormal;
			}

			/************************************************************************/
			/* Transformation Matrices                                              */
			/************************************************************************/
			FORCE_INLINE const Matrix2x2 & getTransformationMatrix(int x, int y) {
				return m_transformMatrices(x, y);
			}
			FORCE_INLINE void setTransformationMatrix(const Matrix2x2 &transformationMatrix, int x, int y) {
				m_transformMatrices(x, y) = transformationMatrix; 
			}

			FORCE_INLINE const Matrix2x2 & getInverseTransformationMatrix(int x, int y) {
				return m_inverseTransformMatrices(x, y);
			}
			FORCE_INLINE void setInverseTransformationMatrix(const Matrix2x2 &invTransformationMatrix, int x, int y) {
				m_inverseTransformMatrices(x, y) = invTransformationMatrix; 
			}

			/************************************************************************/
			/* Arrays Access Functions												*/
			/************************************************************************/
			const Array2D<Vector2> & getVelocityArray() const {
				return m_velocities;
			}
			Array2D<Vector2> & getVelocityArray()  {
				return m_velocities;
			}

			Array2D<Vector2> * getVelocityArrayPtr() {
				return &m_velocities;
			}
			const Array2D<Vector2> & getAuxVelocityArray() const {
				return m_auxVelocities;
			}
			Array2D<Vector2> * getAuxVelocityArrayPtr() {
				return &m_auxVelocities;
			}
			const Array2D<Scalar> & getVolumeArray() const {
				return m_volumes;
			}
			const Array2D<Vector2> & getScaleFactorsArray() const {
				return m_scaleFactors;
			}
			const Array2D<Scalar> & getDivergentArray() const {
				return m_divergents;
			}
			const Array2D<Scalar> & getPressureArray() const {
				return m_pressures;
			}
			const Array2D<Scalar> & getLevelSetArray() const {
				return m_levelSetData;
			}
			Array2D<Scalar> * getLevelSetArrayPtr() {
				return &m_levelSetData;
			}
			const Array2D<Vector2> & getGridCentersArray() const {
				return m_gridCenters;
			}
			const Array2D<Vector2> & getGridPointsArray() const {
				return *m_gridPoints;
			}
			const Array2D<Scalar> & getVorticityArray() const {
				return m_vorticities;
			}
			const Array2D<Matrix2x2> &getTransformationMatrices() const {
				return m_transformMatrices;
			}
			const Array2D<Scalar> & getKineticEnergyArray() const {
				return m_kineticEnergy;
			}
			const Array2D<Scalar> & getKineticEnergyChangeArray() const {
				return m_kineticEnergyChange;
			}

			const Array2D<Scalar> & getStreamfunctionArray() const {
				return m_streamfunctions;
			}

			Array2D<Scalar> * getFineGridScalarFieldArrayPtr() {
				return m_pFineGridScalarField;
			}

			Scalar getFineGridScalarFieldDx() const {
				return m_fineGridDx;
			}

			void setFineGridScalarValue(Scalar value, int i, int j) {
				(*m_pFineGridScalarField)(i, j) = value;
			}
 
			/************************************************************************/
			/* Buffer Access Functions												*/
			/************************************************************************/
			/** Temperature buffer. */
			FORCE_INLINE const DoubleBuffer<Scalar, Array2D> & getTemperatureBuffer() const {
				return m_temperatureBuffer;
			}
			FORCE_INLINE DoubleBuffer<Scalar, Array2D> & getTemperatureBuffer()  {
				return m_temperatureBuffer;
			}
			/** Density buffer. */
			FORCE_INLINE const DoubleBuffer<Scalar, Array2D> & getDensityBuffer() const {
				return m_densityBuffer;
			}
			FORCE_INLINE DoubleBuffer<Scalar, Array2D> & getDensityBuffer()  {
				return m_densityBuffer;
			}

			/************************************************************************/
			/* Utils                                                                */
			/************************************************************************/
			FORCE_INLINE const Vector2 & getMinBoundary() const {
				return m_minGridBoundary;
			}

			FORCE_INLINE void setMinBoundary(const Vector2 &minBoundary) {
				m_minGridBoundary = minBoundary;
			}

			FORCE_INLINE const Vector2 & getMaxBoundary() const {
				return m_maxGridBoundary;
			}

			FORCE_INLINE void setMaxBoundary(const Vector2 &maxBoundary) {
				m_maxGridBoundary = maxBoundary;
			}

			FORCE_INLINE dimensions_t getDimensions() const {
				return m_dimensions;
			}

			/** Initializes fine grid scalar field values with the number of the specified number of subdivisions
				Scalar fine grid will have effective dimensions multiplied in each direction by pow(2, numSubdivisions).
			*/
			FORCE_INLINE void initializeFineGridScalarField(unsigned int numSubdivisions) {
				int scaleFactor = pow(2, numSubdivisions);
				m_pFineGridScalarField = new Array2D<Scalar>(dimensions_t(m_dimensions.x*scaleFactor, m_dimensions.y*scaleFactor));
				m_pFineGridScalarField->assign(0.0f);
				m_fineGridDx = m_scaleFactors(0, 0).x / scaleFactor;
			}
			void initDivergence() {
				for (int i = 0; i < m_dimensions.x; i++) {
					for (int j = 0; j < m_dimensions.y; j++) {
						m_divergents(i, j) = 0;
					}
				}
			}
			void initPressure() {
				for (int i = 0; i < m_dimensions.x; i++) {
					for (int j = 0; j < m_dimensions.y; j++) {
						m_pressures(i, j) = 0;
					}
				}
			}
		private:

			/************************************************************************/
			/* Initialization functions                                             */
			/************************************************************************/
			void initData() {
				Vector2 zeroVec;
				for(int i = 0; i < m_dimensions.x; i++) {
					for(int j = 0; j < m_dimensions.y; j++) {
						setVelocity(zeroVec, i, j);
						setAuxiliaryVelocity(zeroVec, i, j);

						m_pressures(i, j) = 0;
						m_divergents(i, j) = 0;
						m_vorticities(i, j) = 0;
						m_levelSetData(i, j) = 0;
						m_kineticEnergy(i, j) = 0;
						m_kineticEnergyChange(i, j) = 0;

						m_densityBuffer.setValueBothBuffers(0, i, j);
						m_temperatureBuffer.setValueBothBuffers(0, i, j);
					}
				}
			}

			void initGridCenters()  {
				Vector2 centerPoint;
				for(int j = 0; j < m_dimensions.y; j++) {
					for(int i = 0; i < m_dimensions.x; i++) {
						centerPoint = (getPoint(i, j) + getPoint(i + 1, j) + 
							getPoint(i + 1, j + 1) + getPoint(i, j + 1))*0.25f;
						setCenterPoint(centerPoint, i, j);
					}
				}
			}

			void initGridBoundaries() {
				m_minGridBoundary = m_maxGridBoundary = getPoint(0, 0);
				for(int j = 0; j <= m_dimensions.y; j++) {
					for(int i = 0; i <= m_dimensions.x; i++) {
						if(getPoint(i, j) < m_minGridBoundary) {
							m_minGridBoundary = getPoint(i, j);							
						} else if(getPoint(i, j) > m_maxGridBoundary) {
							m_maxGridBoundary = getPoint(i, j);
						}
					}
				}
			}

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			/** Grid points array pointer. Since this structure is already initialized by an external loader, this class
			 ** is going to receive the points pointer.*/
			Array2D<Vector2> *m_gridPoints;
			/** Grid centers*/
			Array2D<Vector2> m_gridCenters;

			/** Grid velocity and auxiliary velocity array. */
			Array2D<Vector2> m_velocities, m_auxVelocities;

			/** Gradient field */
			Array2D<Vector2> m_gradientField;

			/** Scalar fields*/
			Array2D<Scalar>	m_pressures, m_divergents, m_vorticities, m_levelSetData, m_kineticEnergy, m_kineticEnergyChange, m_streamfunctions;

			/** Fine grid scalar field */
			Array2D<Scalar> *m_pFineGridScalarField;
			Scalar m_fineGridDx;
			

			/** Double buffers*/
			DoubleBuffer<Scalar, Array2D> m_temperatureBuffer, m_densityBuffer;

			/** Grid scale factors and volumes. In the discrete approximation, the components of the vector 
			are the lengths of the correspondent edges of the grid. */
			Array2D<Vector2> m_scaleFactors;
			Array2D<Scalar>	 m_volumes;

			/** Transformation matrices used on transformToCoordinateSystem function */
			Array2D<Matrix2x2> m_transformMatrices, m_inverseTransformMatrices;

			/** Base normals. Used only in non-Cartesian grids.*/
			Array2D<Vector2> m_xiBaseNormals, m_etaBaseNormals;

			/** Grid boundaries */
			Vector2			m_minGridBoundary, m_maxGridBoundary;

		};
	}
}

#endif