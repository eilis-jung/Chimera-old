#include "Interpolation/MeanValueInterpolant3D.h"

namespace Chimera {

	namespace Interpolation {
		#pragma region Constructors
		template<class valueType>
		MeanValueInterpolant3D<valueType>::MeanValueInterpolant3D(const Array3D<valueType> &values, CutVoxels3D<Vector3> *pCutVoxels, 
																	CutVoxelsVelocities3D *pCutVoxelsVelocities, Scalar gridDx, bool useAuxVels /*= false*/) :
																	Interpolant(values){
			m_useAuxiliaryVelocities = useAuxVels;
			//Initialize nodal interpolant
			m_pNodalInterpolant = new BilinearNodalInterpolant3D<valueType>(values, gridDx);
			
			//Initialize cut-cells and velocities
			m_pCutVoxels = pCutVoxels;
			m_pCutVoxelsVelocity = pCutVoxelsVelocities;
			m_dx = gridDx;
		}
		#pragma endregion

		#pragma region Functionalities
		template<class valueType>
		valueType MeanValueInterpolant3D<valueType>::interpolate(const Vector3 &position) {
			Vector3 gridSpacePosition = position / m_dx;
			int i = floor(gridSpacePosition.x); int j = floor(gridSpacePosition.y); int k = floor(gridSpacePosition.z);
			if (m_pCutVoxels && m_pCutVoxels->isCutVoxel(i, j, k)) { //Actually implementing MVC
				return interpolateCutVoxel(m_pCutVoxels->getCutVoxelIndex(gridSpacePosition), position);
			} else {
				return m_pNodalInterpolant->interpolate(position);
			}
		}
		
		template<class valueType>
		void MeanValueInterpolant3D<valueType>::updateNodalVelocities(const Array3D<Vector3> &sourceStaggered, Array3D<Vector3> &targetNodal, bool useAuxVels) {
			//Update regular grid nodal velocities first
			m_pNodalInterpolant->staggeredToNodeCentered(sourceStaggered, targetNodal, m_pCutVoxels, useAuxVels);
			
			m_pCutVoxelsVelocity->update(targetNodal, useAuxVels);
		}

		#pragma endregion
		#pragma region PrivateFunctionalities		
		/** Vector-based nodal interpolation in regular grids */
		template<class valueType>
		valueType MeanValueInterpolant3D<valueType>::interpolateCutVoxel(int ithCutCell, const Vector3 &position) {
			
			//const vector<Vector3D> &nodalPoints = mesh.getPoints();
			//const vector<typename Data::Mesh3D<Vector3D>::meshPolygon_t> &polygons = mesh.getTriangleMeshPolygons();
			
			auto currCutVoxel = m_pCutVoxels->getCutVoxel(ithCutCell);
			uint numberVertices = currCutVoxel.getVerticesMap().size();

			// Auxiliary vectors
			vector<DoubleScalar> weights(numberVertices, 0);
			vector<DoubleScalar> dist(numberVertices, 0);
			vector<Vector3> uVec(numberVertices);
			vector<Vector3> vfs(currCutVoxel.getHalfFaces().size());
			vector<vector<DoubleScalar>> perFaceLambdas(currCutVoxel.getHalfFaces().size());
			map<uint, DoubleScalar> weightMap;

			static const double eps = 1e-6;

			
			/* Calculate VF for each face*/
			for (int i = 0; i < currCutVoxel.getHalfFaces().size(); i++) {
				Vector3 vf;

				/** First compute VFs per face and the total denominator */
				auto halfEdges = currCutVoxel.getHalfFaces()[i]->getHalfEdges();
				for (int j = 0; j < halfEdges.size(); j++) {
					Vector3 currV = halfEdges[j]->getVertices().first->getPosition() - position;
					Vector3 nextV = halfEdges[j]->getVertices().second->getPosition() - position;

					Vector3 currNormal = currV.cross(nextV).normalized();
					Scalar normalizedDot = currV.dot(nextV) / (currV.length()*nextV.length());
					DoubleScalar tetaAngle = acos(clamp<Scalar>(normalizedDot, -(1.f - eps), 1.f - eps));
					if (abs(tetaAngle) < eps) {
						vf = currV;
						if (m_useAuxiliaryVelocities)
							return valueType(	halfEdges[j]->getVertices().first->getAuxiliaryVelocity().x,
												halfEdges[j]->getVertices().first->getAuxiliaryVelocity().y,
												halfEdges[j]->getVertices().first->getAuxiliaryVelocity().z);
						else
							return valueType(	halfEdges[j]->getVertices().first->getVelocity().x,
												halfEdges[j]->getVertices().first->getVelocity().y,
												halfEdges[j]->getVertices().first->getVelocity().z);
					}
					vf += (currNormal)* 0.5 * tetaAngle;
				}

				vector<DoubleScalar> currFaceLambdas(halfEdges.size());
				vfs[i] = vf;
				Vector3 v = vfs[i].normalized();

				DoubleScalar denominator = 0;
				for (int j = 0; j < halfEdges.size(); j++) {
					int prevJ = roundClamp<int>(j - 1, 0, halfEdges.size());

					Vector3 prevV = halfEdges[prevJ]->getVertices().first->getPosition() - position;
					Vector3 currV = halfEdges[j]->getVertices().first->getPosition() - position;
					Vector3 nextV = halfEdges[j]->getVertices().second->getPosition() - position;

					//Calculate lambda first
					DoubleScalar lambda = vfs[i].length() / (currV.length() + eps);

					//Normalize everythin
					prevV.normalize();
					currV.normalize();
					nextV.normalize();

					DoubleScalar tetaAngle = acos(clamp<Scalar>(v.dot(currV), -(1.f - eps), 1.f - eps));
					prevV = v.cross(prevV).normalized();
					currV = v.cross(currV).normalized();
					nextV = v.cross(nextV).normalized();
					
					
					DoubleScalar prevAlfaAngle	= acos(clamp<Scalar>(prevV.dot(currV),	-(1.f - eps), 1.f - eps));
					DoubleScalar alfaAngle		= acos(clamp<Scalar>(currV.dot(nextV),	-(1.f - eps), 1.f - eps));

					DoubleScalar tempLambda = std::tan(prevAlfaAngle*0.5) + std::tan(alfaAngle*0.5);
					lambda *= tempLambda / (std::sin(tetaAngle) + eps);

					denominator += tempLambda*std::tan(1.57079632679489661923 - tetaAngle);
					currFaceLambdas[j] = lambda;
				}

				/** Point is exactly on the face, break the search on all polygons, return current weights undiviced by denominator */
				if (abs(denominator) < eps) {
					perFaceLambdas[i] = currFaceLambdas;
					DoubleScalar sumCurrLambdas = 0;
					for (int j = 0; j < currFaceLambdas.size(); j++) {
						sumCurrLambdas += currFaceLambdas[j];
					}
					//if (abs(sumCurrLambdas) < eps) {
					for (int j = 0; j < currFaceLambdas.size(); j++) {
						currFaceLambdas[j] *= 1 / ((DoubleScalar)currFaceLambdas.size());
					}
					Vector3 velocity;
					for (int j = 0; j < currFaceLambdas.size(); j++) {
						if (m_useAuxiliaryVelocities)
							velocity += halfEdges[j]->getVertices().first->getAuxiliaryVelocity() * currFaceLambdas[j];
						else
							velocity += halfEdges[j]->getVertices().first->getVelocity() * currFaceLambdas[j];
					}
					return valueType(velocity.x, velocity.y, velocity.z);
					//}
				}

				for (int j = 0; j < currFaceLambdas.size(); j++) {
					currFaceLambdas[j] /= denominator;
				}

				//TODO: add verifications back

				//Vector3 otherVerification;
				//for (int j = 0; j < polygons[i].edges.size(); j++) {
				//	otherVerification += convertToVector3F(vertices[polygons[i].edges[j].first] - transformedPosition)*currFaceLambdas[j];
				//}

				//DoubleScalar weightsSum = 0;
				//for (int j = 0; j < polygons[i].edges.size(); j++) {
				//	weightsSum += currFaceLambdas[j];
				//}

				///*if (weightsSum < 0.0f) {
				//for (int j = 0; j < polygons[i].edges.size(); j++) {
				//currFaceLambdas[j] = 0;
				//}
				//}*/

				perFaceLambdas[i] = currFaceLambdas;
			}

			Vector3 sumVF;
			for (int i = 0; i < vfs.size(); i++) {
				sumVF += vfs[i];
			}

			for (int i = 0; i < currCutVoxel.getHalfFaces().size(); i++) {
				auto halfEdges = currCutVoxel.getHalfFaces()[i]->getHalfEdges();
				for (int j = 0; j < halfEdges.size(); j++) {
					weightMap[halfEdges[j]->getVertices().first->getID()] += perFaceLambdas[i][j];
				}
			}

			DoubleScalar allWeightsSum = 0;
			for (auto iter = weightMap.begin(); iter != weightMap.end(); iter++) {
				allWeightsSum += iter->second;
			}

			if (abs(allWeightsSum) > eps) {
				for (auto iter = weightMap.begin(); iter != weightMap.end(); iter++) {
					iter->second /= allWeightsSum;
				}
			}
			Vector3 velocity;
			for (auto iter = weightMap.begin(); iter != weightMap.end(); iter++) {
				auto vertexIter = currCutVoxel.getVerticesMap().find(iter->first);
				if (!m_useAuxiliaryVelocities)
					velocity += vertexIter->second->getVelocity()*iter->second;
				else
					velocity += vertexIter->second->getAuxiliaryVelocity()*iter->second;
			}




			//DoubleScalar verification = 0;
			//for (int i = 0; i < weights.size(); i++) {
			//	verification += Vector3D(weights[i], weights[i], weights[i]).dot(convertToVector3D(vertices[i]) - convertToVector3D(position));
			//}

			return valueType(velocity.x, velocity.y, velocity.z);
		}
		#pragma endregion

		/** Template linker trickerino for templated classes in CPP*/
		template class MeanValueInterpolant3D<Vector3>;
		template class MeanValueInterpolant3D<Vector3D>;

	}
}