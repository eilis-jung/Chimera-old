#include "CutCells/CutVoxels3D.h"

namespace Chimera {


	namespace CutCells {

		#pragma region Constructors
		template <class VectorType>
		CutVoxels3D<VectorType>::CutVoxels3D(const vector<PolygonalMesh<VectorType> *> &polygonalMeshes, Scalar gridSpacing, const dimensions_t &gridDimensions)
			: m_gridSpacing(gridSpacing), m_gridDimensions(gridDimensions) , m_volumesArray(gridDimensions), m_XZFaces(gridDimensions), m_YZFaces(gridDimensions), 
			m_XYFaces(gridDimensions), m_nodeVertices(gridDimensions), m_polyPatches(gridDimensions), m_horizontalEdges(gridDimensions, true), 
			m_verticalEdges(gridDimensions, true), m_transversalEdges(gridDimensions, true) {

			m_nodeVertices.assign(nullptr);
			m_volumesArray.assign(nullptr);
			m_polyMeshes = polygonalMeshes;

			buildPolyPatches();
			buildLineMeshes();
			buildNodeVertices();
			buildCutSlices();
			buildRegularGridEdges();
			buildFaces();
			buildVolumes();
			buildHalfVolumes();
			//Adds corrects half-faces to vertices
			buildVertexAdjacencies();
			//buildTriangleHalfVolumes();
		}
		#pragma endregion

		#pragma region Functionalities
		template<class VectorType>
		void CutVoxels3D<VectorType>::reinitialize(const vector<PolygonalMesh<VectorType>*> &polygonalMeshes) {

		}
		#pragma endregion

		#pragma region AccessFunctions
		template<class VectorType>
		uint CutVoxels3D<VectorType>::getCutVoxelIndex(const VectorType & position) {
			dimensions_t currCellIndex(floor(position.x), floor(position.y), floor(position.z));
			if (isCutVoxel(currCellIndex)) {
				return m_volumesArray(currCellIndex)->getHalfVolume(position*m_gridSpacing)->getID();
			}
			throw("Error: invalid cut-voxel look-up on getCutCellIndex.");
			return uint(UINT_MAX);
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template<class VectorType>
		void CutVoxels3D<VectorType>::addFacesToVertices(HalfFace<VectorType> *pHalfFace) {
			for (int m = 0; m < pHalfFace->getHalfEdges().size(); m++) {
				pHalfFace->getHalfEdges()[m]->getVertices().first->addConnectedFace(pHalfFace->getFace());
			}
		}
		#pragma endregion
		#pragma region InitializationFunctions
		template <class VectorType>
		void CutVoxels3D<VectorType>::buildPolyPatches() {
			for (int i = 0; i < m_gridDimensions.x - 1; i++) {
				for (int j = 0; j < m_gridDimensions.y - 1; j++) {
					for (int k = 0; k < m_gridDimensions.z - 1; k++) {

						for (int l = 0; l < m_polyMeshes.size(); l++) {
							if (m_polyMeshes[l]->getPatchesIndices(i, j, k).size() > 0) {
								m_polyPatches(i, j, k).push_back(pair<uint, vector<uint>>(l, m_polyMeshes[l]->getPatchesIndices(i, j, k)));
							}
						}
					}
				}
			}
		}

		template <class VectorType>
		void CutVoxels3D<VectorType>::buildNodeVertices() {
			for (int i = 0; i < m_gridDimensions.x - 1; i++) {
				for (int j = 0; j < m_gridDimensions.y - 1; j++) {
					for (int k = 0; k < m_gridDimensions.z - 1; k++) {
						if (m_polyPatches(i, j, k).size() > 0) {
							buildOnNodeVerticesFacesPatch(dimensions_t(i, j, k));

							if (m_nodeVertices(i, j, k) == nullptr) {
								createNodeVertex(dimensions_t(i, j, k));
							}

							if (m_nodeVertices(i + 1, j, k) == nullptr && m_polyPatches(i + 1, j, k).size() == 0) {
								createNodeVertex(dimensions_t(i + 1, j, k));
							}
							
							if (m_nodeVertices(i + 1, j + 1, k) == nullptr && m_polyPatches(i + 1, j + 1, k).size() == 0) {
								createNodeVertex(dimensions_t(i + 1, j + 1, k));
							}

							if (m_nodeVertices(i, j + 1, k) == nullptr && m_polyPatches(i, j + 1, k).size() == 0) {
								createNodeVertex(dimensions_t(i, j + 1, k));
							}

							// k + 1
							if (m_nodeVertices(i, j, k + 1) == nullptr && m_polyPatches(i, j, k + 1).size() == 0) {
								createNodeVertex(dimensions_t(i, j, k + 1));

							}
							if (m_nodeVertices(i + 1, j, k + 1) == nullptr && m_polyPatches(i + 1, j, k + 1).size() == 0) {
								createNodeVertex(dimensions_t(i + 1, j, k + 1));
							}

							if (m_nodeVertices(i + 1, j + 1, k + 1) == nullptr && m_polyPatches(i + 1, j + 1, k + 1).size() == 0) {
								createNodeVertex(dimensions_t(i + 1, j + 1, k + 1));
							}

							if (m_nodeVertices(i, j + 1, k + 1) == nullptr && m_polyPatches(i, j + 1, k + 1).size() == 0) {
								createNodeVertex(dimensions_t(i, j + 1, k + 1));
							}
						}
					}
				}
			}
		}

		template <class VectorType>
		void CutVoxels3D<VectorType>::buildOnNodeVerticesFacesPatch(const dimensions_t &cellIndex) {
			//Check if any of the vertices of the line mesh are on top of a grid nodes
			for (int i = 0; i < m_polyPatches(cellIndex).size(); i++) {
				uint polyMeshIndex = m_polyPatches(cellIndex)[i].first;
				auto polyMeshPatches = m_polyPatches(cellIndex)[i].second;

				/** Verifying if there's geometry nodes on top of grid nodes */
				/*for (int j = 0; j < m_polyPatches.size(); j++) {
					Face<VectorType> *pFace = m_polyMeshes[polyMeshIndex]->getElements()[polyMeshPatches[j]];
					HalfFace<VectorType> *pHalfFface = pFace->getHalfFaces().front();
					for (int k = 0; k < pHalfFface->getHalfEdges().size(); k++) {
						if(pHalfFface->getVertices().first->isOnGridNode()) {
							const VectorType &position = pHalfFface->getVertices().first->getPosition();
							dimensions_t nodeVertexDim(position.x / m_gridSpacing, position.y / m_gridSpacing, position.z / m_gridSpacing);
							m_nodeVertices(nodeVertexDim) = pHalfFface->getVertices().first;
						}
					}
				}*/
			}
		}

		template <class VectorType>
		void CutVoxels3D<VectorType>::buildLineMeshes() {
			/** Mesh slicer is responsible for this now*/
			for (int i = 0; i < m_polyMeshes.size(); i++) {
				auto currXYLineMeshes = m_polyMeshes[i]->getMeshSlicer()->getXYLineMeshes();
				for (auto iter = currXYLineMeshes.begin(); iter != currXYLineMeshes.end(); iter++) {
					m_XYLineMeshes[iter->first].insert(m_XYLineMeshes[iter->first].end(), iter->second.begin(), iter->second.end());
				}

				auto currXZLineMeshes = m_polyMeshes[i]->getMeshSlicer()->getXZLineMeshes();
				for (auto iter = currXZLineMeshes.begin(); iter != currXZLineMeshes.end(); iter++) {
					m_XZLineMeshes[iter->first].insert(m_XZLineMeshes[iter->first].end(), iter->second.begin(), iter->second.end());
				}


				auto currYZLineMeshes = m_polyMeshes[i]->getMeshSlicer()->getYZLineMeshes();
				for (auto iter = currYZLineMeshes.begin(); iter != currYZLineMeshes.end(); iter++) {
					m_YZLineMeshes[iter->first].insert(m_YZLineMeshes[iter->first].end(), iter->second.begin(), iter->second.end());
				}


			}
			//m_XYLineMeshes = m_polyMeshes.front()->getMeshSlicer()->getXYLineMeshes();
			//m_XZLineMeshes = m_polyMeshes.front()->getMeshSlicer()->getXZLineMeshes();
			//m_YZLineMeshes = m_polyMeshes.front()->getMeshSlicer()->getYZLineMeshes();
		}

		template <class VectorType>
		void CutVoxels3D<VectorType>::buildCutSlices() {
			/** Try to respect XY, XZ, YZ order */

			/** Building XYCutCells */
			dimensions_t xyGridDimensions(m_gridDimensions.x, m_gridDimensions.y);
			for (auto iter = m_XYLineMeshes.begin(); iter != m_XYLineMeshes.end(); iter++) {
				CutCellsBase<VectorType>::extStructures_t extStructures;
				extStructures.pHorizontalEdges = Array3D<vector<Edge<VectorType> *> *>::createArraySlicePtr(m_horizontalEdges, iter->first, Array3D<vector<Edge<VectorType> *> *>::XYSlice);
				extStructures.pVerticalEdges = Array3D<vector<Edge<VectorType> *> *>::createArraySlicePtr(m_verticalEdges, iter->first, Array3D<vector<Edge<VectorType> *> *>::XYSlice);
				extStructures.pNodeVertices = Array3D<Vertex<VectorType> *>::createArraySlicePtr(m_nodeVertices, iter->first, Array3D<Vertex<VectorType> *>::XYSlice);
				m_XYCutCells[iter->first] = new CutCells3D<VectorType>(extStructures, iter->second, m_gridSpacing, xyGridDimensions, XYFace);
				m_XYCutCells[iter->first]->initialize();	
			}

			/** Building YZCutCells */
			dimensions_t yzGridDimensions(m_gridDimensions.z, m_gridDimensions.y);
			for (auto iter = m_YZLineMeshes.begin(); iter != m_YZLineMeshes.end(); iter++) {
				CutCellsBase<VectorType>::extStructures_t extStructures;
				extStructures.pHorizontalEdges = Array3D<vector<Edge<VectorType> *> *>::createArraySlicePtr(m_transversalEdges, iter->first, Array3D<vector<Edge<VectorType> *> *>::YZSlice);
				extStructures.pVerticalEdges = Array3D<vector<Edge<VectorType> *> *>::createArraySlicePtr(m_verticalEdges, iter->first, Array3D<vector<Edge<VectorType> *> *>::YZSlice);
				extStructures.pNodeVertices = Array3D<Vertex<VectorType> *>::createArraySlicePtr(m_nodeVertices, iter->first, Array3D<Vertex<VectorType> *>::YZSlice);
				m_YZCutCells[iter->first] = new CutCells3D<VectorType>(extStructures, iter->second, m_gridSpacing, yzGridDimensions, YZFace);
				m_YZCutCells[iter->first]->initialize();
			}

			/** Building XZCutCells */
			dimensions_t xzGridDimensions(m_gridDimensions.x, m_gridDimensions.z);
			for (auto iter = m_XZLineMeshes.begin(); iter != m_XZLineMeshes.end(); iter++) {
				CutCellsBase<VectorType>::extStructures_t extStructures;
				extStructures.pHorizontalEdges = Array3D<vector<Edge<VectorType> *> *>::createArraySlicePtr(m_horizontalEdges, iter->first, Array3D<vector<Edge<VectorType> *> *>::XZSlice);
				extStructures.pVerticalEdges = Array3D<vector<Edge<VectorType> *> *>::createArraySlicePtr(m_transversalEdges, iter->first, Array3D<vector<Edge<VectorType> *> *>::XZSlice);
				extStructures.pNodeVertices = Array3D<Vertex<VectorType> *>::createArraySlicePtr(m_nodeVertices, iter->first, Array3D<Vertex<VectorType> *>::XZSlice);
				m_XZCutCells[iter->first] = new CutCells3D<VectorType>(extStructures, iter->second, m_gridSpacing, xzGridDimensions, XZFace);
				m_XZCutCells[iter->first]->initialize();
			}

			
		}

		template <class VectorType>
		void CutVoxels3D<VectorType>::buildRegularGridEdges() {
			for (int i = 0; i < m_gridDimensions.x - 1; i++) {
				for (int j = 0; j < m_gridDimensions.y - 1; j++) {
					for (int k = 0; k < m_gridDimensions.z - 1; k++) {
						if (m_polyPatches(i, j, k).size() > 0) {
							/** For checking purposes, it should be enough to check if the size is 0. The Cut-cells 3-D
								will access these structures through slices and will initialize them on the right locations.
								Thus this method will only tap cells and edges that were not initialized before */

							/** Initializing horizontal edges */
							if (m_horizontalEdges(i, j, k)->size() == 0) {
								m_horizontalEdges(i, j, k)->push_back(new Edge<VectorType>(m_nodeVertices(i, j, k), m_nodeVertices(i + 1, j, k), xAlignedEdge));
							}

							if (m_horizontalEdges(i, j + 1, k)->size() == 0 && m_polyPatches(i, j + 1, k).size() == 0) {
								m_horizontalEdges(i, j + 1, k)->push_back(new Edge<VectorType>(m_nodeVertices(i, j + 1, k), m_nodeVertices(i + 1, j + 1, k), xAlignedEdge));
							}

							if (m_horizontalEdges(i, j + 1, k + 1)->size() == 0 && m_polyPatches(i, j + 1, k + 1).size() == 0) {
								m_horizontalEdges(i, j + 1, k + 1)->push_back(new Edge<VectorType>(m_nodeVertices(i, j + 1, k + 1), m_nodeVertices(i + 1, j + 1, k + 1), xAlignedEdge));
							}

							if (m_horizontalEdges(i, j, k + 1)->size() == 0 && m_polyPatches(i, j, k + 1).size() == 0) {
								m_horizontalEdges(i, j, k + 1)->push_back(new Edge<VectorType>(m_nodeVertices(i, j, k + 1), m_nodeVertices(i + 1, j, k + 1), xAlignedEdge));
							}

							/** Initializing vertical edges */
							if (m_verticalEdges(i, j, k)->size() == 0) {
								m_verticalEdges(i, j, k)->push_back(new Edge<VectorType>(m_nodeVertices(i, j, k), m_nodeVertices(i, j + 1, k), yAlignedEdge));
							}

							if (m_verticalEdges(i + 1, j, k)->size() == 0 && m_polyPatches(i + 1, j, k).size() == 0) {
								m_verticalEdges(i + 1, j, k)->push_back(new Edge<VectorType>(m_nodeVertices(i + 1, j, k), m_nodeVertices(i + 1, j + 1, k), yAlignedEdge));
							}

							if (m_verticalEdges(i + 1, j, k + 1)->size() == 0 && m_polyPatches(i + 1, j, k + 1).size() == 0) {
								m_verticalEdges(i + 1, j, k + 1)->push_back(new Edge<VectorType>(m_nodeVertices(i + 1, j, k + 1), m_nodeVertices(i + 1, j + 1, k + 1), yAlignedEdge));
							}

							if (m_verticalEdges(i, j, k + 1)->size() == 0 && m_polyPatches(i, j, k + 1).size() == 0) {
								m_verticalEdges(i, j, k + 1)->push_back(new Edge<VectorType>(m_nodeVertices(i, j, k + 1), m_nodeVertices(i, j + 1, k + 1), yAlignedEdge));
							}

							/** Initializing transversal edges */
							if (m_transversalEdges(i, j, k)->size() == 0) {
								m_transversalEdges(i, j, k)->push_back(new Edge<VectorType>(m_nodeVertices(i, j, k), m_nodeVertices(i, j, k + 1), zAlignedEdge));
							}

							if (m_transversalEdges(i + 1, j, k)->size() == 0 && m_polyPatches(i + 1, j, k).size() == 0) {
								m_transversalEdges(i + 1, j, k)->push_back(new Edge<VectorType>(m_nodeVertices(i + 1, j, k), m_nodeVertices(i + 1, j, k + 1), zAlignedEdge));
							}

							if (m_transversalEdges(i + 1, j + 1, k)->size() == 0 && m_polyPatches(i + 1, j + 1, k).size() == 0) {
								m_transversalEdges(i + 1, j + 1, k)->push_back(new Edge<VectorType>(m_nodeVertices(i + 1, j + 1, k), m_nodeVertices(i + 1, j + 1, k + 1), zAlignedEdge));
							}

							if (m_transversalEdges(i, j + 1, k)->size() == 0 && m_polyPatches(i, j + 1, k).size() == 0) {
								m_transversalEdges(i, j + 1, k)->push_back(new Edge<VectorType>(m_nodeVertices(i, j + 1, k), m_nodeVertices(i, j + 1, k + 1), zAlignedEdge));
							}

						}
					}
				}
			}
		}

		template<class VectorType>
		Face<VectorType> * CutVoxels3D<VectorType>::createRegularGridFace(uint i, uint j, uint k, faceLocation_t faceLocation) {
			Face<VectorType> *pFace = nullptr;
			vector<Edge<VectorType> *> regularEdges;
			dimensions_t gridDim;
			switch (faceLocation) {
				case XYFace:
					//Setting up grid dim accordingly to CutCells base: only two dimensions are used
					gridDim = dimensions_t(i, j);

					if (m_horizontalEdges(i, j, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_horizontalEdges(i, j, k)->front());

					if (m_verticalEdges(i + 1, j, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_verticalEdges(i + 1, j, k)->front());

					if (m_horizontalEdges(i, j + 1, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_horizontalEdges(i, j + 1, k)->front());

					if (m_verticalEdges(i, j, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_verticalEdges(i, j, k)->front());
				break;
				case YZFace:
					//Setting up grid dim accordingly to CutCells base: only two dimensions are used
					gridDim = dimensions_t(k, j);

					if (m_transversalEdges(i, j, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_transversalEdges(i, j, k)->front());

					if (m_verticalEdges(i, j, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_verticalEdges(i, j, k)->front());

					if (m_transversalEdges(i, j + 1, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_transversalEdges(i, j + 1, k)->front());

					if (m_verticalEdges(i, j, k + 1)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_verticalEdges(i, j, k + 1)->front());
				break;
				case XZFace:
					//Setting up grid dim accordingly to CutCells base: only two dimensions are used
					gridDim = dimensions_t(i, k);

					if (m_transversalEdges(i, j, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_transversalEdges(i, j, k)->front());

					if (m_horizontalEdges(i, j, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_horizontalEdges(i, j, k)->front());

					if (m_transversalEdges(i + 1, j, k)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_transversalEdges(i + 1, j, k)->front());

					if (m_horizontalEdges(i, j, k + 1)->size() != 1)
						throw (exception("CutVoxels3D buildFaces: Invalid edges on build-faces procedures"));
					regularEdges.push_back(m_horizontalEdges(i, j, k + 1)->front());
				break;
			}
			pFace = new Face<VectorType>(regularEdges, gridDim, m_gridSpacing, faceLocation);
			//Since this is a regular grid face, split will initialize a single half-face to it
			pFace->split();
			//Thus we should add a reversed copy to this face to get the two correct half faces attached to it
			if(pFace->getHalfFaces().size() > 0)
				pFace->addHalfFace(pFace->getHalfFaces().front()->reversedCopy());

			pFace->setRelativeFraction(1.0f);
			return pFace;
		}

		template<class VectorType>
		void CutVoxels3D<VectorType>::buildFaces() {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					for (int k = 0; k < m_gridDimensions.z; k++) {
						if (m_polyPatches(i, j, k).size() > 0) {
							/** Initialize XY faces */
							if (m_XYCutCells.find(k) != m_XYCutCells.end() && m_XYCutCells.find(k)->second->isCutCellAt(i, j)) { //Theres cut-cells faces on the kth plane
								m_XYFaces(i, j, k).insert(m_XYFaces(i, j, k).end(), m_XYCutCells[k]->getFaces(i, j).begin(), m_XYCutCells[k]->getFaces(i, j).end());
							} else { //Initialize a regular grid face on this location
								if (m_XYFaces(i, j, k).size() > 0) {
									throw (exception("CutVoxels3D buildFaces: faces array already initialized"));
								}
								m_XYFaces(i, j, k).push_back(createRegularGridFace(i, j, k, XYFace));
							}

							/** Fixing faces normals */
							for (int l = 0; l < m_XYFaces(i, j, k).size(); l++) {
								m_XYFaces(i, j, k)[l]->getHalfFaces().front()->setNormal(VectorType(0, 0, 1));
								m_XYFaces(i, j, k)[l]->getHalfFaces().back()->setNormal(VectorType(0, 0, -1));
							}

							if (m_polyPatches(i, j, k + 1).size() == 0) {
								if (m_XYFaces(i, j, k + 1).size() > 0) {
									throw (exception("CutVoxels3D buildFaces: faces array already initialized"));
								}
								m_XYFaces(i, j, k + 1).push_back(createRegularGridFace(i, j, k + 1, XYFace));

								for (int l = 0; l < m_XYFaces(i, j, k + 1).size(); l++) {
									m_XYFaces(i, j, k + 1)[l]->getHalfFaces().front()->setNormal(VectorType(0, 0, 1));
									m_XYFaces(i, j, k + 1)[l]->getHalfFaces().back()->setNormal(VectorType(0, 0, -1));
								}
							}

							/** Initialize YZ faces */
							if (m_YZCutCells.find(i) != m_YZCutCells.end() && m_YZCutCells.find(i)->second->isCutCellAt(k, j)) { //Theres cut-cells faces on the ith plane
								m_YZFaces(i, j, k).insert(m_YZFaces(i, j, k).end(), m_YZCutCells[i]->getFaces(k, j).begin(), m_YZCutCells[i]->getFaces(k, j).end());
							}
							else { //Initialize a regular grid face on this location
								if (m_YZFaces(i, j, k).size() > 0) {
									throw (exception("CutVoxels3D buildFaces: faces array already initialized"));
								}
								m_YZFaces(i, j, k).push_back(createRegularGridFace(i, j, k, YZFace));
							}

							/** Fixing faces normals */
							for (int l = 0; l < m_YZFaces(i, j, k).size(); l++) {
								m_YZFaces(i, j, k)[l]->getHalfFaces().front()->setNormal(VectorType(-1, 0, 0));
								m_YZFaces(i, j, k)[l]->getHalfFaces().back()->setNormal(VectorType(1, 0, 0));
							}

							if (m_polyPatches(i + 1, j, k).size() == 0) {
								if (m_YZFaces(i + 1, j, k).size() > 0) {
									throw (exception("CutVoxels3D buildFaces: faces array already initialized"));
								}
								m_YZFaces(i + 1, j, k).push_back(createRegularGridFace(i + 1, j, k, YZFace));

								for (int l = 0; l < m_YZFaces(i + 1, j, k).size(); l++) {
									m_YZFaces(i + 1, j, k)[l]->getHalfFaces().front()->setNormal(VectorType(-1, 0, 0));
									m_YZFaces(i + 1, j, k)[l]->getHalfFaces().back()->setNormal(VectorType(1, 0, 0));
								}
							}

							/** Initialize XZ faces */
							if (m_XZCutCells.find(j) != m_XZCutCells.end() && m_XZCutCells.find(j)->second->isCutCellAt(i, k)) { //Theres cut-cells faces on the jth plane
								m_XZFaces(i, j, k).insert(m_XZFaces(i, j, k).end(), m_XZCutCells[j]->getFaces(i, k).begin(), m_XZCutCells[j]->getFaces(i, k).end());
							}
							else { //Initialize a regular grid face on this location
								if (m_XZFaces(i, j, k).size() > 0) {
									throw (exception("CutVoxels3D buildFaces: faces array already initialized"));
								}
								m_XZFaces(i, j, k).push_back(createRegularGridFace(i, j, k, XZFace));
							}

							/** Fixing faces normals */
							for (int l = 0; l < m_XZFaces(i, j, k).size(); l++) {
								m_XZFaces(i, j, k)[l]->getHalfFaces().front()->setNormal(VectorType(0, -1, 0));
								m_XZFaces(i, j, k)[l]->getHalfFaces().back()->setNormal(VectorType(0, 1, 0));
							}

							if (m_polyPatches(i, j + 1, k).size() == 0) {
								if (m_XZFaces(i, j + 1, k).size() > 0) {
									throw (exception("CutVoxels3D buildFaces: faces array already initialized"));
								}
								m_XZFaces(i, j + 1, k).push_back(createRegularGridFace(i, j + 1, k, XZFace));

								for (int l = 0; l < m_XZFaces(i, j + 1, k).size(); l++) {
									m_XZFaces(i, j + 1, k)[l]->getHalfFaces().front()->setNormal(VectorType(0, -1, 0));
									m_XZFaces(i, j + 1, k)[l]->getHalfFaces().back()->setNormal(VectorType(0, 1, 0));
								}
							}
						}
					}
				}
			}

			//Remove thisss!!!!
			/** After building faces, flip half-faces of XZ and YZ faces. This will ensure that half-edges from half-faces
				are correctly oriented.*/
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					for (int k = 0; k < m_gridDimensions.z; k++) {
						if (m_XZFaces(i, j, k).size()) {
							for (int l = 0; l < m_XZFaces(i, j, k).size(); l++) {
								m_XZFaces(i, j, k)[l]->swapHalfFaces();
							}
						}

						if (m_YZFaces(i, j, k).size()) {
							for (int l = 0; l < m_YZFaces(i, j, k).size(); l++) {
								m_YZFaces(i, j, k)[l]->swapHalfFaces();
							}
						}
					}
				}
			}

			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					for (int k = 0; k < m_gridDimensions.z; k++) {
						for (int l = 0; l < m_XYFaces(i, j, k).size(); l++) {
							HalfFace<VectorType> *pHalfFace = m_XYFaces(i, j, k)[l]->getHalfFaces().front();
							addFacesToVertices(pHalfFace);
						}

						for (int l = 0; l < m_XZFaces(i, j, k).size(); l++) {
							HalfFace<VectorType> *pHalfFace = m_XZFaces(i, j, k)[l]->getHalfFaces().front();
							addFacesToVertices(pHalfFace);
						}
						
						for (int l = 0; l < m_YZFaces(i, j, k).size(); l++) {
							HalfFace<VectorType> *pHalfFace = m_YZFaces(i, j, k)[l]->getHalfFaces().front();
							addFacesToVertices(pHalfFace);
						}
					}
				}
			}
		}


		template<class VectorType>
		void CutVoxels3D<VectorType>::buildVolumes() {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					for (int k = 0; k < m_gridDimensions.z; k++) {
						if (m_polyPatches(i, j, k).size() > 0) {
							vector<Face<VectorType> *> volumeFaces;
							//Back, front, bottom, top, left right faces
							volumeFaces.insert(volumeFaces.end(), m_XYFaces(i, j, k).begin(), m_XYFaces(i, j, k).end());
							volumeFaces.insert(volumeFaces.end(), m_XYFaces(i, j, k + 1).begin(), m_XYFaces(i, j, k + 1).end());
							
							volumeFaces.insert(volumeFaces.end(), m_XZFaces(i, j, k).begin(), m_XZFaces(i, j, k).end());
							volumeFaces.insert(volumeFaces.end(), m_XZFaces(i, j + 1, k).begin(), m_XZFaces(i, j + 1, k).end());
							
							volumeFaces.insert(volumeFaces.end(), m_YZFaces(i, j, k).begin(), m_YZFaces(i, j, k).end());
							volumeFaces.insert(volumeFaces.end(), m_YZFaces(i + 1, j, k).begin(), m_YZFaces(i + 1, j, k).end());

							/** Adding geometric faces */
							for (int l = 0; l < m_polyPatches(i, j, k).size(); l++) {
								uint faceMeshIndex = m_polyPatches(i, j, k)[l].first;
								auto facesFromPatch = m_polyPatches(i, j, k)[l].second;

								//LineMesh vertices
								for (int m = 0; m < facesFromPatch.size(); m++) {
									volumeFaces.push_back(m_polyMeshes[faceMeshIndex]->getElements()[facesFromPatch[m]]);
								}
							}

							m_elements.push_back(new Volume<VectorType>(volumeFaces, dimensions_t(i, j, k), m_gridSpacing));
							m_volumesArray(i, j, k) = m_elements.back();
						}
					}
				}
			}
		}

		template<class VectorType>
		void CutVoxels3D<VectorType>::buildHalfVolumes() {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					for (int k = 0; k < m_gridDimensions.z; k++) {
						if (m_polyPatches(i, j, k).size() > 0) {
							auto currHalfVolumes = m_volumesArray(i, j, k)->split();
							m_halfVolumes.insert(m_halfVolumes.end(), currHalfVolumes.begin(), currHalfVolumes.end());
						}
					}
				}
			}
		}

		template<class VectorType>
		void CutVoxels3D<VectorType>::buildVertexAdjacencies() {
			//Set all vertices to be not updated 
			for (int i = 0; i < m_halfVolumes.size(); i++) {
				auto halfFaces = m_halfVolumes[i]->getHalfFaces();
				for (int j = 0; j < halfFaces.size(); j++) {
					auto halfEdges = halfFaces[j]->getHalfEdges();
					for (int k = 0; k < halfEdges.size(); k++) {
						halfEdges[k]->getVertices().first->setUpdated(false);
					}
				}
			}

			for (int i = 0; i < m_halfVolumes.size(); i++) {
				auto halfFaces = m_halfVolumes[i]->getHalfFaces();
				for (int j = 0; j < halfFaces.size(); j++) {
					if (halfFaces[j]->getLocation() != geometryHalfFace) {
						auto halfEdges = halfFaces[j]->getHalfEdges();
						for (int k = 0; k < halfEdges.size(); k++) {
							int prevK = roundClamp<int>(k - 1, 0, halfEdges.size());
							if (halfEdges[k]->getLocation() == geometryHalfEdge) {
								uint vertexID = halfEdges[k]->getVertices().first->getID();
								if (halfFaces[j]->getLocation() == topHalfFace ||
									halfFaces[j]->getLocation() == rightHalfFace ||
									halfFaces[j]->getLocation() == frontHalfFace) { 

									//Use this half-edge
									halfEdges[k]->getVertices().first->addConnectedHalfFace(halfFaces[j]);
								}
								//else { //Use other half-edge
								//	if (halfFaces[j]->getLocation() == bottomHalfFace) {
								//		HalfEdge<VectorType> *pOtherHalfEdge = nullptr;
								//		if (halfEdges[k]->getEdge()->getHalfEdges().first->getID() == halfEdges[k]->getID()) {
								//			pOtherHalfEdge = halfEdges[k]->getEdge()->getHalfEdges().second;
								//		}
								//		else {
								//			pOtherHalfEdge = halfEdges[k]->getEdge()->getHalfEdges().first;
								//		}
								//		//pOtherHalfEdge->getVertices().second->addConnectedHalfFace(halfFaces[j]);

								//		//Use this half-edge
								//		halfEdges[k]->getVertices().first->addConnectedHalfFace(halfFaces[j]);
								//	}
								//	
								//}
								//

							}
							else if (halfEdges[prevK]->getLocation() == geometryHalfEdge) {
								if (halfFaces[j]->getLocation() == topHalfFace ||
									halfFaces[j]->getLocation() == rightHalfFace ||
									halfFaces[j]->getLocation() == frontHalfFace) {
									//Use this half-edge
									halfEdges[k]->getVertices().first->addConnectedHalfFace(halfFaces[j]);
								}
								else { //Use other half-edge
									/*uint vertexID = halfEdges[k]->getVertices().first->getID();
									HalfEdge<VectorType> *pOtherHalfEdge = nullptr;
									if (halfEdges[prevK]->getEdge()->getHalfEdges().first->getID() == halfEdges[prevK]->getID()) {
										pOtherHalfEdge = halfEdges[prevK]->getEdge()->getHalfEdges().second;
									}
									else {
										pOtherHalfEdge = halfEdges[prevK]->getEdge()->getHalfEdges().first;
									}
									pOtherHalfEdge->getVertices().first->addConnectedHalfFace(halfFaces[j]);*/
								}
							}
						}
					}
				}
			}


			//for (int i = 0; i < m_halfVolumes.size(); i++) {
			//	auto halfFaces = m_halfVolumes[i]->getHalfFaces();
			//	for (int j = 0; j < halfFaces.size(); j++) {
			//		if (halfFaces[j]->getLocation() != geometryHalfFace) {
			//			auto halfEdges = halfFaces[j]->getHalfEdges();
			//			for (int k = 0; k < halfEdges.size(); k++) {
			//				//if (!halfEdges[k]->getVertices().first->hasUpdated()) {
			//					halfEdges[k]->getVertices().first->addConnectedHalfFace(halfFaces[j]);
			//					//halfEdges[k]->getVertices().first->setUpdated(true);
			//				//}
			//			}
			//		}	
			//	}
			//}
		}

		template<class VectorType>
		void CutVoxels3D<VectorType>::buildTriangleHalfVolumes() {
			for (uint i = 0; i < m_halfVolumes.size(); i++) {
				m_triangleHalfVolumes.push_back(new TriangleHalfVolume<VectorType>(m_halfVolumes[i]));
			}
		}
		#pragma endregion

		#pragma region AuxiliaryHelperFunctions 
		
		#pragma region ComparisonCallbacks
		
		#pragma endregion

		template class CutVoxels3D<Vector3>;
		template class CutVoxels3D<Vector3D>;
	}
}