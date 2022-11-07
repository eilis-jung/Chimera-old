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
#ifndef _DATA_CGL_OBJ_IMPORTER_
#define _DATA_CGL_OBJ_IMPORTER_

#pragma  once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "CGALConfig.h"

namespace Chimera {
	using namespace Grids;
	namespace CGALWrapper {
		namespace IO {

			// The BuildCgalPolyhedronFromObj class builds a CGAL::Polyhedron_3 from Wavefront OBJ file.
			// This is very simple reader and only reads vertex coordinates and vertex index for faces.
			// Faces can be polygons and doesn't have to be triangles.
			template<class HDS, class VectorType>
			class BuildCgalPolyhedronFromObj : public CGAL::Modifier_base<HDS>
			{
			public:

				BuildCgalPolyhedronFromObj(const VectorType &position, const std::string& fileName, Scalar dx) : m_position(position),
					mFileName(fileName), m_dx(dx) {}

				void operator() (HDS& hds)
				{
					typedef typename HDS::Vertex   Vertex;
					typedef typename Vertex::Point Point;

					// Open obj file for reading.
					std::ifstream _file(mFileName.c_str());
					if (_file.fail())
					{
						return;
					}

					// Count the number of vertices and facets.
					// This is used to reserve memory in HDS.
					std::string _line;
					int _numVertices = 0;
					int _numFacets = 0;
					while (_file.good())
					{
						std::getline(_file, _line);
						if (_line.size() > 1)
						{
							if (_line[0] == 'v' && _line[1] == ' ') { ++_numVertices; }
							if (_line[0] == 'f' && _line[1] == ' ') { ++_numFacets; }
						}
					}

					// Rewind file to beginning for reading data.
					if (!_file.good())
					{
						_file.clear();
					}
					_file.seekg(0);

					// Postcondition: hds is a valid polyhedral surface.
					CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);

					// Load the data from OBJ file to HDS.
					B.begin_surface(_numVertices, _numFacets, int((_numVertices + _numFacets - 2)*2.1));
					int vertexID = 0;

					std::string _token;
					int currNormal = 0;
					m_normals.clear();
					while (!_file.eof())
					{
						_token = ""; // Reset token.
						_file >> _token;

						// if token is v then its a vertex.
						if (_token == "v")
						{
							double x, y, z;
							_file >> x >> y >> z;
							VectorType xyz(x, y, z);
							if (m_dx != 0 && (isOnGridPoint(xyz, m_dx) /*|| isOnGridFace(xyz, m_dx)*/)) {
								xyz += VectorType(1, 1, 1)*1e-4;
								cout << "Perturbing point " << x << " " << y << " " << z << endl;
							}
							HDS::Vertex_handle vh = B.add_vertex(Point(xyz.x + m_position.x, xyz.y + m_position.y, xyz.z + m_position.z));
							vh->id = vertexID++;
						}


						else if (_token == "vn") {
							double x, y, z;
							_file >> x >> y >> z;
							VectorType faceNormal(x, y, z);
							m_normals.push_back(faceNormal);
						}

						// There are 4 type of facets.
						// a     only vertex index.
						// a/b   vertex and texture index.
						// a/b/c vertex, texture and normal index.
						// a//c  vertex and normal index.
						else if (_token == "f")
						{
							// Read the remaining line for the facet.
							std::string _line;
							std::getline(_file, _line);

							// Split the line into facet's vertices.
							// The length of _vertices is equal to the number of vertices for this face.
							std::istringstream _stream(_line);
							std::vector<std::string> _vertices;
							std::copy(std::istream_iterator<std::string>(_stream),
								std::istream_iterator<std::string>(),
								std::back_inserter(_vertices));

							// For each vertex read only the first number, which is the vertex index.
							HDS::Face_handle currFace = B.begin_facet();
							VectorType currNormalVec;
							for (size_t i = 0; i < _vertices.size(); ++i)
							{
								std::string::size_type _pos = _vertices[i].find('/', 0);
								std::string _indexStr = _vertices[i].substr(0, _pos);
								B.add_vertex_to_facet(stoi(_indexStr) - 1); // -1 is because OBJ file uses 1 based index.
								currNormalVec += m_normals[stoi(_indexStr) - 1];
							}
							currNormalVec.normalize();
							//currNormalVec /= _vertices.size();
							//currFace->normal = m_normals[currNormal++];
							currFace->normal = convertToVector3D(currNormalVec);
							B.end_facet();
						}
					}
					_file.close();

					B.end_surface();
				}

			private:

				std::string mFileName;
				const VectorType m_position;
				vector<VectorType> m_normals;
				Scalar m_dx;
			};


			// Import a OBJ file given by fileName to polyhedron.
			// TPoly is a type of CGAL::Polyhdeon_3.
			template<class TPoly, class VectorType>
			void importOBJ(const VectorType &position, const std::string& fileName, TPoly* polyhedron, Scalar dx = 0.0)
			{
				if (polyhedron)
				{
					try
					{
						// Build Polyhedron_3 from the OBJ file.
						BuildCgalPolyhedronFromObj<TPoly::HalfedgeDS, VectorType> _buildPolyhedron(position, fileName, dx);

						// Calls is_valid at the end. Throws an exception in debug mode if polyhedron is not
						// manifold.
						polyhedron->delegate(_buildPolyhedron);

						// CGAL::Assert_exception is thrown in the debug mode when 
						// CGAL::Polyhedron_incremental_builder_3 is destroyed in BuildCgalPolyhedronFromObj.
						// However, in the release mode assertions is disabled and hence no exception is thrown.
						// Thus for uniform error reporting, if the polyhedron is not valid then throw a dummy 
						// exception in release mode.
						if (!polyhedron->is_valid())
						{
							throw CGAL::Assertion_exception("", "", "", 0, "");
						}
					}
					catch (const CGAL::Assertion_exception&)
					{
						std::string _msg = "SMeshLib::importOBJ: Error loading " + fileName;
						throw std::exception(_msg.c_str());
					}
				}
			}


		};
	};


};

#endif