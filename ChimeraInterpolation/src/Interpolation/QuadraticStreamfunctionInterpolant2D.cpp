#include "Interpolation/QuadraticStreamfunctionInterpolant2D.h"

namespace Chimera {
	namespace Interpolation {

#pragma region Constructors
		template<class valueType>
		QuadraticStreamfunctionInterpolant2D<valueType>::QuadraticStreamfunctionInterpolant2D(const Array2D<valueType>& values, Scalar gridDx)
			: BilinearStreamfunctionInterpolant2D(values, gridDx) {
			m_pStreamDiscontinuousValues = new Array2D <Scalar[4]>(values.getDimensions());
			m_dx = gridDx;
		}
#pragma endregion

#pragma region Functionalities
		template<>
		Scalar QuadraticStreamfunctionInterpolant2D<Vector2>::interpolateScalarStreamfunction(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));

			Vector2 x1Position(i + 0.0f, j + 0.0f);
			Vector2 x2Position(i + 1.0f, j + 1.0f);

			Scalar a[4], b[4], c[4];
			calculateCoefficientsScalar(dimensions_t(i, j), a, b, c);

			Scalar alphaX = (gridSpacePosition.x - x1Position.x);
			Scalar alphaY = (gridSpacePosition.y - x1Position.y);
			Scalar alphaXc = (1 - (gridSpacePosition.x - x1Position.x));
			Scalar alphaYc = (1 - (gridSpacePosition.y - x1Position.y));

			Scalar interpBottom =	interpolateScalar(position, CutCells::bottomFace, a, b, c);
			Scalar interpTop =		interpolateScalar(position, CutCells::topFace, a, b, c);

			
			calculateCoefficientsScalar(dimensions_t(i, j - 1), a, b, c);
			Scalar interpBottom_s =	interpolateScalar(position, CutCells::bottomFace, a, b, c);
			Scalar interpTop_s =	interpolateScalar(position, CutCells::topFace, a, b, c);

			Scalar aF, bF, cF;
			cF = interpBottom;
			bF = (interpTop_s - interpBottom_s);
			aF = interpTop - bF - cF;

			Scalar cAlpha = 0.5;
			Scalar compare1 = cAlpha*interpTop + (1 - cAlpha)*interpBottom;
			Scalar compare2 = aF*cAlpha*cAlpha + bF*cAlpha + cF;

			return aF*alphaY*alphaY + bF*alphaY + cF;
		}

		template<>
		Vector2 QuadraticStreamfunctionInterpolant2D<Vector2>::interpolateAccurate(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			Vector2 interpolatedValue;
			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));

			Vector2 x1Position(i + 0.0f, j + 0.0f);
			Vector2 x2Position(i + 1.0f, j + 1.0f);

			Scalar a[4], b[4], c[4];
			calculateCoefficients(dimensions_t(i, j), a, b, c);

			Scalar alphaX = (gridSpacePosition.x - x1Position.x);
			Scalar alphaY = (gridSpacePosition.y - x1Position.y);
			Scalar alphaXc = 1 - (gridSpacePosition.x - x1Position.x);
			Scalar alphaYc = 1 - (gridSpacePosition.y - x1Position.y);

			Vector2 velocity = m_values(i, j);
			Vector2 velocityX1 = m_values(i - 1, j);
			Vector2 velocityY1 = m_values(i, j - 1);
			Vector2 velocityX2 = m_values(i + 1, j);
			Vector2 velocityY2 = m_values(i, j + 1);

			Scalar interpRight = interpolateGradientApprox(position, CutCells::rightFace, a, b, c);
			Scalar interpLeft = -interpolateGradientApprox(position, CutCells::leftFace, a, b, c);
			Scalar interpBottom = -interpolateGradientApprox(position, CutCells::bottomFace, a, b, c);
			Scalar interpTop = interpolateGradientApprox(position, CutCells::topFace, a, b, c);

			Scalar derivativeLeft = 0;
			Scalar fa, fb, fc;
			{
				Scalar a_l[4], b_l[4], c_l[4];
				calculateCoefficients(dimensions_t(i - 1, j), a_l, b_l, c_l);
				Scalar interpRightFL = interpolateGradientApprox(position, CutCells::rightFace, a_l, b_l, c_l);
				Scalar interpLeftFL = -interpolateGradientApprox(position, CutCells::leftFace, a_l, b_l, c_l);
				derivativeLeft = (interpLeft - interpLeftFL);

				fc = interpLeft;
				fb = derivativeLeft;
				fa = interpRight - fb - fc;

				interpolatedValue.x = fa*alphaX*alphaX + fb*alphaX + fc;
				//interpolatedValue.x = (1 - alphaX)*interpLeft + alphaX*interpRight;
			}

			Scalar derivativeBottom = 0;
			{
				Scalar a_b[4], b_b[4], c_b[4];
				calculateCoefficients(dimensions_t(i, j - 1), a_b, b_b, c_b);
				Scalar interpBottomFB = -interpolateGradientApprox(position, CutCells::bottomFace, a_b, b_b, c_b);
				Scalar interpTopFB = interpolateGradientApprox(position, CutCells::topFace, a_b, b_b, c_b);
				derivativeBottom = (interpBottom - interpBottomFB);

				fc = interpBottom;
				fb = derivativeBottom;
				fa = interpTop - fb - fc;

				interpolatedValue.y = fa*alphaY*alphaY + fb*alphaY + fc;
				//interpolatedValue.y = (1 - alphaY)*interpBottom + alphaY*interpTop;
			}

			return interpolatedValue;
		}

		template<>
		Vector2 QuadraticStreamfunctionInterpolant2D<Vector2>::interpolateApprox(const Vector2 &position) {
			Vector2 gridPosition = position / m_dx;

			Scalar dxS = 0.0001;
			Scalar interp = interpolateScalarStreamfunction(position);
			Scalar interpX, interpY;
			Scalar signX = 1.0f, signY = 1.0f;

			if ((gridPosition.x + dxS / m_dx) - floor(gridPosition.x) >= 1.0f) {
				interpX = interpolateScalarStreamfunction(position - Vector2(dxS, 0));
				signX = -1.0f;
			}
			else {
				interpX = interpolateScalarStreamfunction(position + Vector2(dxS, 0));
			}
			if ((gridPosition.y + dxS / m_dx) - floor(gridPosition.y) >= 1.0f) {
				interpY = interpolateScalarStreamfunction(position - Vector2(0, dxS));
				signY = -1.0f;
			}
			else {
				interpY = interpolateScalarStreamfunction(position + Vector2(0, dxS));
			}


			return Vector2(signY*(interpY - interp) / dxS, -signX*(interpX - interp) / dxS);
		}
		/** Templated interpolation is the only one needed */
		template<>
		Vector2 QuadraticStreamfunctionInterpolant2D<Vector2>::interpolate2(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));


			Vector2 x1Position(i + 0.0f, j + 0.0f);
			Vector2 x2Position(i + 1.0f, j + 1.0f);

			Scalar a[2], b[2];

			Scalar fluxBottom = -m_values(i, j).y;
			Scalar fluxRight = m_values(i + 1, j).x;
			Scalar fluxTop = m_values(i, j + 1).y;
			Scalar fluxLeft = -m_values(i, j).x;

			//First set of linear equations
			a[0] = 0.5*(fluxRight - fluxBottom);
			b[0] = fluxRight - 3 * a[0];

			//Second set of linear equations
			a[1] = 0.5*(fluxLeft - fluxTop);
			b[1] = fluxLeft - 3 * a[1];

			Scalar alphaX = (gridSpacePosition.x - x1Position.x);
			Scalar alphaY = (gridSpacePosition.y - x1Position.y);


			Scalar xParam = alphaY + 1;
			Scalar yParam = alphaX;

			Scalar rFluxBottom = 2 * 0.5*a[0] + b[0];
			Scalar rFluxRight = 2 * 1.5*a[0] + b[0];
			Scalar rFluxTop = 2 * 0.5*a[1] + b[1];
			Scalar rFluxLeft = 2 * 1.5*a[1] + b[1];

			Scalar fluxesDiff = 0;
			fluxesDiff += abs(rFluxBottom - fluxBottom);
			fluxesDiff += abs(rFluxTop - fluxTop);
			fluxesDiff += abs(rFluxRight - fluxRight);
			fluxesDiff += abs(rFluxLeft - fluxLeft);

			if (fluxesDiff > 0.0001) {
				cout << "Error on fluxes reconstruction " << endl;
			}


			Scalar tVelBottom = -(2 * yParam*a[0] + b[0]);
			Scalar tVelRight = 2 * xParam*a[0] + b[0];
			Scalar tVelTop = 2 * yParam*a[1] + b[1];
			Scalar tVelLeft = -(2 * xParam*a[1] + b[1]);

			Scalar a_x = tVelTop - tVelBottom;
			Scalar c_x = tVelRight;
			Scalar b_x = tVelLeft - c_x - a_x;

			Vector2 interpolatedValue;
			interpolatedValue.x = a_x*alphaX*alphaX + b_x*alphaX + c_x;
			//interpolatedValue.x = alphaX*(2*xParam*a[0] + b[0]) + (1 - alphaX)*(-(2*xParam*a[1] + b[1]));


			Scalar a_y = tVelRight - tVelLeft;
			Scalar c_y = tVelBottom;
			Scalar b_y = tVelTop - c_y - a_y;

			interpolatedValue.y = a_y*alphaY*alphaY + b_y*alphaY + c_y;
			//interpolatedValue.y = alphaY*(2*yParam*a[1] + b[1]) + (1 - alphaY)*(-(2*yParam*a[0] + b[0]));

			return interpolatedValue;
		}

		template<>
		Vector2 QuadraticStreamfunctionInterpolant2D<Vector2>::interpolate(const Vector2 &position) {

			Vector2 gridSpacePosition = position / m_dx;
			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));

			if (i <= 2 || j <= 2 || i >= m_gridDimensions.x - 2 || j >= m_gridDimensions.y - 2) {
				return BilinearStreamfunctionInterpolant2D<Vector2>::interpolate(position);
			}

			return interpolateApprox(position);


			Vector2 x1Position(i + 0.0f, j + 0.0f);
			Vector2 x2Position(i + 1.0f, j + 1.0f);

			Scalar auxVelX0 = 0, auxVelX1 = 0, auxVelY0 = 0, auxVelY1 = 0;
			Scalar aX0 = 0, aX1 = 0, aY0 = 0, aY1 = 0;

			//X values initialization
			{
				if (j > 0) {
					auxVelX0 = 0.5*(m_values(i, j - 1).x + m_values(i, j).x);
					aX0 = ((m_values(i, j).x) + auxVelX0);
				}
				else {

				}
				if (i < m_gridDimensions.x - 1) {
					auxVelX1 = 0.5*(m_values(i + 1, j - 1).x + m_values(i + 1, j).x);
					aX1 = (m_values(i + 1, j).x - auxVelX1);
				}
				else {

				}
			}

			//Y values initialization
			{
				if (i > 0) {
					auxVelY0 = 0.5*(m_values(i - 1, j).y + m_values(i, j).y);
					aY0 = ((m_values(i, j).y) + auxVelY0);
				}
				else {

				}
				if (j < m_gridDimensions.y - 1) {
					auxVelY1 = 0.5*(m_values(i - 1, j + 1).y + m_values(i, j + 1).y);
					aY1 = (m_values(i, j + 1).y - auxVelY1);
				}
				else {

				}
			}



			Scalar alphaX = (gridSpacePosition.x - x1Position.x);
			Scalar alphaY = (gridSpacePosition.y - x1Position.y);

			Vector2 interpolatedValue;
			interpolatedValue.x = alphaX*(alphaY*aX0 + auxVelX0) + (1 - alphaX)*(alphaY*aX1 + auxVelX1);
			interpolatedValue.y = alphaY*(alphaX*aY0 + auxVelY0) + (1 - alphaY)*(alphaX*aY1 + auxVelY1);


			return interpolatedValue;
		}

#pragma endregion

#pragma region PrivateFunctionalities
		template<>
		void QuadraticStreamfunctionInterpolant2D<Vector2>::calculateCoefficients(dimensions_t cellIndex, Scalar *a, Scalar *b, Scalar *c) {
			int i = cellIndex.x; int j = cellIndex.y;

			Scalar fluxBottom = m_values(i, j).y;
			Scalar fluxRight = m_values(i + 1, j).x;
			Scalar fluxTop = m_values(i, j + 1).y;
			Scalar fluxLeft = m_values(i, j).x;
			Scalar fluxBottomWest = m_values(i - 1, j).y;
			Scalar fluxRightSouth = m_values(i + 1, j - 1).x;
			Scalar fluxTopEast = m_values(i + 1, j + 1).y;
			Scalar fluxLeftNorth = m_values(i, j + 1).x;

			//Scalar fluxBottom = -m_values(i, j).y;
			//Scalar fluxRight = m_values(i + 1, j).x;
			//Scalar fluxTop = m_values(i, j + 1).y;
			//Scalar fluxLeft = -m_values(i, j).x;
			//Scalar fluxBottomWest = -m_values(i - 1, j).y;
			//Scalar fluxRightSouth = m_values(i + 1, j - 1).x;
			//Scalar fluxTopEast = m_values(i + 1, j + 1).y;
			//Scalar fluxLeftNorth = -m_values(i, j + 1).x;

			Scalar div2 = (m_values(i + 1, j).x - m_values(i, j).x) / m_dx + (m_values(i, j + 1).y - m_values(i, j).y) / m_dx;

			if (abs(div2) > 2.6703e-5) {
				cout << "Div alet" << endl;
			}


			c[0] = 0;
			c[1] = fluxBottom;
			c[2] = c[1] + fluxRight;
			c[3] = c[2] + fluxTop;

			b[0] = 0.5*(fluxBottomWest + fluxBottom);
			b[1] = 0.5*(fluxRightSouth + fluxRight);
			b[2] = 0.5*(fluxTopEast + fluxTop);
			b[3] = 0.5*(fluxLeftNorth + fluxLeft);

			a[0] = fluxBottom - b[0];
			a[1] = fluxRight - b[1];
			a[2] = fluxTop - b[2];
			a[3] = fluxLeft - b[3];

			Scalar fluxes[4];
			fluxes[0] = fluxBottom;
			fluxes[1] = fluxRight;
			fluxes[2] = fluxTop;
			fluxes[3] = fluxLeft;

			Scalar boundaries[4][2];
			for (int i = 0; i < 4; i++) {
				boundaries[i][0] = c[i];
				boundaries[i][1] = a[i] + b[i] + c[i];
			}

			/*for (int i = 0; i < 4; i++) {
				int nextI = roundClamp(i + 1, 0, 4);
				if (abs(boundaries[i][1] - boundaries[nextI][0]) > 1e-5) {
					cout << "Wrong continuity" << endl;
				}
			}
			for (int i = 0; i < 4; i++) {
				if (abs(a[i] + b[i] - fluxes[i]) > 1e-5) {
					cout << "Wrong compatibility" << endl;
				}
			}
			*/
		}

		template<>
		void QuadraticStreamfunctionInterpolant2D<Vector2>::calculateCoefficientsScalar(dimensions_t cellIndex, Scalar *a, Scalar *b, Scalar *c) {
			int i = cellIndex.x; int j = cellIndex.y;
			Scalar fluxBottom = -m_values(i, j).y*m_dx;
			Scalar fluxRight = m_values(i + 1, j).x*m_dx;
			Scalar fluxTop = m_values(i, j + 1).y*m_dx;
			Scalar fluxLeft = -m_values(i, j).x*m_dx;
			Scalar fluxBottomWest = -m_values(i - 1, j).y*m_dx;
			Scalar fluxRightSouth = m_values(i + 1, j - 1).x*m_dx;
			Scalar fluxTopEast = m_values(i + 1, j + 1).y*m_dx;
			Scalar fluxLeftNorth = -m_values(i, j + 1).x*m_dx;

			Scalar fluxTopWest = m_values(i - 1, j + 1).y*m_dx;
			Scalar flutLeftSouth = -m_values(i, j - 1).x*m_dx;

			c[0] = 0;
			c[1] = fluxBottom;
			c[2] = c[1] + fluxRight;
			c[3] = c[2] + fluxTop;

			Scalar sumFluxes = fluxBottom + fluxRight + fluxTop + fluxLeft;
			Scalar div2 = (m_values(i + 1, j).x - m_values(i, j).x)/m_dx + (m_values(i, j + 1).y - m_values(i, j).y)/m_dx;

			/*if (abs(div2) > 2.6703e-5) {
				cout << "Div alet" << endl;
			}*/

			b[0] = 0.5*(fluxBottomWest + fluxBottom);
			b[1] = 0.5*(fluxRightSouth + fluxRight);
			Scalar derivativeTopWest = 0.5*(fluxTopWest + fluxTop);
			b[2] = 2 * fluxTop - derivativeTopWest;
			//b[2] = 0.5*(fluxTopEast + fluxTop);
			Scalar derivativeLeftSouth = 0.5*(flutLeftSouth + fluxLeft);
			b[3] = 2 * fluxLeft - derivativeLeftSouth;
			//b[3] = 0.5*(fluxLeftNorth + fluxLeft);

			a[0] = fluxBottom - b[0];
			a[1] = fluxRight - b[1];
			a[2] = fluxTop - b[2];
			a[3] = fluxLeft - b[3];

			Scalar fluxes[4];
			fluxes[0] = fluxBottom;
			fluxes[1] = fluxRight;
			fluxes[2] = fluxTop;
			fluxes[3] = fluxLeft;

			Scalar boundaries[4][2];
			for (int i = 0; i < 4; i++) {
				boundaries[i][0] = c[i];
				boundaries[i][1] = a[i] + b[i] + c[i];
			}
			for (int i = 0; i < 4; i++) {
				int nextI = roundClamp(i + 1, 0, 4);
				if (abs(boundaries[i][1] - boundaries[nextI][0]) > 1e-5) {
					cout << "Wrong continuity" << endl;
				}
			}
			for (int i = 0; i < 4; i++) {
				if (abs(a[i] + b[i] - fluxes[i]) > 1e-5) {
					cout << "Wrong compatibility" << endl;
				}
			}
		}

		template<>
		Scalar QuadraticStreamfunctionInterpolant2D<Vector2>::interpolateGradientAccurate(const Vector2 &position, CutCells::faceLocation_t faceLocation, Scalar * a, Scalar * b, Scalar * c) {
			Vector2 gridSpacePosition = position / m_dx;
			Vector2 x1Position(floor(gridSpacePosition.x), floor(gridSpacePosition.y));
			Scalar alphaX = (gridSpacePosition.x - x1Position.x);
			Scalar alphaY = (gridSpacePosition.y - x1Position.y);
			Scalar alphaXc = 1 - (gridSpacePosition.x - x1Position.x);
			Scalar alphaYc = 1 - (gridSpacePosition.y - x1Position.y);

			switch (faceLocation) {
				case CutCells::bottomFace:
					return 2 * a[0] * alphaX + b[0];
				break;
				case CutCells::rightFace:
					return 2 * a[1] * alphaY + b[1];
				break;
				case CutCells::topFace:
					return 2 * a[2] * alphaXc + b[2];
				break;
				case CutCells::leftFace:
					return 2 * a[3] * alphaYc + b[3];
				break;
			}
			
			return 0.0f;
		}

		template<>
		Scalar QuadraticStreamfunctionInterpolant2D<Vector2>::interpolateGradientApprox(const Vector2 &position, CutCells::faceLocation_t faceLocation, Scalar * a, Scalar * b, Scalar * c) {
			Vector2 gridSpacePosition = position / m_dx;
			Vector2 x1Position(floor(gridSpacePosition.x), floor(gridSpacePosition.y));
			Scalar alphaX = (gridSpacePosition.x - x1Position.x);
			Scalar alphaY = (gridSpacePosition.y - x1Position.y);
			Scalar alphaXc = 1 - (gridSpacePosition.x - x1Position.x);
			Scalar alphaYc = 1 - (gridSpacePosition.y - x1Position.y);

			Scalar dxS = 0.0001;
			Scalar interp = interpolateScalar(position, faceLocation, a, b, c)*m_dx;
			Scalar interpX, interpY;
			Scalar signX = 1.0f, signY = 1.0f;

			switch (faceLocation) {
				case CutCells::bottomFace:
				case CutCells::topFace:
					if ((gridSpacePosition.x + dxS / m_dx) - floor(gridSpacePosition.x) >= 1.0f) {
						interpX = interpolateScalar(position - Vector2(dxS, 0), faceLocation, a, b, c)*m_dx;
						signX = -1.0f;
					}
					else {
						interpX = interpolateScalar(position + Vector2(dxS, 0), faceLocation, a, b, c)*m_dx;
					}
					return -signX*(interpX - interp) / dxS;
				break;
				case CutCells::leftFace:
				case CutCells::rightFace:
					if ((gridSpacePosition.y + dxS / m_dx) - floor(gridSpacePosition.y) >= 1.0f) {
						interpY = interpolateScalar(position - Vector2(0, dxS), faceLocation, a, b, c)*m_dx;
						signY = -1.0f;
					}
					else {
						interpY = interpolateScalar(position + Vector2(0, dxS), faceLocation, a, b, c)*m_dx;
					}
					return signY*(interpY - interp) / dxS;
				break;
			}
			
			return 0.0f;
		}

		template<>
		Scalar QuadraticStreamfunctionInterpolant2D<Vector2>::interpolateScalar(const Vector2 &position, CutCells::faceLocation_t faceLocation, Scalar * a, Scalar * b, Scalar * c) {
			Vector2 gridSpacePosition = position / m_dx;
			Vector2 x1Position(floor(gridSpacePosition.x), floor(gridSpacePosition.y));
			Scalar alphaX = (gridSpacePosition.x - x1Position.x);
			Scalar alphaY = (gridSpacePosition.y - x1Position.y);
			Scalar alphaXc = 1 - (gridSpacePosition.x - x1Position.x);
			Scalar alphaYc = 1 - (gridSpacePosition.y - x1Position.y);

			switch (faceLocation) {
				case CutCells::bottomFace:
					return a[0] * alphaX * alphaX + b[0] * alphaX + c[0]; 
				break;
				case CutCells::rightFace:
					return a[1] * alphaY * alphaY + b[1] * alphaY + c[1];
				break;
				case CutCells::topFace:
					return a[2] * alphaXc * alphaXc + b[2] * alphaXc + c[2];
				break;
				case CutCells::leftFace:
					return a[3] * alphaYc * alphaYc + b[3] * alphaYc + c[3];
				break;
			}
			return 0.0f;
		}

	#pragma endregion
		/** Template linker trickerino for templated classes in CPP*/
		template class QuadraticStreamfunctionInterpolant2D<Vector2>;
	}
}