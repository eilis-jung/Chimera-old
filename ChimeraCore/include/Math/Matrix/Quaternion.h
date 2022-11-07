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


// Quaternion.h: interface for the Quaternion class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_QUAT4F_H__F1B3C959_7040_4622_8798_4243FDA11191__INCLUDED_)
#define AFX_QUAT4F_H__F1B3C959_7040_4622_8798_4243FDA11191__INCLUDED_

#include <cmath>
#include <cassert>

//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

#include "Math/Vector3.h"

//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

namespace Chimera {
	namespace Core {

		//////////////////////////////////////////////////////////////////////
		//
		//////////////////////////////////////////////////////////////////////

		class Quaternion {
		public:
			Scalar w;
			Scalar x, y, z;

			Quaternion() : w(1), x(0), y(0), z(0) {}
			Quaternion( const Scalar w, const Scalar x, const Scalar y, const Scalar z ) : w(w), x(x), y(y), z(z) {}
			Quaternion( const Scalar w, const Vector3 v) {
				this->w = w;
				x = v.x;
				y = v.y;
				z = v.z;
			}
			/// construct quaternion from angle-axis
			Quaternion(const Vector3 &axis, Scalar angle)
			{
				fromAA(axis, angle);
			}
			virtual ~Quaternion();

			//assign
			const Quaternion& operator = ( const Quaternion& q )
			{
				w = q.w;
				x = q.x;
				y = q.y;
				z = q.z;

				return *this;
			}

			//compare
			int operator == ( const Quaternion& q ) const
			{
				return (q.w==w && q.x==x && q.y==y && q.z==z);
			}

			int operator != ( const Quaternion& q ) const
			{
				return (q.w!=w || q.x!=x || q.y!=y || q.z!=z);
			}

			//negate
			Quaternion operator - () const
			{
				return Quaternion( -w, -x, -y, -z );
			}

			void operator += ( const Quaternion& q )
			{
				w += q.w;
				x += q.x;
				y += q.y;
				z += q.z;
			}

			void operator -= ( const Quaternion& q )
			{
				w -= q.w;
				x -= q.x;
				y -= q.y;
				z -= q.z;
			}

			void operator *= ( const Scalar s )
			{
				w *= s;
				x *= s;
				y *= s;
				z *= s;
			}

			void operator /= ( Scalar s )
			{
				s = 1/s;
				w *= s;
				x *= s;
				y *= s;
				z *= s;
			}

			//add
			const Quaternion operator + ( const Quaternion& q ) const
			{
				return Quaternion( w + q.w, x + q.x, y + q.y, z + q.z );
			}

			//subtract
			const Quaternion operator - ( const Quaternion& q ) const
			{
				return Quaternion( w - q.w, x - q.x, y - q.y, z - q.z );
			}

			//multiply
			const Quaternion operator * ( const Scalar s ) const
			{
				return Quaternion( w * s, x * s, y * s, z * s );
			}

			//multiply
			const Quaternion operator * ( const Quaternion& q) const
			{
				Vector3 vec;
				vec.x = x;
				vec.y = y;
				vec.z = z;
				Vector3 vecQ;
				vecQ.x = q.x;
				vecQ.y = q.y;
				vecQ.z = q.z;
				Vector3 vecNew;
				vecNew = vec.cross(vec, vecQ) + vecQ * w + vec * q.w;
				Quaternion qResult = Quaternion( w * q.w - vec.dot(vecQ), vecNew);
				if (qResult.length() > 0) {
					qResult.normalize();
				}
				return qResult;
			}

			//pre - multiply
			friend inline const Quaternion operator * ( const Scalar s, const Quaternion& v )
			{
				return v * s;
			}

			//divide
			const Quaternion operator / ( Scalar s ) const
			{
				s = 1/s;
				return Quaternion( w*s, x*s, y*s, z*s );
			}

			//dot product
			const Scalar dot( const Quaternion& q ) const
			{
				return( w*q.w + x*q.x + y*q.y + z*q.z );
			}

			//magnitude
			const Scalar length() const
			{
				return( (Scalar)sqrt( (double)(w*w + x*x + y*y + z*z) ) );
			}

			//unit vector
			const Quaternion unit() const
			{
				return (*this) / length();
			}

			//make this a unit vector
			void normalize()
			{
				(*this) /= length();
			}

			//equal within an error 'w'
			bool nearlyEquals( const Quaternion& q, const Scalar e ) const
			{
				return fabs(w-q.w)<e && fabs(x-q.x)<e && fabs(y-q.y)<e && fabs(z-q.z)<e;
			}

			/* old
			//spherically linearly interpolate between two unit quaternions, a and b
			inline const Quaternion Slerp( const Quaternion& a, const Quaternion& b, const Scalar u )
			{
			const Scalar theta = acosf( a.dot( b ) );//angle between two unit quaternions
			const Scalar t = 1 / sinf( theta );

			return  t * (a * sinf( (1-u)*theta ) + b * sinf( u*theta ) );
			}
			*/

			//spherically linearly interpolate between two unit quaternions, a and b
			inline const Quaternion Slerp( const Quaternion& a, const Quaternion& b, const Scalar u )
			{

				const Scalar fDot = a.dot( b );
				const Scalar theta = acosf( fDot );//angle between two unit quaternions
				if (theta < 0.001) {
					return a;
				}
				else {
					const Scalar t = 1 / sinf( theta );
					Scalar fCoeff0 = (Scalar)sinf((Scalar)((1.0-u)*theta))*t;
					Scalar fCoeff1 = (Scalar)sinf((Scalar)(u*theta))*t;
					// Do we need to invert rotation?
					if (fDot < 0.0f)
					{
						fCoeff0 = -fCoeff0;
						// taking the complement requires renormalisation
						Quaternion q(fCoeff0 * a + fCoeff1 * b);
						q.normalize();
						return q;
					}
					else {
						return fCoeff0 * a + fCoeff1 * b;
					}

					//return  t * (a * sinf( (1-u)*theta ) + b * sinf( u*theta ) );
				}

			}

			inline void multiply(const Quaternion& left, const Vector3& right) {
				Scalar a,b,c,d;

				a = - left.x*right.x - left.y*right.y - left.z *right.z;
				b =   left.w*right.x + left.y*right.z - right.y*left.z;
				c =   left.w*right.y + left.z*right.x - right.z*left.x;
				d =   left.w*right.z + left.x*right.y - right.x*left.y;


				w = a;
				x = b;
				y = c;
				z = d;
			}
			// convert quaternion to axis angle
			void quatToAA(Vector3 *pAxis, Scalar *pAngle)
			{
				Quaternion q = *this;
				q.normalize();
				*pAngle = (Scalar)(2.0 * std::acos(q.w));

				Scalar fS = (Scalar)(std::sin((*pAngle) / 2.0));
				if ( fS != 0 ) {
					(*pAxis).x = q.x / fS;
					(*pAxis).y = q.y / fS;
					(*pAxis).z = q.z / fS;
				} else {
					*pAxis = Vector3(0, 0, 0);
				}

				// convert from radians to degrees
				*pAngle = (Scalar)(*pAngle * 180.0 / 3.14159225);
			}

			// convert aa to quat
			void fromAA(Vector3 axis, Scalar a)
			{
				Scalar a2 = a*0.01745329252f/2.0f;
				w = std::cos(a2);
				x = std::sin(a2) * axis.x;
				y = std::sin(a2) * axis.y;
				z = std::sin(a2) * axis.z;
				normalize();
			}

			void rotate(Vector3 *v);
		};


	}
}

#endif // !defined(AFX_QUAT4F_H__F1B3C959_7040_4622_8798_4243FDA11191__INCLUDED_)

