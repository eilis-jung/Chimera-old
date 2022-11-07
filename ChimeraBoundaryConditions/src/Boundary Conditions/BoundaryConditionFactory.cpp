#include "Boundary Conditions/BoundaryConditionFactory.h"


namespace Chimera {
	namespace BoundaryConditions {

		BoundaryCondition<Vector2> * BoundaryConditionFactory::loadBoundaryCondition2D(const auto_ptr<ifstream> &fileStream, dimensions_t dimensions){ 
			string fString;
			boundaryLocation_t boundaryLocation = None;
			range1D_t boundaryRange;
			boundaryRange.initialRange = 0; boundaryRange.finalRange = 0;

			/************************************************************************/
			/* Boundary condition location                                          */
			/************************************************************************/
			(*fileStream) >> fString;

			/** North */
			if(fString == "allNorth") {
				boundaryLocation = North;
				boundaryRange.initialRange = 0;
				boundaryRange.finalRange = dimensions.x; 
			} else if(fString == "partialNorth") {
				boundaryLocation = North;
				(*fileStream) >> boundaryRange.initialRange;
				(*fileStream) >> boundaryRange.finalRange;
			}

			/** South */
			else if(fString == "allSouth") {
				boundaryLocation = South;
				boundaryRange.initialRange = 0;
				boundaryRange.finalRange = dimensions.x; 
			} else if(fString == "partialSouth") {
				boundaryLocation = South;
				Scalar partialXIni, partialXFin;
				(*fileStream) >> partialXIni;
				(*fileStream) >> partialXFin;
				boundaryRange.initialRange = floor(partialXIni*dimensions.x);
				boundaryRange.finalRange = floor(partialXFin*dimensions.x);
			}

			/** West */
			else if(fString == "allWest") {
				boundaryLocation = West;
				boundaryRange.initialRange = 0;
				boundaryRange.finalRange = dimensions.y; 
			} else if(fString == "partialWest") {
				boundaryLocation = West;
				(*fileStream) >> boundaryRange.initialRange;
				(*fileStream) >> boundaryRange.finalRange;
			}

			/** West */
			else if(fString == "allEast") {
				boundaryLocation = East;
				boundaryRange.initialRange = 0;
				boundaryRange.finalRange = dimensions.y; 
			} else if(fString == "partialEast") {
				boundaryLocation = East;
				(*fileStream) >> boundaryRange.initialRange;
				(*fileStream) >> boundaryRange.finalRange;
			}

			if(boundaryRange.initialRange > boundaryRange.finalRange)
				throw "Initial is greater than final range.";

			/************************************************************************/
			/* Boundary condition type                                              */
			/************************************************************************/
			(*fileStream) >> fString;

			BoundaryCondition<Vector2> *pCondition = NULL;
			/** Inflow */
			if(fString == "inflow") {
				Vector2 velocity;
				(*fileStream) >> velocity.x;
				(*fileStream) >> velocity.y;
				pCondition = new InflowBC<Vector2>(boundaryLocation, boundaryRange, dimensions);
				((InflowBC<Vector2> *) pCondition)->setVelocity(velocity);
			} 

			/** Outflow */
			else if(fString == "outflow") {
				Scalar pressure = -1;
				(*fileStream) >> pressure;
				pCondition = new OutflowBC<Vector2>(boundaryLocation, boundaryRange, dimensions);
			}

			else if(fString == "noSlip") {
				pCondition = new NoSlipBC<Vector2>(boundaryLocation, boundaryRange, dimensions);
			}

			else if(fString == "freeSlip") {
				pCondition = new FreeSlipBC<Vector2>(boundaryLocation, boundaryRange, dimensions);
			}

			else if(fString == "periodic") {
				pCondition = new PeriodicBC<Vector2>(boundaryLocation, boundaryRange, dimensions);
			}

			else if(fString == "jet") {
				Logger::get() << "Boundary Type: Jet " << endl;
				Vector2 velocity;
				Scalar size, densityVariation, temperatureVariation, minDensity, minTemperature;
				(*fileStream) >> velocity.x;
				(*fileStream) >> velocity.y;
				(*fileStream) >> fString; //size
				(*fileStream) >> size;
				(*fileStream) >> fString; //alpha
				(*fileStream) >> densityVariation;
				(*fileStream) >> fString; //beta
				(*fileStream) >> temperatureVariation;
				(*fileStream) >> fString; //minDensity
				(*fileStream) >> minDensity;
				(*fileStream) >> fString; //minTemperature
				(*fileStream) >> minTemperature;
				pCondition = new JetBC<Vector2>(boundaryLocation, boundaryRange, dimensions);
				((JetBC<Vector2> *) pCondition)->setParameters(velocity, size, temperatureVariation, densityVariation, minDensity, minTemperature);

			} else if(fString == "farfield") {
				Vector2 velocity;
				(*fileStream) >> velocity.x;
				(*fileStream) >> velocity.y;
				pCondition = new FarFieldBC<Vector2>(boundaryLocation, boundaryRange, dimensions);
				((FarFieldBC<Vector2> *) pCondition)->setVelocity(velocity);
			}

			return pCondition;
		}

		BoundaryCondition<Vector3> * BoundaryConditionFactory::loadBoundaryCondition3D(const auto_ptr<ifstream> &fileStream, dimensions_t dimensions) { 
			string fString;
			boundaryLocation_t boundaryLocation = None;
			range1D_t boundaryRange;
			boundaryRange.initialRange = 0; boundaryRange.finalRange = 0;

			/************************************************************************/
			/* Boundary condition location                                          */
			/************************************************************************/
			(*fileStream) >> fString;

			/** North */
			if(fString == "allNorth") {
				Logger::get() << "Boundary Location: North " << endl;
				boundaryLocation = North;
				boundaryRange.initialRange = 1;
				boundaryRange.finalRange = dimensions.x - 1; 
			} else if(fString == "partialNorth") {
				boundaryLocation = North;
				(*fileStream) >> boundaryRange.initialRange;
				(*fileStream) >> boundaryRange.finalRange;
			}

			/** South */
			else if(fString == "allSouth") {
				Logger::get() << "Boundary Location: South " << endl;
				boundaryLocation = South;
				boundaryRange.initialRange = 1;
				boundaryRange.finalRange = dimensions.x - 1; 
			} else if(fString == "partialSouth") {
				boundaryLocation = South;
				(*fileStream) >> boundaryRange.initialRange;
				(*fileStream) >> boundaryRange.finalRange;
			}

			/** West */
			else if(fString == "allWest") {
				Logger::get() << "Boundary Location: West " << endl;
				boundaryLocation = West;
				boundaryRange.initialRange = 1;
				boundaryRange.finalRange = dimensions.y - 1; 
			} else if(fString == "partialWest") {
				boundaryLocation = West;
				(*fileStream) >> boundaryRange.initialRange;
				(*fileStream) >> boundaryRange.finalRange;
			}

			/** East */
			else if(fString == "allEast") {
				Logger::get() << "Boundary Location: East " << endl;
				boundaryLocation = East;
				boundaryRange.initialRange = 1;
				boundaryRange.finalRange = dimensions.y - 1; 
			} else if(fString == "partialEast") {
				boundaryLocation = East;
				(*fileStream) >> boundaryRange.initialRange;
				(*fileStream) >> boundaryRange.finalRange;
			}

			else if(fString == "allFront") {
				Logger::get() << "Boundary Location: Front " << endl;
				boundaryLocation = Front;
				boundaryRange.initialRange = 1;
				boundaryRange.finalRange = dimensions.z - 1; 
			} else if(fString == "partialFront") {

			}

			else if(fString == "allBack") {
				Logger::get() << "Boundary Location: Back " << endl;
				boundaryLocation = Back;
				boundaryRange.initialRange = 1;
				boundaryRange.finalRange = dimensions.z - 1; 
			} else if(fString == "partialBack") {

			}

			if(boundaryRange.initialRange > boundaryRange.finalRange)
				throw "Initial is greater than final range.";

			/************************************************************************/
			/* Boundary condition type                                              */
			/************************************************************************/
			(*fileStream) >> fString;

			BoundaryCondition<Vector3> *pCondition= NULL;
			/** Inflow */
			if(fString == "inflow") {
				Vector3 velocity;
				(*fileStream) >> velocity.x;
				(*fileStream) >> velocity.y;
				(*fileStream) >> velocity.z;

				Logger::get() << "Boundary Type: Inflow " << endl;
				Logger::get() << "Velocity (" << velocity.x << ", " << velocity.y << ", " << velocity.z << ") " << endl;
				pCondition = new InflowBC<Vector3>(boundaryLocation, boundaryRange, dimensions);
				((InflowBC<Vector3> *) pCondition)->setVelocity(velocity);
			} 

			/** Outflow */
			else if(fString == "outflow") {
				Scalar pressure = -1;
				(*fileStream) >> pressure;

				Logger::get() << "Boundary Type: Outflow " << endl;
				pCondition = new OutflowBC<Vector3>(boundaryLocation, boundaryRange, dimensions);
			}

			else if(fString == "noSlip") {
				Logger::get() << "Boundary Type: NoSlip " << endl;
				pCondition = new NoSlipBC<Vector3>(boundaryLocation, boundaryRange, dimensions);
			}

			else if(fString == "freeSlip") {
				Logger::get() << "Boundary Type: FreeSlip " << endl;
				pCondition = new FreeSlipBC<Vector3>(boundaryLocation, boundaryRange, dimensions);
			}

			else if(fString == "jet") {
				Logger::get() << "Boundary Type: Jet " << endl;
				Vector3 velocity;
				Scalar size, densityVariation, temperatureVariation, minDensity, minTemperature;
				(*fileStream) >> velocity.x;
				(*fileStream) >> velocity.y;
				(*fileStream) >> velocity.z;
				(*fileStream) >> fString; //size
				(*fileStream) >> size;
				(*fileStream) >> fString; //alpha
				(*fileStream) >> densityVariation;
				(*fileStream) >> fString; //beta
				(*fileStream) >> temperatureVariation;
				(*fileStream) >> fString; //minDensity
				(*fileStream) >> minDensity;
				(*fileStream) >> fString; //minTemperature
				(*fileStream) >> minTemperature;
				pCondition = new JetBC<Vector3>(boundaryLocation, boundaryRange, dimensions);
				((JetBC<Vector3> *) pCondition)->setParameters(velocity, size, temperatureVariation, densityVariation, minDensity, minTemperature);

			}

			else if(fString == "periodic") {
				pCondition = new PeriodicBC<Vector3>(boundaryLocation, boundaryRange, dimensions);
			}

			return pCondition;
		}
	}
}