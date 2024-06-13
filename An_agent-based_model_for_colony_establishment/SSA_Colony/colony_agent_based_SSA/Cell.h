/*
 * Cell.h
 *
 *  Created on: Dec 15, 2011
 *      Author: mya
 */

#ifndef CELL_H_
#define CELL_H_
#include "Constants.h"
#include "tools.h"


// Cell structure contains information specific to each cell
// Toggle parameters: G, R, GrowthRate, nextReaction
struct Cell{
	Segment Position; 	// a segment has two coordinates p and q for the two vertices of the spherocylinder
	double Length;
	double Radius;
	double GrowthRate;
    double nextReaction;
	DoubleCoord Velocity;
	DoubleCoord AngularVelocity; // now a vector
	DoubleCoord DynFric;
	DoubleCoord StaFric;
	int Type;
	int Ancestor;
    double G;
    double R;

    Cell(){
        this->nextReaction = 0.;
        this->G = Green_conc;
        this->R = Red_conc;
    }
};

#endif /* CELL_H_ */
