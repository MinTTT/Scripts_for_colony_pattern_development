#include "Constants.h"
#include "tools.h"
#include "cmath"
#define CELL_V 1. // calculate rets is　1023

double cellRadius = 0.34;		// radius of each cell cap (constant for now, microns)
double L_divide = 5.75;		// 3 length when cells divide (microns)
double k_cc = 100000.0;			// elastic constant for cell-cell interactions (atm?)
double k_wc = 100000.0;			// elastic constant between cells and wall
double varL = 0.0;			// variation in the length of daughter cells
double varAngle = 0.0;
double var_pos = 0.0;
double viscosity = 1.0;		// dimensionless viscosity of the medium: vis*growth_rate/p_thresh
double wall_rough = 0.0;  // max. height fluctuation of agar surface
double gamma_t = 10000.0;  // Gamma_{cc, t}
double gamma_n = 100.0;  // Gamma_{cc, n}
double wall_mu = 0.15;  // Wall_Mu in Params
double cell_mu = 0.15;  // Cell_Mu in Params
double density_threshold = 0.6;   // Mya set this value to 0.6 previously.
double tension = 10.0;  //Surface_Tension in parameter file.
double DH = 0.0;			// determines how tightly the surface tension holds the cells (+ value is low agar concentration, - value is high agar concentration)

//================= Start -> Toggle Switch parameters ======================//
double Green_conc = 100.;
double Red_conc = 10.;
double tau_G = 0.015;
double tau_R = 0.13;
double maxCellVol = L_divide * pow(cellRadius, 2.) * PI + PI * pow(cellRadius, 3.)* 4./3.;
double kG = 14. / CELL_V;  // default 20.
double kR = 10. / CELL_V;  // default 10.
double nG = 4.;
double nR = 2.;
double deltaP = 0.05;
double e = 2.7182818284590452353602874713527;
int reactionMetric[4][2] = {{1, 0},
                      {0, 1},
                      {-1, 0},
                      {0, -1}};
//=================Toggle Switch parameters <- End ==========================//

// time
double t_max = 20.0; // total simulation time
double initial_dt = 0.00001;	// initial time step
double OutputTime = 100.*initial_dt;		// how often to output
double t0 = 0.0;	// time at start of simulation
double UpdateTime = 100.*initial_dt;

// non-physical constants
int BoxX = 200;	// number of boxes per side of the grid
int BoxY = 200;
int BoxZ = 50;
int BoxZAgar = 30;
int maxLevels = 4;
double BoxLength = 2.;  // CHUPan: Box length is defined in main file, BoxLength = (L_divide+2.*cellRadius);
int FilterLen = 5;


// maximum number of cells in the simulation
int maxCells = 10000000;

// Nutrient constants
double Tortuosity = 2.0; // 2.0; CHU Pan: No use
double KC = 0.02;
double C_rate = 0.;// 3.7e-5*Tortuosity*5.0; CHUPan: defined in main file, C_rate = rho0 * maxGrowthRate / yield / molCarbon * 1e9 ;
double maxGrowthRate = 1.0;
double Maintenance_rate = 0.0;
double Rc = 3.0;
double DiffColony = 1.0;
double DiffAgar = 3.0;
double maxCarbon = 0.01;

double yield = 0.5;  // CHUPan: carbon yield factor
double rho0 = 0.102;  // CHUPan: local mass density
double molCarbon = 180.;  // CHUPan: carbon molecular mass


// Colony constants
double SigmaMoL = 1.5850;  //log(3)/log(2)
double Cdt = 0.1;
double ConvCrit = 1.0e-4;
int minIter = 500;
int maxIter = 20000;
int InterfaceCondition = 1; // 1: continuous $\partial C/partial n$; 2: flux continuity with qC; 3: continuous $\partial C/\partial t$;
bool NutrientGSI = false;
int refinementGridHeight = 4;

// directory name
char DirName[500] = "";
// field file out put flag;
bool FieldOutPut = true;