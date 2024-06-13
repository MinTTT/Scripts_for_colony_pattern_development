//
// Created by pan_c on 8/28/2022.
//
#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include <math.h>

#include "SSAToggle.h"
#include "omp.h"
#define CELL_V 1. // calculate rets is　1023
//#define SEED 123

//double Green_conc = 100.;
//double Red_conc = 10.;

//================= Start -> Toggle Switch parameters ======================//
double tau_G = 0.015;
double tau_R = 0.13;
double kG = 14. / CELL_V;  // default 20.
double kR = 10. / CELL_V;  // default 10.
double nG = 4.;
double nR = 2.;
double deltaP = 0.05;
double e = 2.7182818284590452353602874713527;
//=================Toggle Switch parameters <- End ==========================//

//Hill function for regulators
double hillFunc(const double &leakage, const double &k, const double &n, const double &p){
    // leakage, k, n, p
    return leakage + (1.0 - leakage) / (1.0  + std::pow(p/k, n));
}

//Expression rate for Green state.
double alphaG(const double &gr){
    return 1.1 * gr *(25.609+ 627.747/(1.0+ std::pow(gr/0.865, 4.635)));
//    return gr*(16.609+ 627.747/(1.0+ pow(gr/0.865, 4.635)));

}

//Expression rate for Red state.
double alphaR(const double &gr){
    return 1.1 * gr *(26.836 + 320.215/ (1.0 + std::pow(gr/0.661, 4.09)));
//    return gr*( 26.836 + 320.215/ (1.0+ pow(gr/0.661, 4.09)));

}

// Generates tau where the next reaction will happen.
int generateTau(std::uniform_real_distribution<>& dist,
                std::mt19937& gen,
                const double& sumPropensity, double* pTau){
    if (sumPropensity > 0.){
        *pTau = -log(dist(gen)) / sumPropensity;
        return 0;
    } else{
        return 1;
    }
}

// Finds out the mean value in an array.
template<typename T>
T arrayMin(const T* array, const int& arrayLength){
    T tempMin = array[0];
    for(int i =0; i<arrayLength; ++i){
        tempMin = (tempMin < array[i]) ? array[i] : tempMin;
    }
    return tempMin;
}

// Selects the reaction that will happen at tau
template<size_t SIZE>
int selectReaction(std::uniform_real_distribution<>& dist,
                   std::mt19937& gen,
                   const double& sumPropensity,
                   const double (&p)[SIZE],int* reaction){
    double sp = 0.;
    double rP = dist(gen) * sumPropensity;  //    double rP = static_cast<double>(std::rand())/RAND_MAX * sumPropensity;
    int i;

//    std::cout << SIZE;
    for(i=0; i!=SIZE; i++){
        sp += p[i];
        if(rP < sp){
            *reaction = i;
//            std::cout<< "rP: " << rP << " in selectReaction \n";
            break;
        }
    }
    return 0;
}

// x[ ] = {G, R}
template<size_t SIZE>
void updateP(const int (&x)[SIZE], const double& gr, double *p){
//    const int* px = &x;
    p[0] = alphaG(gr) * hillFunc(tau_G, kR, nR, (double)x[1] / CELL_V); // O -> G
    p[1] = alphaR(gr) * hillFunc(tau_R, kG, nG, (double)x[0] / CELL_V); // O -> R
    p[2] = gr * (double)x[0]; // G -> O
    p[3] = gr * (double)x[1]; // R -> O
}


template<size_t SIZE>
double sum(const double (&p)[SIZE]){
    double sumP = 0.;
    for(unsigned int i=0; i!=SIZE; i++){
        sumP += p[i];
    }
    return sumP;
}


template<size_t rows, size_t cols>
void updateX(const int& reaction, int (&u1)[rows][cols], int* x){
    for(unsigned int i=0; i!=cols; i++){
        *(x + i) += u1[reaction][i];
//        std::cout << u1[reaction][i] << '\n';
    }
}
template<size_t rows, size_t cols>
void initPars(const int& g, const int& r,
              int* x, int (&u1)[rows][cols]){
    ////////////////////////////////////////////////
    // Chemical reaction                       /////
    //                                         /////
    //                                         /////
    // O -> G                                  /////
    // O -> R                                  /////
    // G -> O                                  /////
    // R -> O                                  /////
    ////////////////////////////////////////////////
    x[0] = g;
    x[1] = r;
    u1[0][0] = 1;
    u1[0][1] = 0;
    u1[1][0] = 0;
    u1[1][1] = 1;
    u1[2][0] = -1;
    u1[2][1] = 0;
    u1[3][0] = 0;
    u1[3][1] = -1;
}


template<size_t size, typename T>
void prtV(const T (&x)[size]){
    for(unsigned int i=0; i!=size; i++){
        std::cout << x[i] << "\t";
    }
}

template<size_t size, typename T>
void prtRet(const T (&x)[size], double t){
    std::cout << t << '\t';
    prtV(x);
    std::cout << '\n';
}

/**
 *
 * @param gr cell growth rate
 * @param green initial condition of green state
 * @param red initial condition of red state
 * @param endTime simulation end time (h)
 * @param outputTime record time interval (h)
 * @param t0 start time (h)
 * @param[out] saveT the pointer of the first time point.
 * @param[out] saveX1 the pointer of the first Green state point.
 * @param[out] saveX2 the pointer of the first Red state point.
 * @param[out] saveSize the number of time points that saved in saveT, saveX1, and saveX2 when call this function.
 * @return 0
 */
int runSim(const double& gr, const int& green, const int& red,
           const double& endTime, const double& outputTime, const double& t0,
           double* saveT, int* saveX1, int* saveX2, int* saveSize){
    double sumPropensity = 0.;
    double tau = 0.0;
    double t = t0;
    int reaction;
    double nextOutput = t0;
    int saveIndex = 0;

    int x[2];       // population of chemical species
    double p[4];    // propensity of reactions
    int u1[4][2];   // data structure for updating x[]
    // random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniDist(0.0, 1.0);
    initPars(green, red, x, u1);

    while (true){
        /* Save results, save initial rets and the final state */
        if(t >= nextOutput){
            *(saveT + saveIndex) = t;
            *(saveX1 + saveIndex) = x[0];
            *(saveX2 + saveIndex) = x[1];
            nextOutput += outputTime;
            saveIndex += 1;  // next save index, and also equal to save size.
        }
        /* End save results*/
        updateP(x, gr, p);
        sumPropensity = sum(p);
//        std::cout << "Sum P: "<< sumPropensity << "\n";
        generateTau(uniDist, gen, sumPropensity, &tau); // Generate tau

        /*update the final reaction state*/
        if(t+tau > endTime){
            *saveSize = saveIndex;
            *(saveT + saveIndex-1) = t;
            *(saveX1 + saveIndex-1) = x[0];
            *(saveX2 + saveIndex-1) = x[1];
            break;
        } else{
            t += tau;
        }
        selectReaction(uniDist, gen, sumPropensity, p, &reaction);  // select a reaction
        updateX(reaction, u1, x);
//        std::cout << "t: " <<t<<"\n";
    }
    return 0;
}


int rumMultiSim(const int& threadNum, const double& gr, int* green, int* red,
                      const double& endTime, const double& outputTime, int simsize, double* saveBuff, int* saveLength){
    omp_set_num_threads(threadNum);
    int runSize = (int) floor(endTime / outputTime) + 1;
    double * saveT = new double[runSize*simsize];
    int* saveG = new int[runSize*simsize];
    int* saveR = new int[runSize*simsize];
    int dim12 = 3 * simsize;
    int* sizeArray = new int[simsize];

    #pragma omp parallel for
    for(int iSim=0; iSim < simsize; ++iSim){
//        std::cout<< "Sim #: " << iSim << "\n";
//        std::cout<< "Thread: " << omp_get_thread_num() << '\n';
        runSim(gr, green[iSim], red[iSim], endTime, outputTime, 0,
                    &saveT[iSim*runSize], &saveG[iSim*runSize], &saveR[iSim*runSize],
                    (sizeArray+iSim));
        for(int i=0; i<runSize; ++i) {
            *(saveBuff + i * dim12 + 0 * simsize + iSim) = saveT[iSim * runSize + i];
            *(saveBuff + i * dim12 + 1 * simsize + iSim) = (double) saveG[iSim * runSize + i];
            *(saveBuff + i * dim12 + 2 * simsize + iSim) = (double) saveR[iSim * runSize + i];
        }
    }
    *saveLength = arrayMin(sizeArray, simsize);
    delete []saveT;
    delete []saveG;
    delete []saveR;

    return 0;
}


void appendCell(ToggleCell& cell, ToggleCell* cellpp, const int& size){
    auto * tempCells = new ToggleCell[size + 1];
    for(int i=0; i < size; ++i){
        *(tempCells+i) = *(cellpp + i);
    }
    tempCells[size + 1] = cell;

    delete[] cellpp;
    cellpp = tempCells;
}

// Initializes the Toggle state of a cell, this function is identical to ToggleCell(...).
/**
 * Initializes the Toggle state of a cell, this function is identical to ToggleCell(...).
 * @param cell ToggleCell instance.
 * @param startTime start time point.
 * @param endTime end time point. It will determine the .green .red and .time array size. make sure that the endTime is long enough.
 * @param outputTime time interval that determine the interval between two record points.
 * @param initgreen
 * @param initred
 * @param parent
 * @param lineage
 */
void initCell(ToggleCell* cell, const double& startTime, const double &endTime, const double &outputTime,
              const int& initgreen, const int& initred, const int& parent, const int& lineage){
    int runsize = (int) floor((endTime-startTime ) / outputTime) +10;
    cell->green = new int[runsize];
    cell->red = new int[runsize];
    cell->time = new double [runsize];
    *(cell->green) = initgreen;
    *(cell->red) = initred;
    *(cell->time) = startTime;
    cell->parent = parent;
    cell->lineage= lineage;
    cell->rcdSize = 1;

}


void freeCellMem(ToggleCell* cell){

    delete[] cell->green;
    delete[] cell->red;
    delete[] cell->time;
    cell->green = nullptr;
    cell->red = nullptr;
    cell->time = nullptr;
}

void freeCellArray(ToggleCell* cells, int& size){
//    std::cout << "free cells array in " << cell;
    for(int i=0; i<size; ++i){
        freeCellMem(cells+i);
    }
    delete[] cells;
    cells = nullptr;
}


 /**
  * Simulate the cells in a batch culture.
  * @param[in] threadNum thread numbers used for simulation;
  * @param[in] gr growth rate of cells;
  * @param[in] green initial condition of green state;
  * @param[in] red initial condition of red state;
  * @param[in] endTime
  * @param[in] outputTime
  * @param[in] maxCell
  * @param[out] cellsarray
  * @param[out] cellsSize
  */
void runBatchSim(const int& threadNum, const double& gr, const int& green, const int& red,
      const double& endTime, const double& outputTime, const int &maxCell,
      ToggleCell** cellsarray, int* cellsSize){
    omp_set_num_threads(threadNum);
    double dblTime = std::log(2.) / gr;
    int totalcell = (int) floor(std::pow(e, gr * endTime) * 1.1) ; // allocate more memory resource.
    if(totalcell >= maxCell){
        totalcell = maxCell;
    }
    auto* cellsp = new ToggleCell[totalcell];
    int currentCellnum = 1;
    int divisionCellNum;
    double simFinishTime = endTime;
    double recordTimeInterval = outputTime;

    // init the mother cell
    initCell(&(cellsp[0]), 0.0, simFinishTime, recordTimeInterval, green, red, 0, 1);

    double growthTime = dblTime;  // stop time of each rounds of cells growth simulation.

    while (true){
        // growth all cells in a doubling tine (growthTime)
        #pragma omp parallel for
        for(int i = 0; i<currentCellnum; ++i){
            int stari = cellsp[i].rcdSize-1;  // write the initial state again in first memory block in each growth loop.
            int greenNow = cellsp[i].green[stari];
            int redNow = cellsp[i].red[stari];
            int recordNumberThisLoop = 0;
            runSim(gr, greenNow, redNow, growthTime, outputTime, cellsp[i].time[stari],
                   &cellsp[i].time[stari], &cellsp[i].green[stari], &cellsp[i].red[stari], &recordNumberThisLoop);
            cellsp[i].rcdSize += (recordNumberThisLoop-1);  // because of overwritten in start point, the ACTUAL recording size is recdSize-1
        }
        // check if stop conditions acquired.
        if(growthTime>=endTime){
            *cellsarray = cellsp;
            *cellsSize = currentCellnum;
            // Following code was deprecated because of memory limit.
            if(currentCellnum < totalcell){
                // if the live cell number is not extend to maximum number, we will free the memory of blank cell.
                auto* cells = new ToggleCell[currentCellnum];
                #pragma omp parallel for
                for(int i=0; i<currentCellnum; ++i){
                    *(cells+i) = *(cellsp + i);
                }
                delete[] cellsp;
                *cellsarray = cells;
            }
            break;  // finish the simulation.
        }

        if(endTime - growthTime > dblTime) {
            growthTime += dblTime;  // if the lasting time are longer than a dbl time.
        } else{
            growthTime = endTime;  // lasting time less than a dbl time
        }
        // do cell division
        if(currentCellnum*2 <= maxCell){
            divisionCellNum = currentCellnum;
        } else{
            divisionCellNum = maxCell - currentCellnum;
        }
        #pragma omp parallel for shared(currentCellnum)
        for(int i = 0; i<divisionCellNum; ++i){
            ToggleCell* ptCellp = &cellsp[i];
            int rcdi = ptCellp->rcdSize-1;  // copy the cell states to daughter cell.
            int dgtCelli = i+currentCellnum;
            initCell(&cellsp[dgtCelli], ptCellp->time[rcdi], endTime, outputTime,
                     ptCellp->green[rcdi], ptCellp->red[rcdi], ptCellp->lineage, dgtCelli+1);
        }
        currentCellnum +=divisionCellNum;
    }
}

int runSim(const double& gr, const double& endTime, double* green, double* red){
    double sumPropensity;
    double tau = 0.0;
    double t = 0.0;
    int reaction;

    int x[2];       // population of chemical species
    double p[4];    // propensity of reactions
    int u1[4][2];   // data structure for updating x[]

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniDist(0.0, 1.0); // random generator

    initPars(int(*green), int(*red), x, u1);
    while (true){
        updateP(x, gr, p);
        sumPropensity = sum(p);
        generateTau(uniDist, gen, sumPropensity, &tau); // Generate tau
        std::cout << "Tau: " << tau << "\n";
        /*update the final reaction state*/
        if(t+tau > endTime){
            *green = double(x[0]);
            *red = double(x[1]);
//            printf("SSA_success");
            break;
        } else{
            t += tau;
        }
        selectReaction(uniDist, gen, sumPropensity, p, &reaction);  // select a reaction
        updateX(reaction, u1, x);
    }
    return 0;
}






int main(){
    int repeatSize = 100;
//    cellBatch cells;
//    runBatchSim(24, 1.3, 10,1000,10, 0.1, 10e4, &cells.cells, &cells.size);
//    std::cout << "Total Cells:" << cells.size << '\n';
//    for(int i=0; i<cells.size; ++i){
//        ToggleCell* Cell = &cells.cells[i];
//        std::cout << "Cell #: "<< Cell->lineage << "\t";
//        std::cout << "Cell Time: "<< Cell->time[Cell->rcdSize-1] << "\t";
//        std::cout << "Cell Green: "<< Cell->green[Cell->rcdSize-1] << "\t";
//        std::cout << "Cell Red: "<< Cell->red[Cell->rcdSize-1] << "\n";
//    }
    for(int i=0; i<repeatSize; ++i){
//        cellBatch cells;
        Cell cell = Cell(1.0, 1., 1.);

        Cell* pCell;
        pCell = &cell;
//        runBatchSim(22, 1.2, 1,1,10., 0.1, 10e6, &cells.cells, &cells.size);
        runSim(1.0, 0.005, &(pCell->green), &(pCell->red));
        std::cout << "Repeat Number: " << '\t' <<  i+1 << '\t' << "Cell Green:" <<  pCell->green  <<'\n';

    }
    return 0;
}