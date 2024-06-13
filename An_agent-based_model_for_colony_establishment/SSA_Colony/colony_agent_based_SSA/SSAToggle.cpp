//
// Created by pan_c on 8/28/2022.
//
#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include "Constants.h"
#include "SSAToggle.h"
#include "Cell.h"


std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> uniDist(0.0, 1.0); // random generator

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
                std::mt19937& generator_ran,
                const double& sumPropensity, double* pTau){
    if (sumPropensity > 0.){
        *pTau = -log(dist(generator_ran)) / sumPropensity;
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
                   std::mt19937& generator_ran,
                   const double& sumPropensity,
                   const double (&p)[SIZE],int* reaction){
    double sp = 0.;
    double rP = dist(generator_ran) * sumPropensity;  //    double rP = static_cast<double>(std::rand())/RAND_MAX * sumPropensity;
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

/*
 * Calculate the cell volume.
 *
 * @param cell_length cell length, cylinder length.
 * @param cell_radius cell radius, hemisphere radius.
 */
double length2Vol(const double& cell_length, const double& cell_radius){
    return cell_length * pow(cell_radius, 2.) * PI + PI * pow(cell_radius, 3.)* 4./6.;
}

void updateP(const Cell& cell, double*p){
    double growthRate = cell.GrowthRate / SigmaMoL;
//    double cellVol = length2Vol(cell.Length, cell.Radius);
    p[0] = alphaG(growthRate) * hillFunc(tau_G, kR, nR, (double)cell.R / CELL_V); // O -> G
    p[1] = alphaR(growthRate) * hillFunc(tau_R, kG, nG, (double)cell.G / CELL_V); // O -> R
    p[2] = growthRate * (double)cell.G; // G -> O
    p[3] = growthRate * (double)cell.R; // R -> O
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
void updateX(const int& reaction, int (&u1)[rows][cols], Cell* pCell){
    pCell->G += u1[reaction][0];
    pCell->R += u1[reaction][1];
}

/*
 * Initial the parameters of the chemical molecules number matrix and the reaction function.
 *colon
 * @param g GFP number
 * @param r RFP number
 * @param[out] x pin of the matrix storing the molecular numbers
 * @param[Out] u1 chemical reaction matrix
 */
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


/*
*/
int runSim(const double& gr, const double& endTime, double* green, double* red){
    double sumPropensity;
    double tau = 0.0;
    double t = 0.0;
    int reaction;
    int x[2];       // population of chemical species
    double p[4];    // propensity of reactions
    int u1[4][2];   // data structure for updating x[]
//
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<double> uniDist(0.0, 1.0); // random generator
    initPars(int(*green), int(*red), x, u1);
    while (true){
        updateP(x, gr, p);
        sumPropensity = sum(p);
        generateTau(uniDist, gen, sumPropensity, &tau); // Generate tau
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

int runSim(const double& timeNow, Cell* pCell){
    double sumPropensity;
    double tau = 0.0;
    int reaction;
    double timeCell = pCell->nextReaction;
    double p[4];    // propensity of reactions

//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<double> uniDist(0.0, 1.0); // random generator

    while (timeCell <= timeNow){
        updateP(*pCell, p);
        sumPropensity = sum(p);
        generateTau(uniDist, gen, sumPropensity, &tau); // Generate tau
        timeCell +=tau;
        if (timeCell > timeNow){
            break;
        } else{
            selectReaction(uniDist, gen, sumPropensity, p, &reaction);  // select a reaction
            updateX(reaction, reactionMetric, pCell);
        }
    }
    pCell->nextReaction = timeCell - tau;  // next reaction failed and stay at origin state.

    return 0;
}


int runSim(const double& gr, const int& green, const int& red,
           const double& endTime, const double& outputTime, const double& t_0,
           double* saveT, int* saveX1, int* saveX2, int* saveSize){
    double sumPropensity = 0.;
    double tau = 0.0;
    double t = t_0;
    int reaction;
    double nextOutput = t_0;
    int saveIndex = 0;

    int x[2];       // population of chemical species
    double p[4];    // propensity of reactions
    int u1[4][2];   // data structure for updating x[]
//    // random generator
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<double> uniDist(0.0, 1.0);
    initPars(green, red, x, u1);

    while (true){
        /* Save results, save initial rets and the final state */
        if(t >= nextOutput){
            *(saveT + saveIndex) = t;
            *(saveX1 + saveIndex) = x[0];
            *(saveX2 + saveIndex) = x[1];
            nextOutput += outputTime;
            saveIndex += 1;
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
    int runSize = (int) floor(endTime / outputTime) + 1;
    auto* saveT = new double[runSize*simsize];
    auto* saveG = new int[runSize*simsize];
    auto* saveR = new int[runSize*simsize];
    int dim12 = 3 * simsize;
    auto* sizeArray = new int[simsize];

#pragma omp parallel for num_threads(threadNum)
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
    auto* tempCells = new ToggleCell[size + 1];
    for(int i=0; i < size; ++i){
        *(tempCells+i) = *(cellpp + i);
    }
    tempCells[size + 1] = cell;
    delete[] cellpp;
    cellpp = tempCells;
}

// Initializes the Toggle state of a cell, this function is identical to ToggleCell(...).
void initCell(ToggleCell* cell, const double& startTime, const double &endTime, const double &outputTime,
              const int& initgreen, const int& initred, const int& parent, const int& lineage){
    int runsize = (int) floor((endTime-startTime ) / outputTime) +1;
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


void runBatchSim(const int& threadNum, const double& gr, const int& green, const int& red,
      const double& endTime, const double& outputTime, const int &maxCell,
      ToggleCell** cellsarray, int* cellsSize){

    double dblTime = log(2.) / gr;
    int totalcell = (int) floor(std::pow(e, gr * endTime) * 1.1) ; // allocate more memory resource.
    if(totalcell >= maxCell || totalcell <0){
        totalcell = maxCell;
    }
    auto* cellsp = new ToggleCell[totalcell];
    int currentCellnum = 1;
    int divisionCellNum;
//    std::cout << "Estimate cells: "<< totalcell << "\n";

    initCell(&cellsp[currentCellnum-1], 0., endTime, outputTime, green, red, 0, 1);
//    std::cout << "Mother cells: " << cellsp[currentCellnum-1].lineage << "\n";
//    std::cout << "Mother cells Green: " << *cellsp[currentCellnum-1].green << "\n";
//    std::cout << "Mother cells Red: " << *cellsp[currentCellnum-1].red << "\n";

    double growthTime = dblTime;
    while (true){
//        std::cout<< "Current Cell Number:" << currentCellnum <<"\t";
//        std::cout<< "Time lapsed: " << growthTime <<"\n";

#pragma omp parallel for num_threads(threadNum)
        for(int i = 0; i<currentCellnum; ++i){
            int stari = cellsp[i].rcdSize-1;  // write the initial state again in first memory block.
            int greenNow = cellsp[i].green[stari];
            int redNow = cellsp[i].red[stari];
            int recdSize=0;

            runSim(gr, greenNow, redNow, growthTime, outputTime, cellsp[i].time[stari],
                   &cellsp[i].time[stari], &cellsp[i].green[stari], &cellsp[i].red[stari], &recdSize);
            cellsp[i].rcdSize += (recdSize-1);
        }

        if(growthTime==endTime){
            *cellsarray = cellsp;
            *cellsSize = currentCellnum;
            if(currentCellnum < totalcell){
                // if the live cell number is not extend to maximum number, we will free the memory of blank cell.
                auto* cells = new ToggleCell[currentCellnum];
                for(int i=0; i<currentCellnum; ++i){
                    *(cells+i) = *(cellsp + i);
                }
                *cellsarray = cells;
                delete[] cellsp;
//                std::cout << "delete cells \n";
            }
            break;
        }

        if(endTime - growthTime > dblTime) {
            growthTime += dblTime;
        } else{
            growthTime = endTime;
        }
        // division
        if(currentCellnum*2 <= maxCell){
            divisionCellNum = currentCellnum;
        } else{
            divisionCellNum = maxCell - currentCellnum;
        }
#pragma omp parallel for num_threads(threadNum)
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

//int main(){
//    int repeatSize = 100;
////    cellBatch cells;
////    runBatchSim(24, 1.3, 10,1000,10, 0.1, 10e4, &cells.cells, &cells.size);
////    std::cout << "Total Cells:" << cells.size << '\n';
////    for(int i=0; i<cells.size; ++i){
////        ToggleCell* Cell = &cells.cells[i];
////        std::cout << "Cell #: "<< Cell->lineage << "\t";
////        std::cout << "Cell Time: "<< Cell->time[Cell->rcdSize-1] << "\t";
////        std::cout << "Cell Green: "<< Cell->green[Cell->rcdSize-1] << "\t";
////        std::cout << "Cell Red: "<< Cell->red[Cell->rcdSize-1] << "\n";
////    }
//    for(int i=0; i<repeatSize; ++i){
//        cellBatch cells;
//        runBatchSim(22, 1.2, 1,1,10., 0.1, 10e6, &cells.cells, &cells.size);
//        std::cout << "Repeat Number: " << '\t' <<  i+1 << '\t' << "Cell Number:" <<  cells.size  <<'\n';
//
//    }
//    return 0;
//}