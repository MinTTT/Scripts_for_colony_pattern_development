# distutils: language=c++

import numpy as np
cimport numpy as np
from cython cimport view
cdef extern from "SSAToggle.h":

    cdef cppclass ToggleCell:
            int lineage
            int parent
            int rcdSize
            double * time
            int * green
            int * red

    cdef cppclass cellBatch:
        ToggleCell* cells
        int size


    int runSim(const double& gr, const int& green, const int& red,
               const double& endTime, const double& outputTime, const double& t0,
               double * saveT, int * saveX1, int * saveX2, int * saveSize)

    int rumMultiSim(const int& threadNum, const double& gr, int * green, int * red,
                    const double& endTime, const double& outputTime, int simsize, double * saveBuff, int * saveLength);

    void runBatchSim(const int& threadNum, const double& gr, const int& green, const int& red,
                     const double& endTime, const double& outputTime, const int & maxCell,
                     ToggleCell** cellsarray, int * cellsSize)

    void freeCellMem(ToggleCell * cell)
    void freeCellArray(ToggleCell * cell, int& size)


cdef class pyToggleCell:
    cdef ToggleCell* p_cell
    cdef bint ptr_owner


    def __cinit__(self):
        self.ptr_owner = False

    @property
    def lineage(self):
        return self.p_cell.lineage
    @property
    def parent(self):
        return self.p_cell.parent
    @property
    def record_size(self):
        return self.p_cell.rcdSize
    @property
    def green(self):
        return np.asarray(<int[:self.p_cell.rcdSize]> self.p_cell.green)
    @property
    def red(self):
        return np.asarray(<int[:self.p_cell.rcdSize]> self.p_cell.red)
    @property
    def times(self):
        return np.asarray(<double[:self.p_cell.rcdSize]> self.p_cell.time)

    @staticmethod
    cdef pyToggleCell from_ptr(ToggleCell* _ptr, bint owner=False):
        cdef pyToggleCell wrapper = pyToggleCell.__new__(pyToggleCell)
        wrapper.p_cell = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class pyBatchCell:

    cdef cellBatch* cell_batch
    cdef bint ptr_owner
    cdef list cellsList
    cdef int index


    def __cinit__(self):
        self.ptr_owner = True
        self.cell_batch = new cellBatch()
        self.cellsList = []

    def __dealloc__(self):
        del self.cell_batch
        for i in range(self.cell_batch.size):
            cell = self.cellsList.pop()
            del cell
        self.cellsList = []

    @property
    def cells_list(self):
        return self.cellsList

    @property
    def cells_number(self):
        return self.cell_batch.size

    # def __init__(self):
    #     self.cellsList = []

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.cells_number

    def __next__(self):
        if self.index < self.cells_number:
            cell  = self.cells_list[self.index]
            self.index += 1
            return cell
        else:
            raise StopIteration


    @staticmethod
    cdef pyBatchCell from_ptr(cellBatch* _ptr, bint ptr_owner=False):
        # TODO: this part have a bug, can not pass right memory address.
        cdef pyBatchCell wrapper = pyBatchCell.__new__(pyBatchCell)
        wrapper.cell_batch = _ptr
        wrapper.ptr_owner = False
        return wrapper



def pyrunSim(double gr, int green, int red, double initTime, double endTime, double outputTime):
    """
    Parameters
    -------------
    gr(double): growth rate

    """

    cdef int sizeArray = int(endTime // outputTime)
    cdef unsigned int i
    cdef int ret
    cdef int length

    cdef np.ndarray[np.double_t, ndim=1] saveT = np.ones((sizeArray, ), dtype=np.double) * -1.
    cdef np.ndarray[np.int_t, ndim=1] saveX1 = np.ones((sizeArray, ), dtype=int) * -1
    cdef np.ndarray[np.int_t, ndim=1] saveX2 = np.ones((sizeArray, ), dtype=int) * -1

    ret = runSim(gr, green, red, endTime, outputTime, initTime,
                 <double *> saveT.data, <int *> saveX1.data, <int *> saveX2.data, &length)
    if ret  != 0:
        raise IOError()

    saveT = saveT[:length]
    saveX1 = saveX1[:length]
    saveX2 = saveX2[:length]
    rets = np.hstack([saveT.reshape((-1, 1)), saveX1.reshape((-1, 1)), saveX2.reshape((-1, 1))])

    return rets


def pyrunMultSim(double gr, np.ndarray[np.int_t, ndim=1] green, np.ndarray[np.int_t, ndim=1] red, double endTime,
                 double outputTime, int threadNum=24):
    """

    Parameters
    ----------
    gr : float
        cell growth rate.
    green : np.ndarray
        cells' initial GFP level.
    red : np.ndarray
        cells' initial mCherry level.
    endTime : float
        The total time for SSA simulation.
    outputTime :
    threadNum :

    Returns
    -------
        Results
            [Time, Cell states, Cells]. Cells sates: (current time, green level, red leve). Cells: (Cell #1, Cell #2, ...)
    """
    cdef int sizeofSime = green.size
    # print(sizeofSime)
    cdef int sizeofTime = int(endTime // outputTime) + 1
    cdef int saveSize

    cdef np.ndarray[np.double_t, ndim=3] saveBuff = np.ones((sizeofTime, 3, sizeofSime), dtype=np.double) * -1

    # print(saveBuff)

    rumMultiSim(threadNum,
                gr, <int *> green.data, <int *> red.data,
                endTime, outputTime, sizeofSime,
                &saveBuff[0, 0, 0], &saveSize)
    saveBuff = saveBuff[:saveSize, :, : ]

    return saveBuff

def pyrunBatchSim(int threadNum, double gr, int green, int red,
                     double endTime, double outputTime, int maxCell,
                     ):

    cdef pyBatchCell batch_cells = pyBatchCell()

    runBatchSim(threadNum,  gr, green, red, endTime, outputTime, maxCell,
                &batch_cells.cell_batch.cells, &batch_cells.cell_batch.size)

    # batch_cells.cellsList = [pyToggleCell.from_ptr(&batch_cells.cell_batch.cells[i])
    #                       for i in range(batch_cells.cell_batch.size)]
    for i in range(batch_cells.cell_batch.size):
        batch_cells.cellsList.append(pyToggleCell.from_ptr(&batch_cells.cell_batch.cells[i]))

    return batch_cells
