# cython: language_level=3
# distutils: language=c++
cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

cdef extern from *:
    """
    static const int ioff[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int joff[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    """
    const int ioff[8]
    const int joff[8]

import time
import logging
import sys
import heapq

from osgeo import gdal
import scipy.sparse.csgraph
import numpy

cimport cython
cimport numpy
from libc.math cimport sqrtf
from libc.time cimport time as ctime
from libc.time cimport time_t
from libcpp.deque cimport deque

# exposing stl::priority_queue so we can have all 3 template arguments so
# we can pass a different Compare functor
cdef extern from "<queue>" namespace "std" nogil:
    cdef cppclass priority_queue[T, Container, Compare]:
        priority_queue() except +
        priority_queue(priority_queue&) except +
        priority_queue(Container&)
        bint empty()
        void pop()
        void push(T&)
        size_t size()
        T& top()

# this is the class type that'll get stored in the priority queue
cdef struct ValuePixelType:
    float t_time  # pixel value
    float edge_weight  # pixel value
    int i  # pixel i coordinate in the raster
    int j  # pixel j coordinate in the raster


# this type is used to create a priority queue on a time/coordinate type
ctypedef priority_queue[
    ValuePixelType, deque[ValuePixelType], LessPixel] DistPriorityQueueType

# functor for priority queue of pixels
cdef cppclass LessPixel nogil:
    bint get "operator()"(ValuePixelType& lhs, ValuePixelType& rhs):
        return lhs.t_time > rhs.t_time


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def find_mask_reach(
        numpy.ndarray[float, ndim=2] friction_array,
        numpy.ndarray[numpy.int8_t, ndim=2] mask_array,
        float cell_length_m,
        int n_cols, int n_rows,
        float max_time):
    """Define later

    Parameters:
        friction_array (numpy.ndarray): array with friction values for
            determining lcp in units minutes/pixel.
        mask_array (numpy.ndarray): array with 1 or 0 indicating mask location
        cell_length_m (float): length of cell in meters.
        n_cols/n_rows (int): number of cells in i/j direction of given arrays.
        max_time (float): the time allowed when computing population reach
            in minutes.

    Returns:
        2D array of mask reach of the same size as input arrays.

    """
    cdef int i, j
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] mask_coverage = numpy.zeros(
        (n_rows, n_cols), dtype=numpy.uint8)
    cdef numpy.ndarray[float, ndim=2] current_time = numpy.where(
        mask_array == 1, 0, numpy.inf).astype(numpy.float32)

    cdef float diag_cell_length_m = sqrtf(<float>(2*cell_length_m*cell_length_m))
    cdef float frict_n, c_time, n_time, edge_weight
    cdef int i_start, j_start, i_n, j_n
    cdef int min_i, min_j, max_i, max_j
    cdef int mask_val

    cdef DistPriorityQueueType dist_queue
    cdef ValuePixelType pixel
    with nogil:
        for i_start in range(n_cols):
            for j_start in range(n_rows):
                if mask_array[j_start, i_start] != 1:
                    continue
                mask_coverage[j_start, i_start] = 1

                pixel.t_time = 0
                pixel.edge_weight = 0
                pixel.i = i_start
                pixel.j = j_start
                dist_queue.push(pixel)
                min_i = i_start
                max_i = i_start
                min_j = j_start
                max_j = j_start

                # c_ -- current, n_ -- neighbor
                while dist_queue.size() > 0:
                    pixel = dist_queue.top()
                    dist_queue.pop()
                    c_time = pixel.t_time
                    i = pixel.i
                    j = pixel.j
                    if c_time > current_time[j, i]:
                        # this means another path already reached here that's
                        # better
                        continue
                    mask_coverage[j, i] = 1
                    if i < min_i:
                        min_i = i
                    elif i > max_i:
                        max_i = i
                    if j < min_j:
                        min_j = j
                    elif j > max_j:
                        max_j = j

                    for v in range(8):
                        i_n = i+ioff[v]
                        j_n = j+joff[v]
                        if i_n < 0 or i_n >= n_cols:
                            continue
                        if j_n < 0 or j_n >= n_rows:
                            continue
                        if mask_array[j_n, i_n] < 0:
                            # nodata, so skip
                            continue
                        frict_n = friction_array[j_n, i_n]
                        # the nodata value is undefined but will present as 0.
                        if frict_n <= 0:
                            continue
                        if v & 1:  # if msd is 1, it's odd
                            edge_weight = frict_n*diag_cell_length_m
                        else:
                            edge_weight = frict_n*cell_length_m

                        n_time = c_time + edge_weight
                        if n_time > max_time:
                            continue
                        # if visited before and we got there faster, then skip
                        if n_time >= current_time[j_n, i_n]:
                            continue
                        current_time[j_n, i_n] = n_time
                        pixel.t_time = n_time
                        pixel.edge_weight = edge_weight
                        pixel.i = i_n
                        pixel.j = j_n
                        dist_queue.push(pixel)
    return mask_coverage
