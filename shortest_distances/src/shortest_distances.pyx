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
import heapq

from osgeo import gdal
import scipy.sparse.csgraph
import numpy

cimport cython
cimport numpy
from libc.math cimport sqrt
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


LOGGER = logging.getLogger(__name__)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def find_mask_reach(
        numpy.ndarray[float, ndim=2] friction_array,
        numpy.ndarray[numpy.int8_t, ndim=2] mask_array,
        float cell_length_m,
        int n_cols, int n_rows,
        float max_time,
        int progress_interval_seconds=10):
    """Define later

    Parameters:
        friction_array (numpy.ndarray): array with friction values for
            determining lcp in units minutes/pixel.
        mask_array (numpy.ndarray): array with 1 or 0 indicating mask location
        cell_length_m (float): length of cell in meters.
        n_cols/n_rows (int): number of cells in i/j direction of given arrays.
        max_time (float): the time allowed when computing population reach
            in minutes.
        progress_interval_seconds (int): minimum number of seconds between
            travel-reach heartbeat log messages. Set to 0 to disable progress
            logging.

    Returns:
        2D array of mask reach of the same size as input arrays.

    """
    cdef int i, j
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] mask_coverage = numpy.zeros(
        (n_rows, n_cols), dtype=numpy.uint8)
    cdef numpy.ndarray[float, ndim=2] current_time

    cdef float diag_cell_length_m = <float>sqrt(
        <double>(2*cell_length_m*cell_length_m))
    cdef float frict_n, c_time, n_time, edge_weight
    cdef int i_start, j_start, i_n, j_n
    cdef int v
    cdef bint is_frontier
    cdef unsigned long long covered_count = 0
    cdef unsigned long long pop_count = 0
    cdef unsigned long long frontier_source_count = 0
    cdef unsigned long long heartbeat_count = 0
    cdef unsigned long long progress_check_interval = 100000
    cdef unsigned long long next_progress_check = progress_check_interval
    cdef time_t start_time = ctime(NULL)
    cdef time_t last_report_time = start_time
    cdef time_t report_time
    cdef double elapsed_seconds = 0
    cdef double pop_rate = 0
    cdef double frontier_percent = 0
    cdef size_t queue_size
    cdef bint report_needed = False

    cdef DistPriorityQueueType dist_queue
    cdef ValuePixelType pixel
    if progress_interval_seconds > 0:
        LOGGER.info(
            "preparing travel reach frontier; raster=%sx%s px, "
            "max_time=%.1f min",
            n_cols, n_rows, max_time)
    with nogil:
        for i_start in range(n_cols):
            for j_start in range(n_rows):
                if mask_array[j_start, i_start] != 1:
                    continue
                mask_coverage[j_start, i_start] = 1
                covered_count += 1

                is_frontier = False
                for v in range(8):
                    i_n = i_start+ioff[v]
                    j_n = j_start+joff[v]
                    if i_n < 0 or i_n >= n_cols:
                        continue
                    if j_n < 0 or j_n >= n_rows:
                        continue
                    if mask_array[j_n, i_n] != 0:
                        continue
                    if friction_array[j_n, i_n] <= 0:
                        continue
                    is_frontier = True
                    break

                if not is_frontier:
                    continue

                pixel.t_time = 0
                pixel.edge_weight = 0
                pixel.i = i_start
                pixel.j = j_start
                dist_queue.push(pixel)
                frontier_source_count += 1

    if dist_queue.empty():
        if progress_interval_seconds > 0:
            LOGGER.info(
                "travel reach completed without expansion; covered=%s px",
                covered_count)
        return mask_coverage

    current_time = numpy.full((n_rows, n_cols), max_time+1, dtype=numpy.float32)
    if progress_interval_seconds > 0:
        LOGGER.info(
            "starting travel reach expansion; frontier_sources=%s, "
            "covered_sources=%s px, max_time=%.1f min",
            frontier_source_count, covered_count, max_time)

    while dist_queue.size() > 0:
        report_needed = False
        with nogil:
            # c_ -- current, n_ -- neighbor
            while dist_queue.size() > 0:
                pixel = dist_queue.top()
                dist_queue.pop()
                pop_count += 1
                c_time = pixel.t_time
                i = pixel.i
                j = pixel.j

                if (progress_interval_seconds > 0 and
                        pop_count >= next_progress_check):
                    next_progress_check = pop_count + progress_check_interval
                    report_time = ctime(NULL)
                    if report_time - last_report_time >= progress_interval_seconds:
                        last_report_time = report_time
                        heartbeat_count += 1
                        elapsed_seconds = <double>(report_time - start_time)
                        if elapsed_seconds > 0:
                            pop_rate = pop_count / elapsed_seconds
                        else:
                            pop_rate = 0
                        if max_time > 0:
                            frontier_percent = 100.0 * c_time / max_time
                        else:
                            frontier_percent = 0
                        report_needed = True

                if mask_array[j, i] != 1 and c_time > current_time[j, i]:
                    # this means another path already reached here that's better
                    if report_needed:
                        queue_size = dist_queue.size()
                        break
                    continue
                if mask_coverage[j, i] == 0:
                    covered_count += 1
                mask_coverage[j, i] = 1

                for v in range(8):
                    i_n = i+ioff[v]
                    j_n = j+joff[v]
                    if i_n < 0 or i_n >= n_cols:
                        continue
                    if j_n < 0 or j_n >= n_rows:
                        continue
                    if mask_array[j_n, i_n] != 0:
                        # nodata or source cells are already covered
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

                if report_needed:
                    queue_size = dist_queue.size()
                    break
        if report_needed:
            LOGGER.info(
                "travel reach heartbeat: frontier=%.1f/%.1f min "
                "(%.1f%% of limit), covered=%s px, queue=%s, "
                "pops=%s, rate=%.0f pops/s, elapsed=%.0fs",
                c_time, max_time, frontier_percent, covered_count,
                queue_size, pop_count, pop_rate, elapsed_seconds)
    if progress_interval_seconds > 0:
        elapsed_seconds = <double>(ctime(NULL) - start_time)
        if elapsed_seconds > 0:
            pop_rate = pop_count / elapsed_seconds
        else:
            pop_rate = 0
        LOGGER.info(
            "travel reach complete: covered=%s px, pops=%s, "
            "rate=%.0f pops/s, elapsed=%.0fs, heartbeats=%s",
            covered_count, pop_count, pop_rate, elapsed_seconds,
            heartbeat_count)
    return mask_coverage
