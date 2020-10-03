/*
    This file is part of darktable,
    Copyright (C) 2019-2020 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "common/fast_guided_filter.h"
#include "common/gaussian.h"

/***
 * DOCUMENTATION
 *
 * Exposure-Independent Guided Filter (EIGF)
 *
 * This filter is a modification of guided filter to make it exposure independent
 * As variance depends on the exposure, the original guided filter preserves
 * much better the edges in the highlights than in the shadows.
 * In particular doing:
 * (1) increase exposure by 1EV
 * (2) guided filtering
 * (3) decrease exposure by 1EV
 * is NOT equivalent to doing the guided filtering only.
 *
 * To overcome this, instead of using variance directly to determine "a",
 * we use a ratio:
 * variance / (pixel_value)^2
 * we tried also the following ratios:
 * - variance / average^2
 * - variance / (pixel_value * average)
 * we kept variance / (pixel_value)^2 as it seemed to behave a bit better than
 * the other (dividing by average^2 smoothed too much dark details surrounded
 * by bright pixels).
 *
 * This modification makes the filter exposure-independent.
 * However, due to the fact that the average advantages the bright pixels
 * compared to dark pixels if we consider that human eye sees in log,
 * we get strong bright halos.
 * These are due to the spatial averaging of "a" and "b" that is performed at
 * the end of the filter, especially due to the spatial averaging of "b".
 * However, removing completely this averaging gives results which are not
 * smoothed enough.
 * Hence, we use a weighted averaging of "a" and "b" to overcome this problem.
 * We weight each "a" and "b" by 1 / pixel_value^2, which gives much less
 * halos problems, and gives a smooth result.
 * Weighting by 1 / pixel_value^2 is empirical.
 * It was found to work well, while weighting by 1/pixel_value gives sometimes
 * worse results.
 * The idea of using a weighted averaging here comes from:
 * C. N. Ochotorena and Y. Yamashita, "Anisotropic Guided Filtering," in IEEE Transactions on Image Processing, vol. 29, pp. 1397-1412, 2020, doi: 10.1109/TIP.2019.2941326.
 * although we don't use the same weighting system (their weight is based
 * on variance, ours is based on the pixel value).
 * We tried a mixed weight using their weight in combination to ours, but
 * it did not improve the result.
 *
 * The implementation EIGF uses downscaling to speed-up the filtering,
 * just like what is done in fast_guided_filter.h
**/
