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

#pragma once

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

static inline void exposure_independent_guided_filter(const float *const restrict guide, // I
                                    const float *const restrict mask, //p
                                    float *const restrict ab,
                                    const size_t width, const size_t height,
                                    const float sigma, const float feathering)
{
  // We also use gaussian blurs instead of the square blurs of the guided filter
  const size_t Ndim = width * height;
  float *const restrict blurred_guide = dt_alloc_sse_ps(Ndim);
  // guide_x_guide = (guide - blurred_guide)^2
  float *const restrict guide_x_guide = dt_alloc_sse_ps(Ndim);
  // guide_variance = blur(guide_x_guide)
  float *const restrict guide_variance = dt_alloc_sse_ps(Ndim);
  float *const restrict guide_x_mask = dt_alloc_sse_ps(Ndim);
  // guide_mask_covariance = blur(guide_x_mask)
  float *const restrict guide_mask_covariance = dt_alloc_sse_ps(Ndim);
  float *const restrict blurred_mask = dt_alloc_sse_ps(Ndim);
  float *const restrict a = dt_alloc_sse_ps(Ndim);
  float *const restrict b = dt_alloc_sse_ps(Ndim);
  // weight to compute the weighted blur of a and b
  float *const restrict weights = dt_alloc_sse_ps(Ndim);
  float *const restrict blurred_a = dt_alloc_sse_ps(Ndim);
  float *const restrict blurred_b = dt_alloc_sse_ps(Ndim);
  float *const restrict blurred_weights = dt_alloc_sse_ps(Ndim);

  float ming = 10000000.0f;
  float maxg = 0.0f;
  float minm = 10000000.0f;
  float maxm = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(guide, mask, Ndim) \
  schedule(simd:static) aligned(guide, mask:64) \
  reduction(max:maxg, maxm)\
  reduction(min:ming, minm)
#endif
  for(size_t k = 0; k < Ndim; k++)
  {
    const float pixelg = guide[k];
    const float pixelm = mask[k];
    if(pixelg < ming) ming = pixelg;
    if(pixelg > maxg) maxg = pixelg;
    if(pixelm < minm) minm = pixelm;
    if(pixelm > maxm) maxm = pixelm;
  }

  dt_gaussian_t *g = dt_gaussian_init(width, height, 1, &maxg, &ming, sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, guide, blurred_guide);
  dt_gaussian_free(g);

  g = dt_gaussian_init(width, height, 1, &maxm, &minm, sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, mask, blurred_mask);
  dt_gaussian_free(g);

  float mingg = 10000000.0f;
  float maxgg = 0.0f;
  float mingm = 10000000.0f;
  float maxgm = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
  dt_omp_firstprivate(guide, mask, blurred_guide, blurred_mask, guide_x_guide, guide_x_mask, Ndim) \
  schedule(simd:static) aligned(guide, mask, blurred_guide, blurred_mask, guide_x_guide, guide_x_mask:64) \
  reduction(max:maxgg, maxgm)\
  reduction(min:mingg, mingm)
#endif
  for(size_t k = 0; k < Ndim; k++)
  {
    const float deviation = guide[k] - blurred_guide[k];
    const float squared_deviation = deviation * deviation;
    guide_x_guide[k] = squared_deviation;
    if(squared_deviation < mingg) mingg = squared_deviation;
    if(squared_deviation > maxgg) maxgg = squared_deviation;
    const float cov = (guide[k] - blurred_guide[k]) * (mask[k] - blurred_mask[k]);
    guide_x_mask[k] = cov;
    if(cov < mingm) mingm = cov;
    if(cov > maxgm) maxgm = cov;
  }

  g = dt_gaussian_init(width, height, 1, &maxgg, &mingg, sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, guide_x_guide, guide_variance);
  dt_gaussian_free(g);

  g = dt_gaussian_init(width, height, 1, &maxgm, &mingm, sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, guide_x_mask, guide_mask_covariance);
  dt_gaussian_free(g);

  float mina = 10000000.0f;
  float minb = 10000000.0f;
  float minw = 10000000.0f;
  float maxa = 0.0f;
  float maxb = 0.0f;
  float maxw = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
  dt_omp_firstprivate(guide, mask, blurred_guide, guide_variance, blurred_mask, guide_mask_covariance, a, b, weights, Ndim, feathering) \
  schedule(simd:static) aligned(guide, mask, blurred_guide, guide_variance, blurred_mask, guide_mask_covariance, a, b, weights:64) \
  reduction(max:maxa, maxb, maxw)\
  reduction(min:mina, minb, minw)
#endif
  for(size_t k = 0; k < Ndim; k++)
  {
    const float pixelg = fmaxf(guide[k], 1E-6);
    const float pixelm = fmaxf(mask[k], 1E-6);
    const float normalized_var_guide = guide_variance[k] / (pixelg * pixelg);
    const float normalized_covar = guide_mask_covariance[k] / (pixelg * pixelm);
    // empirical weight
    float w = 1.f / pixelg;
    w *= w;
    a[k] = w * normalized_covar / (normalized_var_guide + feathering);
    b[k] = w * blurred_mask[k] - a[k] * blurred_guide[k];
    weights[k] = w;
    if(a[k] < mina) mina = a[k];
    if(b[k] < minb) minb = b[k];
    if(w < minw) minw = w;
    if(a[k] > maxa) maxa = a[k];
    if(b[k] > maxb) maxb = b[k];
    if(w > maxw) maxw = w;
  }

  g = dt_gaussian_init(width, height, 1, &maxa, &mina, sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, a, blurred_a);
  dt_gaussian_free(g);

  g = dt_gaussian_init(width, height, 1, &maxb, &minb, sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, b, blurred_b);
  dt_gaussian_free(g);

  g = dt_gaussian_init(width, height, 1, &maxw, &minw, sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, weights, blurred_weights);
  dt_gaussian_free(g);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
  dt_omp_firstprivate(ab, blurred_a, blurred_b, blurred_weights, Ndim) \
  schedule(simd:static) aligned(ab, blurred_a, blurred_b, blurred_weights:64)
#endif
  for(size_t k = 0; k < Ndim; k++)
  {
    const float normalize = blurred_weights[k];
    ab[2 * k] = blurred_a[k] / normalize;
    ab[2 * k + 1] = blurred_b[k] / normalize;
  }

  dt_free_align(blurred_guide);
  dt_free_align(guide_x_guide);
  dt_free_align(guide_variance);
  dt_free_align(guide_x_mask);
  dt_free_align(guide_mask_covariance);
  dt_free_align(blurred_mask);
  dt_free_align(a);
  dt_free_align(b);
  dt_free_align(weights);
  dt_free_align(blurred_a);
  dt_free_align(blurred_b);
  dt_free_align(blurred_weights);
}

__DT_CLONE_TARGETS__
static inline void fast_eigf_surface_blur(float *const restrict image,
                                      const size_t width, const size_t height,
                                      const int radius, float feathering, const int iterations,
                                      const dt_iop_guided_filter_blending_t filter, const float scale,
                                      const float quantization, const float quantize_min, const float quantize_max)
{
  // Works in-place on a grey image
  // mostly similar with fast_surface_blur from fast_guided_filter.h

  // A down-scaling of 4 seems empirically safe and consistent no matter the image zoom level
  // see reference paper above for proof.
  const float scaling = 4.0f;
  const float ds_sigma = fmaxf((float)radius / scaling / 2.0f, 1.0f);

  const size_t ds_height = height / scaling;
  const size_t ds_width = width / scaling;

  const size_t num_elem_ds = ds_width * ds_height;
  const size_t num_elem = width * height;

  float *const restrict ds_image = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds));
  float *const restrict ds_mask = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds));
  float *const restrict ds_ab = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds * 2));
  float *const restrict ab = dt_alloc_sse_ps(dt_round_size_sse(num_elem * 2));

  if(!ds_image || !ds_mask || !ds_ab || !ab)
  {
    dt_control_log(_("fast exposure independent guided filter failed to allocate memory, check your RAM settings"));
    goto clean;
  }

  // Downsample the image for speed-up
  interpolate_bilinear(image, width, height, ds_image, ds_width, ds_height, 1);

  // empirical formula to have consistent smoothing when increasing the radius
  const float adapted_feathering = feathering * radius * sqrt(radius) / 40.0f;
  // Iterations of filter models the diffusion, sort of
  for(int i = 0; i < iterations; i++)
  {
    // (Re)build the mask from the quantized image to help guiding
    quantize(ds_image, ds_mask, ds_width * ds_height, quantization, quantize_min, quantize_max);
    exposure_independent_guided_filter(ds_mask, ds_image, ds_ab, ds_width, ds_height, ds_sigma, adapted_feathering);

    if(i != iterations - 1)
    {
      // Process the intermediate filtered image
      apply_linear_blending(ds_image, ds_ab, num_elem_ds);
    }
  }

  // Upsample the blending parameters a and b
  interpolate_bilinear(ds_ab, ds_width, ds_height, ab, width, height, 2);

  // Finally, blend the guided image
  if(filter == DT_GF_BLENDING_LINEAR)
    apply_linear_blending(image, ab, num_elem);
  else if(filter == DT_GF_BLENDING_GEOMEAN)
    apply_linear_blending_w_geomean(image, ab, num_elem);

clean:
  if(ab) dt_free_align(ab);
  if(ds_ab) dt_free_align(ds_ab);
  if(ds_mask) dt_free_align(ds_mask);
  if(ds_image) dt_free_align(ds_image);
}
