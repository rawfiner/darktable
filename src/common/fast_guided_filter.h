/*
    This file is part of darktable,
    copyright (c) 2019 Aurélien Pierre.

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

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "common/darktable.h"
#include "common/gaussian.h"

/** Note :
 * we use finite-math-only and fast-math because divisions by zero are manually avoided in the code
 * fp-contract=fast enables hardware-accelerated Fused Multiply-Add
 * the rest is loop reorganization and vectorization optimization
 **/

#if defined(__GNUC__)
#pragma GCC optimize ("unroll-loops", "tree-loop-if-convert", \
                      "tree-loop-distribution", "no-strict-aliasing", \
                      "loop-interchange", "loop-nest-optimize", "tree-loop-im", \
                      "unswitch-loops", "tree-loop-ivcanon", "ira-loop-pressure", \
                      "split-ivs-in-unroller", "variable-expansion-in-unroller", \
                      "split-loops", "ivopts", "predictive-commoning",\
                      "tree-loop-linear", "loop-block", "loop-strip-mine", \
                      "finite-math-only", "fp-contract=fast", "fast-math")
#endif

#define MIN_FLOAT exp2f(-16.0f)


typedef enum dt_iop_guided_filter_blending_t
{
  DT_GF_BLENDING_LINEAR = 0,
  DT_GF_BLENDING_GEOMEAN
} dt_iop_guided_filter_blending_t;


/***
 * DOCUMENTATION
 *
 * Fast Iterative Guided filter for surface blur
 *
 * This is a fast vectorized implementation of guided filter for grey images optimized for
 * the special case where the guiding and the guided image are the same, which is useful
 * for edge-aware surface blur.
 *
 * Since the guided filter is a linear application, we can safely downscale
 * the guiding and the guided image by a factor of 4, using a bilinear interpolation,
 * compute the guidance at this scale, then upscale back to the original size
 * and get a free 10× speed-up.
 *
 * Then, the vectorization adds another substantial speed-up. Overall, it brings a ×50 to ×200
 * speed-up compared to the guided_filter.h lib. Of course, it requires every buffer to be
 * 64-bits aligned.
 *
 * On top of the default guided filter, several pre- and post-processing options are provided :
 *
 *  - mask quantization : perform a posterization of the guiding image in log2 space to
 *    help the guiding to produce smoother areas,
 *
 *  - blending : perform a regular (linear) blending of a and b parameters after the
 *    variance analysis (aka the by-the-book guided filter), or a geometric mean of the filter output (by-the-book)
 *    and the original image, which produces a pleasing trade-off.
 *
 *  - iterations : apply the guided filtering recursively, with kernel size increasing by sqrt(2)
 *    between each iteration, to diffuse the filter and soften edges transitions.
 *
 * Reference : 
 *  Kaiming He, Jian Sun, Microsoft : https://arxiv.org/abs/1505.00996
 **/


 #ifdef _OPENMP
#pragma omp declare simd
#endif
__DT_CLONE_TARGETS__
static float fast_clamp(const float value, const float bottom, const float top)
{
  // vectorizable clamping between bottom and top values
  return fmax(fmin(value, top), bottom);
}


__DT_CLONE_TARGETS__
static inline void interpolate_bilinear(const float *const restrict in, const size_t width_in, const size_t height_in,
                                        float *const restrict out, const size_t width_out, const size_t height_out,
                                        const size_t ch)
{
  // Fast vectorized bilinear interpolation on ch channels
#ifdef _OPENMP
#pragma omp parallel for simd collapse(2) default(none) \
  schedule(simd:static) aligned(in, out:64) \
  dt_omp_firstprivate(in, out, width_out, height_out, width_in, height_in, ch)
#endif
  for(size_t i = 0; i < height_out; i++)
  {
    for(size_t j = 0; j < width_out; j++)
    {
      // Relative coordinates of the pixel in output space
      const float x_out = (float)j /(float)width_out;
      const float y_out = (float)i /(float)height_out;

      // Corresponding absolute coordinates of the pixel in input space
      const float x_in = x_out * (float)width_in;
      const float y_in = y_out * (float)height_in;

      // Nearest neighbours coordinates in input space
      size_t x_prev = (size_t)floorf(x_in);
      size_t x_next = x_prev + 1;
      size_t y_prev = (size_t)floorf(y_in);
      size_t y_next = y_prev + 1;

      x_prev = (x_prev < width_in) ? x_prev : width_in - 1;
      x_next = (x_next < width_in) ? x_next : width_in - 1;
      y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
      y_next = (y_next < height_in) ? y_next : height_in - 1;

      // Nearest pixels in input array (nodes in grid)
      const size_t Y_prev = y_prev * width_in;
      const size_t Y_next =  y_next * width_in;
      const float *const Q_NW = (float *)in + (Y_prev + x_prev) * ch;
      const float *const Q_NE = (float *)in + (Y_prev + x_next) * ch;
      const float *const Q_SE = (float *)in + (Y_next + x_next) * ch;
      const float *const Q_SW = (float *)in + (Y_next + x_prev) * ch;

      // Spatial differences between nodes
      const float Dy_next = (float)y_next - y_in;
      const float Dy_prev = 1.f - Dy_next; // because next - prev = 1
      const float Dx_next = (float)x_next - x_in;
      const float Dx_prev = 1.f - Dx_next; // because next - prev = 1

      // Interpolate over ch layers
      float *const pixel_out = (float *)out + (i * width_out + j) * ch;

#pragma unroll
      for(size_t c = 0; c < ch; c++)
      {
        pixel_out[c] = Dy_prev * (Q_SW[c] * Dx_next + Q_SE[c] * Dx_prev) +
                       Dy_next * (Q_NW[c] * Dx_next + Q_NE[c] * Dx_prev);
      }
    }
  }
}

static inline void box_average(float *const restrict in,
                               const size_t width, const size_t height, const size_t ch,
                               const int radius);

__DT_CLONE_TARGETS__
static inline void variance_and_avg(const float *const restrict in,
                                   float *const restrict avg,
                                   float *const restrict var,
                                   const size_t width, const size_t height,
                                   const int radius)
{
 const size_t Ndim = width * height;
 const size_t Ndimch = width * height * 4;

 float *const restrict temp = dt_alloc_sse_ps(Ndimch); // array of structs { { mean_I, mean_p, corr_I, corr_Ip } }
 float *const restrict guide_x_mask = dt_alloc_sse_ps(Ndim);
 float *const restrict guide_x_guide = dt_alloc_sse_ps(Ndim);

 // Pre-multiply in with itself
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
 dt_omp_firstprivate(in, guide_x_mask, guide_x_guide, Ndim, radius) \
 schedule(simd:static) aligned(in, guide_x_mask, guide_x_guide:64)
#endif
 for(size_t k = 0; k < Ndim; k++)
 {
   guide_x_mask[k] = in[k] * in[k];
   guide_x_guide[k] = in[k] * in[k];
 }

 // Convolve box average along columns
#ifdef _OPENMP
#pragma omp parallel for default(none) \
 dt_omp_firstprivate(in, temp, guide_x_mask, guide_x_guide, width, height, radius) \
 schedule(simd:static) collapse(2)
#endif
 for(size_t i = 0; i < height; i++)
 {
   for(size_t j = 0; j < width; j++)
   {
     const size_t begin_convol = (i < radius) ? 0 : i - radius;
     size_t end_convol = i + radius;
     end_convol = (end_convol < height) ? end_convol : height - 1;
     const float num_elem = 1.0f / ((float)end_convol - (float)begin_convol + 1.0f);
     double tmp[4] DT_ALIGNED_PIXEL = { 0.0f }; // = { w_mean_I, w_mean_p, w_corr_I, w_corr_Ip }

#ifdef _OPENMP
#pragma omp simd reduction(+:tmp) aligned(tmp:16) aligned(in, guide_x_mask, guide_x_guide:64)
#endif
     for(size_t c = begin_convol; c <= end_convol; c++)
     {
       const size_t index = c * width + j;
       tmp[0] += in[index];
       tmp[1] += in[index];
       tmp[2] += guide_x_guide[index];
       tmp[3] += guide_x_mask[index];
     }

     const size_t index = (i * width + j) * 4;

#ifdef _OPENMP
#pragma omp simd aligned(tmp:16) aligned(temp:64)
#endif
     for(size_t c = 0; c < 4; c++)
       temp[index + c] = tmp[c] * num_elem;
   }
 }

 if(guide_x_guide != NULL) dt_free_align(guide_x_guide);
 if(guide_x_mask != NULL) dt_free_align(guide_x_mask);

 // Convolve box average along rows and output result
#ifdef _OPENMP
#pragma omp parallel for default(none) \
 dt_omp_firstprivate(var, avg, temp, width, height, radius) \
 schedule(simd:static) collapse(2)
#endif
 for(size_t i = 0; i < height; i++)
 {
   for(size_t j = 0; j < width; j++)
   {
     const size_t begin_convol = (j < radius) ? 0 : j - radius;
     size_t end_convol = j + radius;
     end_convol = (end_convol < width) ? end_convol : width - 1;
     const float num_elem = 1.0f / ((float)end_convol - (float)begin_convol + 1.0f);
     float tmp[4] DT_ALIGNED_PIXEL = { 0.0f }; // = { w_mean_I, w_mean_p, w_corr_I, w_corr_Ip }

     for(size_t c = begin_convol; c <= end_convol; c++)
     {
       const size_t index = (i * width + c) * 4;
#ifdef _OPENMP
#pragma omp simd aligned(temp:64) aligned(tmp:16) reduction(+:tmp)
#endif
       for(size_t k = 0; k < 4; ++k)
         tmp[k] += temp[index + k];
     }

#ifdef _OPENMP
#pragma omp simd aligned(tmp:16) reduction(*:tmp)
#endif
     for(size_t c = 0; c < 4; c++)
       tmp[c] *= num_elem;

     const size_t idx = (i * width + j);
     avg[idx] = tmp[0];
     var[idx] = (tmp[2] - tmp[0] * tmp[0]);
   }
 }

 if(temp != NULL) dt_free_align(temp);
}


__DT_CLONE_TARGETS__
static inline void variance_analyse(const float *const restrict guide, // I
                                    const float *const restrict mask, //p
                                    float *const restrict ab,
                                    const size_t width, const size_t height,
                                    const int radius, const float feathering,
                                    const float *const restrict exp_blur)
{
  // Compute a box average (filter) on a grey image over a window of size 2*radius + 1
  // then get the variance of the guide and covariance with its mask
  // output a and b, the linear blending params
  // p, the mask is the quantised guide I


  //
  // float *const restrict local_avg = dt_alloc_sse_ps(dt_round_size_sse(width * height * sizeof(float)));
  // float *const restrict local_var = dt_alloc_sse_ps(dt_round_size_sse(width * height * sizeof(float)));
  // variance_and_avg(guide_blurred, local_avg, local_var, width, height, 3);

  const size_t Ndim = width * height;
  const size_t Ndimch = width * height * 4;

  float *const restrict temp = dt_alloc_sse_ps(Ndimch); // array of structs { { mean_I, mean_p, corr_I, corr_Ip } }
  float *const restrict guide_x_mask = dt_alloc_sse_ps(Ndim);
  float *const restrict guide_x_guide = dt_alloc_sse_ps(Ndim);

  // Pre-multiply guide and mask
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
  dt_omp_firstprivate(guide, mask, guide_x_mask, guide_x_guide, Ndim, radius) \
  schedule(simd:static) aligned(guide, mask, guide_x_mask, guide_x_guide:64)
#endif
  for(size_t k = 0; k < Ndim; k++)
  {
    guide_x_mask[k] = guide[k] * mask[k];
    guide_x_guide[k] = guide[k] * guide[k];
  }

  // Convolve box average along columns
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(guide, mask, temp, guide_x_mask, guide_x_guide, width, height, radius) \
  schedule(simd:static) collapse(2)
#endif
  for(size_t i = 0; i < height; i++)
  {
    for(size_t j = 0; j < width; j++)
    {
      const size_t begin_convol = (i < radius) ? 0 : i - radius;
      size_t end_convol = i + radius;
      end_convol = (end_convol < height) ? end_convol : height - 1;
      const float num_elem = 1.0f / ((float)end_convol - (float)begin_convol + 1.0f);
      double tmp[4] DT_ALIGNED_PIXEL = { 0.0f }; // = { w_mean_I, w_mean_p, w_corr_I, w_corr_Ip }

#ifdef _OPENMP
#pragma omp simd reduction(+:tmp) aligned(tmp:16) aligned(guide, mask, guide_x_mask, guide_x_guide:64)
#endif
      for(size_t c = begin_convol; c <= end_convol; c++)
      {
        const size_t index = c * width + j;
        tmp[0] += guide[index];
        tmp[1] += mask[index];
        tmp[2] += guide_x_guide[index];
        tmp[3] += guide_x_mask[index];
      }

      const size_t index = (i * width + j) * 4;

#ifdef _OPENMP
#pragma omp simd aligned(tmp:16) aligned(temp:64)
#endif
      for(size_t c = 0; c < 4; c++)
        temp[index + c] = tmp[c] * num_elem;
    }
  }

  if(guide_x_guide != NULL) dt_free_align(guide_x_guide);
  if(guide_x_mask != NULL) dt_free_align(guide_x_mask);

  // Convolve box average along rows and output result
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(ab, temp, width, height, radius, feathering, mask, exp_blur) \
  schedule(simd:static) collapse(2)
#endif
  for(size_t i = 0; i < height; i++)
  {
    for(size_t j = 0; j < width; j++)
    {
      const size_t begin_convol = (j < radius) ? 0 : j - radius;
      size_t end_convol = j + radius;
      end_convol = (end_convol < width) ? end_convol : width - 1;
      const float num_elem = 1.0f / ((float)end_convol - (float)begin_convol + 1.0f);
      float tmp[4] DT_ALIGNED_PIXEL = { 0.0f }; // = { w_mean_I, w_mean_p, w_corr_I, w_corr_Ip }

      for(size_t c = begin_convol; c <= end_convol; c++)
      {
        const size_t index = (i * width + c) * 4;
#ifdef _OPENMP
#pragma omp simd aligned(temp:64) aligned(tmp:16) reduction(+:tmp)
#endif
        for(size_t k = 0; k < 4; ++k)
          tmp[k] += temp[index + k];
      }

#ifdef _OPENMP
#pragma omp simd aligned(tmp:16) reduction(*:tmp)
#endif
      for(size_t c = 0; c < 4; c++)
        tmp[c] *= num_elem;

      const size_t index = (i * width + j) * 2;
      const size_t idx = (i * width + j);
      float var = (tmp[2] - tmp[0] * tmp[0]);
      // construct mean as the average of several means of different radius
      float mean = exp_blur[idx];
      const float d = fmaxf(var + mean * mean * feathering, 1e-15f); // avoid division by 0.
      //const float d = fmaxf((tmp[2] - tmp[0] * tmp[0]) + (0.9f * mask[idx] * mask[idx] + 0.1f * tmp[0] * tmp[0]) * feathering, 1e-15f); // avoid division by 0.
      const float a = (tmp[3] - tmp[0] * tmp[1]) / d;
      const float b = tmp[1] - a * tmp[0];
      const float ab_temp[2] DT_ALIGNED_PIXEL = { a, b };

#ifdef _OPENMP
#pragma omp simd aligned(ab_temp:16) aligned(ab:64)
#endif
      for(size_t c = 0; c < 2; c++)
        ab[index + c] = ab_temp[c];
    }
  }

  if(temp != NULL) dt_free_align(temp);
}


__DT_CLONE_TARGETS__
static inline void box_average(float *const restrict in,
                               const size_t width, const size_t height, const size_t ch,
                               const int radius)
{
  // Compute in-place a box average (filter) on a multi-channel image over a window of size 2*radius + 1
  // We make use of the separable nature of the filter kernel to speed-up the computation
  // by convolving along columns and rows separately (complexity O(2 × radius) instead of O(radius²)).

  assert(ch <= 4);

  const size_t Ndim = width * height * ch;
  float *const restrict temp = dt_alloc_sse_ps(Ndim);

  // Convolve box average along columns
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, temp, width, height, ch, radius) \
  schedule(simd:static) collapse(2)
#endif
  for(size_t j = 0; j < width; j++)
  {
    for(size_t i = 0; i < height; i++)
    {
      const size_t begin_convol = (i < radius) ? 0 : i - radius;
      size_t end_convol = i + radius;
      end_convol = (end_convol < height) ? end_convol : height - 1;
      const float num_elem = (float)end_convol - (float)begin_convol + 1.0f;
      const size_t index = (i * width + j) * ch;

      float w[4] DT_ALIGNED_PIXEL = { 0.0f };

      // Convolve
      for(size_t c = begin_convol; c <= end_convol; c++)
      {
        const size_t index_c = (c * width + j) * ch;
#ifdef _OPENMP
#pragma omp simd aligned(in:64) aligned(w:16) reduction(+:w)
#endif
        for(size_t k = 0; k < ch; ++k)
          w[k] += in[index_c + k];
      }

    // Normalize and Save
#ifdef _OPENMP
#pragma omp simd aligned(temp:64) aligned(w:16)
#endif
      for(size_t k = 0; k < ch; ++k)
        temp[index + k] = w[k] / num_elem;
    }
  }

  // Convolve box average along rows and output result
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, temp, width, height, ch, radius) \
  schedule(simd:static) collapse(2)
#endif
  for(size_t i = 0; i < height; i++)
  {
    for(size_t j = 0; j < width; j++)
    {
      const size_t begin_convol = (j < radius) ? 0 : j - radius;
      size_t end_convol = j + radius;
      end_convol = (end_convol < width) ? end_convol : width - 1;
      const float num_elem = (float)end_convol - (float)begin_convol + 1.0f;
      const size_t stride = i * width;
      const size_t index = (stride + j) * ch;

      float w[4] DT_ALIGNED_PIXEL = { 0.0f };

      // Convolve
      for(size_t c = begin_convol; c <= end_convol; c++)
      {
        const size_t index_c = (stride + c) * ch;
#ifdef _OPENMP
#pragma omp simd aligned(temp:64) aligned(w:16) reduction(+:w)
#endif
        for(size_t k = 0; k < ch; ++k)
          w[k] += temp[index_c + k];
      }

      // Normalize and Save
#ifdef _OPENMP
#pragma omp simd aligned(w:16) aligned(in:64)
#endif
      for(size_t k = 0; k < ch; ++k)
        in[index + k] = w[k] / num_elem;
    }
  }

  if(temp != NULL) dt_free_align(temp);
}

__DT_CLONE_TARGETS__
static inline void apply_linear_blending(float *const restrict image,
                                         const float *const restrict ab,
                                         const size_t num_elem)
{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, ab, num_elem) \
schedule(simd:static) aligned(image, ab:64)
#endif
  for(size_t k = 0; k < num_elem; k++)
  {
    // Note : image[k] is positive at the outside of the luminance mask
    image[k] = fmaxf(image[k] * ab[k * 2] + ab[k * 2 + 1], MIN_FLOAT);
  }
}


__DT_CLONE_TARGETS__
static inline void apply_linear_blending_w_geomean(float *const restrict image,
                                                   const float *const restrict ab,
                                                   const size_t num_elem)
{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, ab, num_elem) \
schedule(simd:static) aligned(image, ab:64)
#endif
  for(size_t k = 0; k < num_elem; k++)
  {
    // Note : image[k] is positive at the outside of the luminance mask
    image[k] = sqrtf(image[k] * fmaxf(image[k] * ab[k * 2] + ab[k * 2 + 1], MIN_FLOAT));
  }
}


__DT_CLONE_TARGETS__
static inline void quantize(const float *const restrict image,
                            float *const restrict out,
                            const size_t num_elem,
                            const float sampling, const float clip_min, const float clip_max)
{
  // Quantize in exposure levels evenly spaced in log by sampling

  if(sampling == 0.0f)
  {
    // No-op
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, out, num_elem, sampling, clip_min, clip_max) \
schedule(simd:static) aligned(image, out:64)
#endif
    for(size_t k = 0; k < num_elem; k++)
      out[k] = image[k];
  }
  else if(sampling == 1.0f)
  {
    // fast track
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, out, num_elem, sampling, clip_min, clip_max) \
schedule(simd:static) aligned(image, out:64)
#endif
    for(size_t k = 0; k < num_elem; k++)
      out[k] = fast_clamp(exp2f(floorf(log2f(image[k]))), clip_min, clip_max);
  }

  else
  {
    // slow track
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, out, num_elem, sampling, clip_min, clip_max) \
schedule(simd:static) aligned(image, out:64)
#endif
    for(size_t k = 0; k < num_elem; k++)
      out[k] = fast_clamp(exp2f(floorf(log2f(image[k]) / sampling) * sampling), clip_min, clip_max);
  }
}

static void interpolate_with_affinity(float *const restrict ds_ab, float *const restrict ds_image,
                              const size_t ds_width, const size_t ds_height, float *const restrict ab,
                              float *const restrict image, const size_t width, const size_t height)
{
#ifdef _OPENMP
#pragma omp parallel for default(none) \
dt_omp_firstprivate(ds_ab, ds_image, ds_width, ds_height, ab, image, width, height) \
schedule(static)
#endif
  for(int j = 0; j < height; j++)
  {
    for(int i = 0; i < width; i++)
    {
      // find affinity using ds_image and image
      float current = image[j * width + i];
      int j_ds0 = MIN(j >> 1, ds_height - 1);
      int i_ds0 = MIN(i >> 1, ds_width - 1);
      int j_ds1, i_ds1, j_ds2, i_ds2, j_ds3, i_ds3, j_ds4, i_ds4,
          j_ds5, i_ds5, j_ds6, i_ds6, j_ds7, i_ds7, j_ds8, i_ds8;
      // if(((j & 1) == 0) && ((i & 1) == 0))
      // {
      //   // top left corner
        j_ds1 = MIN(MAX(j_ds0, 0), ds_height - 1);
        i_ds1 = MIN(MAX(i_ds0 - 1, 0), ds_width - 1);
        j_ds2 = MIN(MAX(j_ds0 - 1, 0), ds_height - 1);
        i_ds2 = MIN(MAX(i_ds0 - 1, 0), ds_width - 1);
        j_ds3 = MIN(MAX(j_ds0 - 1, 0), ds_height - 1);
        i_ds3 = MIN(MAX(i_ds0, 0), ds_width - 1);
        j_ds4 = MIN(MAX(j_ds0, 0), ds_height - 1);
        i_ds4 = MIN(MAX(i_ds0 + 1, 0), ds_width - 1);
        j_ds5 = MIN(MAX(j_ds0 + 1, 0), ds_height - 1);
        i_ds5 = MIN(MAX(i_ds0 + 1, 0), ds_width - 1);
        j_ds6 = MIN(MAX(j_ds0 + 1, 0), ds_height - 1);
        i_ds6 = MIN(MAX(i_ds0, 0), ds_width - 1);
        j_ds7 = MIN(MAX(j_ds0 - 1, 0), ds_height - 1);
        i_ds7 = MIN(MAX(i_ds0 + 1, 0), ds_width - 1);
        j_ds8 = MIN(MAX(j_ds0 + 1, 0), ds_height - 1);
        i_ds8 = MIN(MAX(i_ds0 - 1, 0), ds_width - 1);
        #if 0
      }
      else if(((j & 1) == 0) && ((i & 1) == 1))
      {
        // top right corner
        j_ds1 = MIN(MAX(j_ds0, 0), ds_height - 1);
        i_ds1 = MIN(MAX(i_ds0 + 1, 0), ds_width - 1);
        j_ds2 = MIN(MAX(j_ds0 - 1, 0), ds_height - 1);
        i_ds2 = MIN(MAX(i_ds0 + 1, 0), ds_width - 1);
        j_ds3 = MIN(MAX(j_ds0 - 1, 0), ds_height - 1);
        i_ds3 = MIN(MAX(i_ds0, 0), ds_width - 1);
        j_ds4 = MIN(MAX(j_ds0, 0), ds_height - 1);
        i_ds4 = MIN(MAX(i_ds0 + 2, 0), ds_width - 1);
        j_ds5 = MIN(MAX(j_ds0 - 2, 0), ds_height - 1);
        i_ds5 = MIN(MAX(i_ds0 + 2, 0), ds_width - 1);
        j_ds6 = MIN(MAX(j_ds0 - 2, 0), ds_height - 1);
        i_ds6 = MIN(MAX(i_ds0, 0), ds_width - 1);
        j_ds7 = MIN(MAX(j_ds0 - 1, 0), ds_height - 1);
        i_ds7 = MIN(MAX(i_ds0 + 2, 0), ds_width - 1);
        j_ds8 = MIN(MAX(j_ds0 - 2, 0), ds_height - 1);
        i_ds8 = MIN(MAX(i_ds0 + 1, 0), ds_width - 1);
      }
      else if(((j & 1) == 1) && ((i & 1) == 0))
      {
        // bottom left corner
        j_ds1 = MIN(MAX(j_ds0, 0), ds_height - 1);
        i_ds1 = MIN(MAX(i_ds0 - 1, 0), ds_width - 1);
        j_ds2 = MIN(MAX(j_ds0 + 1, 0), ds_height - 1);
        i_ds2 = MIN(MAX(i_ds0 - 1, 0), ds_width - 1);
        j_ds3 = MIN(MAX(j_ds0 + 1, 0), ds_height - 1);
        i_ds3 = MIN(MAX(i_ds0, 0), ds_width - 1);
        j_ds4 = MIN(MAX(j_ds0, 0), ds_height - 1);
        i_ds4 = MIN(MAX(i_ds0 - 2, 0), ds_width - 1);
        j_ds5 = MIN(MAX(j_ds0 + 2, 0), ds_height - 1);
        i_ds5 = MIN(MAX(i_ds0 - 2, 0), ds_width - 1);
        j_ds6 = MIN(MAX(j_ds0 + 2, 0), ds_height - 1);
        i_ds6 = MIN(MAX(i_ds0, 0), ds_width - 1);
        j_ds7 = MIN(MAX(j_ds0 + 1, 0), ds_height - 1);
        i_ds7 = MIN(MAX(i_ds0 - 2, 0), ds_width - 1);
        j_ds8 = MIN(MAX(j_ds0 + 2, 0), ds_height - 1);
        i_ds8 = MIN(MAX(i_ds0 - 1, 0), ds_width - 1);
      }
      else //((j & 1) == 1) && ((i & 1) == 1)
      {
        // bottom right corner
        j_ds1 = MIN(MAX(j_ds0, 0), ds_height - 1);
        i_ds1 = MIN(MAX(i_ds0 + 1, 0), ds_width - 1);
        j_ds2 = MIN(MAX(j_ds0 + 1, 0), ds_height - 1);
        i_ds2 = MIN(MAX(i_ds0 + 1, 0), ds_width - 1);
        j_ds3 = MIN(MAX(j_ds0 + 1, 0), ds_height - 1);
        i_ds3 = MIN(MAX(i_ds0, 0), ds_width - 1);
        j_ds4 = MIN(MAX(j_ds0, 0), ds_height - 1);
        i_ds4 = MIN(MAX(i_ds0 + 2, 0), ds_width - 1);
        j_ds5 = MIN(MAX(j_ds0 + 2, 0), ds_height - 1);
        i_ds5 = MIN(MAX(i_ds0 + 2, 0), ds_width - 1);
        j_ds6 = MIN(MAX(j_ds0 + 2, 0), ds_height - 1);
        i_ds6 = MIN(MAX(i_ds0, 0), ds_width - 1);
        j_ds7 = MIN(MAX(j_ds0 + 1, 0), ds_height - 1);
        i_ds7 = MIN(MAX(i_ds0 + 2, 0), ds_width - 1);
        j_ds8 = MIN(MAX(j_ds0 + 2, 0), ds_height - 1);
        i_ds8 = MIN(MAX(i_ds0 + 1, 0), ds_width - 1);
      }
      #endif
      float value_ds0 = ds_image[j_ds0 * ds_width + i_ds0];
      float value_ds1 = ds_image[j_ds1 * ds_width + i_ds1];
      float value_ds2 = ds_image[j_ds2 * ds_width + i_ds2];
      float value_ds3 = ds_image[j_ds3 * ds_width + i_ds3];
      float value_ds4 = ds_image[j_ds4 * ds_width + i_ds4];
      float value_ds5 = ds_image[j_ds5 * ds_width + i_ds5];
      float value_ds6 = ds_image[j_ds6 * ds_width + i_ds6];
      float value_ds7 = ds_image[j_ds7 * ds_width + i_ds7];
      float value_ds8 = ds_image[j_ds8 * ds_width + i_ds8];
      // find affinity
      float diff_ds0 = current / value_ds0;
      float diff_ds1 = current / value_ds1;
      float diff_ds2 = current / value_ds2;
      float diff_ds3 = current / value_ds3;
      float diff_ds4 = current / value_ds4;
      float diff_ds5 = current / value_ds5;
      float diff_ds6 = current / value_ds6;
      float diff_ds7 = current / value_ds7;
      float diff_ds8 = current / value_ds8;
      if(diff_ds0 < 1.0f) diff_ds0 = 1.0f / diff_ds0;
      if(diff_ds1 < 1.0f) diff_ds1 = 1.0f / diff_ds1;
      if(diff_ds2 < 1.0f) diff_ds2 = 1.0f / diff_ds2;
      if(diff_ds3 < 1.0f) diff_ds3 = 1.0f / diff_ds3;
      if(diff_ds4 < 1.0f) diff_ds4 = 1.0f / diff_ds4;
      if(diff_ds5 < 1.0f) diff_ds5 = 1.0f / diff_ds5;
      if(diff_ds6 < 1.0f) diff_ds6 = 1.0f / diff_ds6;
      if(diff_ds7 < 1.0f) diff_ds7 = 1.0f / diff_ds7;
      if(diff_ds8 < 1.0f) diff_ds8 = 1.0f / diff_ds8;
      float weight_ds0 = 1.0f / (diff_ds0 - 0.9999999f);
      float weight_ds1 = 1.0f / (diff_ds1 - 0.9999999f);
      float weight_ds2 = 1.0f / (diff_ds2 - 0.9999999f);
      float weight_ds3 = 1.0f / (diff_ds3 - 0.9999999f);
      float weight_ds4 = 1.0f / (diff_ds4 - 0.9999999f);
      float weight_ds5 = 1.0f / (diff_ds5 - 0.9999999f);
      float weight_ds6 = 1.0f / (diff_ds6 - 0.9999999f);
      float weight_ds7 = 1.0f / (diff_ds7 - 0.9999999f);
      float weight_ds8 = 1.0f / (diff_ds8 - 0.9999999f);
      float sum_weights = weight_ds0 + weight_ds1 + weight_ds2 + weight_ds3
                        + weight_ds4 + weight_ds5 + weight_ds6 + weight_ds7
                        + weight_ds8;
      float a_ds0 = ds_ab[(j_ds0 * ds_width + i_ds0) * 2];
      float a_ds1 = ds_ab[(j_ds1 * ds_width + i_ds1) * 2];
      float a_ds2 = ds_ab[(j_ds2 * ds_width + i_ds2) * 2];
      float a_ds3 = ds_ab[(j_ds3 * ds_width + i_ds3) * 2];
      float a_ds4 = ds_ab[(j_ds4 * ds_width + i_ds4) * 2];
      float a_ds5 = ds_ab[(j_ds5 * ds_width + i_ds5) * 2];
      float a_ds6 = ds_ab[(j_ds6 * ds_width + i_ds6) * 2];
      float a_ds7 = ds_ab[(j_ds7 * ds_width + i_ds7) * 2];
      float a_ds8 = ds_ab[(j_ds8 * ds_width + i_ds8) * 2];
      float b_ds0 = ds_ab[(j_ds0 * ds_width + i_ds0) * 2 + 1];
      float b_ds1 = ds_ab[(j_ds1 * ds_width + i_ds1) * 2 + 1];
      float b_ds2 = ds_ab[(j_ds2 * ds_width + i_ds2) * 2 + 1];
      float b_ds3 = ds_ab[(j_ds3 * ds_width + i_ds3) * 2 + 1];
      float b_ds4 = ds_ab[(j_ds4 * ds_width + i_ds4) * 2 + 1];
      float b_ds5 = ds_ab[(j_ds5 * ds_width + i_ds5) * 2 + 1];
      float b_ds6 = ds_ab[(j_ds6 * ds_width + i_ds6) * 2 + 1];
      float b_ds7 = ds_ab[(j_ds7 * ds_width + i_ds7) * 2 + 1];
      float b_ds8 = ds_ab[(j_ds8 * ds_width + i_ds8) * 2 + 1];
      weight_ds0 *= weight_ds0;
      weight_ds1 *= weight_ds1;
      weight_ds2 *= weight_ds2;
      weight_ds3 *= weight_ds3;
      weight_ds4 *= weight_ds4;
      weight_ds5 *= weight_ds5;
      weight_ds6 *= weight_ds6;
      weight_ds7 *= weight_ds7;
      weight_ds8 *= weight_ds8;
      sum_weights = weight_ds0 + weight_ds1 + weight_ds2 + weight_ds3
                        + weight_ds4 + weight_ds5 + weight_ds6 + weight_ds7
                        + weight_ds8;
      ab[(j * width + i) * 2] = (weight_ds0 * a_ds0 + weight_ds1 * a_ds1
                                + weight_ds2 * a_ds2 + weight_ds3 * a_ds3
                                + weight_ds4 * a_ds4 + weight_ds5 * a_ds5
                                + weight_ds6 * a_ds6 + weight_ds7 * a_ds7
                                + weight_ds8 * a_ds8) / sum_weights;
      ab[(j * width + i) * 2 + 1] = (weight_ds0 * b_ds0 + weight_ds1 * b_ds1
                                    + weight_ds2 * b_ds2 + weight_ds3 * b_ds3
                                    + weight_ds4 * b_ds4 + weight_ds5 * b_ds5
                                    + weight_ds6 * b_ds6 + weight_ds7 * b_ds7
                                    + weight_ds8 * b_ds8) / sum_weights;
    }
  }
}



__DT_CLONE_TARGETS__
static inline void fast_surface_blur(float *const restrict image,
                                      const size_t width, const size_t height,
                                      const int radius, float feathering, const int iterations,
                                      const dt_iop_guided_filter_blending_t filter, const float scale,
                                      const float quantization, const float quantize_min, const float quantize_max,
                                      const float sigma)
{
  // Works in-place on a grey image

  // A down-scaling of 4 seems empirically safe and consistent no matter the image zoom level
  // see reference paper above for proof.
  const float scaling = 4.0f;
  int ds_radius = (radius < 4) ? 1 : radius / scaling;

  const size_t ds_height = height / scaling * 2.0f;//FIXME ugly hack
  const size_t ds_width = width / scaling * 2.0f;
  const size_t ds_ds_height = height / scaling;
  const size_t ds_ds_width = width / scaling;

  const size_t num_elem_ds = ds_width * ds_height;
  const size_t num_elem_ds_ds = ds_ds_width * ds_ds_height;
  const size_t num_elem = width * height;

  float *const restrict exp_blur = dt_alloc_sse_ps(dt_round_size_sse(num_elem));
  float *const restrict ds_ds_exp_blur = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds_ds));
  float *const restrict ds_image = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds));
  float *const restrict ds_mask = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds));
  float *const restrict ds_ab = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds * 2));
  float *const restrict ds_ds_image = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds_ds));
  float *const restrict ds_ds_mask = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds_ds));
  float *const restrict ds_ds_ab = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds_ds * 2));
  float *const restrict ab = dt_alloc_sse_ps(dt_round_size_sse(num_elem * 2));

  if(!ds_image || !ds_mask || !ds_ab || !ab || !ds_ds_image || !ds_ds_mask || !ds_ds_ab || !exp_blur || !ds_ds_exp_blur)
  {
    dt_control_log(_("fast guided filter failed to allocate memory, check your RAM settings"));
    goto clean;
  }

  //const float sigma = 1.0f;
  const float min[] = {0.0f};
  const float max[] = {1.0f};
  dt_gaussian_t *g = dt_gaussian_init(width, height, 1, max, min, MAX(sigma, 0.00001f), 0);
  if(!g) return;
  dt_gaussian_blur(g, image, exp_blur);
  dt_gaussian_free(g);

  // Downsample the image for speed-up
  interpolate_bilinear(image, width, height, ds_image, ds_width, ds_height, 1);
  interpolate_bilinear(ds_image, ds_width, ds_height, ds_ds_image, ds_ds_width, ds_ds_height, 1);
  interpolate_bilinear(exp_blur, width, height, ds_ds_exp_blur, ds_ds_width, ds_ds_height, 1);

  // Iterations of filter models the diffusion, sort of
  for(int i = 0; i < iterations; ++i)
  {
    // (Re)build the mask from the quantized image to help guiding
    quantize(ds_ds_image, ds_ds_mask, ds_ds_width * ds_ds_height, quantization, quantize_min, quantize_max);

    // Perform the patch-wise variance analyse to get
    // the a and b parameters for the linear blending s.t. mask = a * I + b
    variance_analyse(ds_ds_mask, ds_ds_image, ds_ds_ab, ds_ds_width, ds_ds_height, ds_radius, feathering, ds_ds_exp_blur);

    // Compute the patch-wise average of parameters a and b
    float *const restrict ds_ds_ab_blurred = dt_alloc_sse_ps(dt_round_size_sse(num_elem_ds_ds * 2));
    memcpy(ds_ds_ab_blurred, ds_ds_ab, num_elem_ds_ds * 2 * sizeof(float));
    // box_average(ds_ds_ab_blurred, ds_ds_width, ds_ds_height, 2, 4);
    // for(int j = 0; j < num_elem_ds_ds * 2; j+=2)
    // {
    //   float weight_blur = 0.3f;
    //   ds_ds_ab[j] = 1.0f - powf(1.0f - ds_ds_ab_blurred[j], weight_blur) * powf(1.0f - ds_ds_ab[j], 1.0f - weight_blur);
    //   weight_blur = 0.7f;
    //   ds_ds_ab[j+1] = weight_blur * ds_ds_ab_blurred[j+1] + (1.0f - weight_blur) * ds_ds_ab[j+1];
    // }
    dt_free_align(ds_ds_ab_blurred);

    if(i != iterations - 1)
    {
      // Process the intermediate filtered image
      apply_linear_blending(ds_ds_image, ds_ds_ab, num_elem_ds_ds);
    }
  }

  // Upsample the blending parameters a and b
  //interpolate_bilinear(ds_ab, ds_width, ds_height, ab, width, height, 2);
  interpolate_with_affinity(ds_ds_ab, ds_ds_image, ds_ds_width, ds_ds_height, ds_ab, ds_image, ds_width, ds_height);
  interpolate_with_affinity(ds_ab, ds_image, ds_width, ds_height, ab, image, width, height);
  //interpolate_bilinear(ds_ds_ab, ds_ds_width, ds_ds_height, ab, width, height, 2);

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
  if(ds_ds_ab) dt_free_align(ds_ds_ab);
  if(ds_ds_mask) dt_free_align(ds_ds_mask);
  if(ds_ds_image) dt_free_align(ds_ds_image);
  if(exp_blur) dt_free_align(exp_blur);
  if(ds_ds_exp_blur) dt_free_align(ds_ds_exp_blur);
}
