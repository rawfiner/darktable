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
      // image and ds_image are used as guides to the interpolation of ab
      // for each pixel of destination image, we consider the pixel relative
      // position within the downscaled image:
      // each pixel of the high res image is one of 4 the pixels of the
      // high res image that melt into one pixel P of the downscaled image.
      // one pixel corresponds to the top left of P, one to the top right,
      // one to the bottom left, and one to the bottom right.
      // for the interpolation, we look for the pixels of the downscaled image
      // that are the most similar to the pixel of the high res image.
      // we only consider the pixels in a neighborhood which is approximately
      // centered at the position of the high res pixel within the downscaled
      // image.

      // if this is a pixel from the downscaled image:
      // _____
      // |    |
      // |____|
      // and this would be the same area of the image, but high res:
      // _____
      // |_|_|
      // |_|_|
      //
      // let's say we want to find the value of this pixel:
      // _____
      // |X|_|
      // |_|_|
      //
      // as this pixel is at top left of the low resolution pixel, we consider
      // the following search zone (ie a square slightly at the top and left
      // of the considered point) within the downscaled image:
      // _____________________
      // |    |    |    |    |
      // |____|____|____|____|
      // |    |    |    |    |
      // |____|____|____|____|
      // |    |    |X   |    |
      // |____|____|____|____|
      // |    |    |    |    |
      // |____|____|____|____|
      //
      // if the pixel is instead at the top right, we consider this one:
      // _____________________
      // |    |    |    |    |
      // |____|____|____|____|
      // |    |    |    |    |
      // |____|____|____|____|
      // |    |   X|    |    |
      // |____|____|____|____|
      // |    |    |    |    |
      // |____|____|____|____|
      //
      // if the pixel is instead at the bottom left:
      // _____________________
      // |    |    |    |    |
      // |____|____|____|____|
      // |    |    |    |    |
      // |____|____|X___|____|
      // |    |    |    |    |
      // |____|____|____|____|
      // |    |    |    |    |
      // |____|____|____|____|
      //
      // if the pixel is instead at the bottom right:
      // _____________________
      // |    |    |    |    |
      // |____|____|____|____|
      // |    |    |    |    |
      // |____|___X|____|____|
      // |    |    |    |    |
      // |____|____|____|____|
      // |    |    |    |    |
      // |____|____|____|____|
      //
      // Now that we have the search area, the principle is to find a weight
      // for each pixel of this area, and to do a weighted mean of their values
      // to find the interpolated value.
      //
      // To find the weight, we exploit the knowledge of image and ds_image.
      // We compare the value of the pixel in image (which is high res)
      // with each of the pixels of ds_image that are within the search zone.
      // We weight the pixels in a way that pixels that are very similar to
      // the high res pixel get a higher weight that pixels that are not
      // similar to the high res pixel.
      //
      // Once we have the weights, we do the weighted average of ds_ab
      // values to find the ab values
      float current = image[j * width + i];
      int j_ds0 = MIN(j >> 1, ds_height - 1);
      int i_ds0 = MIN(i >> 1, ds_width - 1);
      int i_offset = i & 1;
      int j_offset = j & 1;
      int i_min = MIN(MAX(i_ds0 - 2 + i_offset, 0), ds_width - 1);
      int i_max = MIN(MAX(i_ds0 + 1 + i_offset, 0), ds_width - 1);
      int j_min = MIN(MAX(j_ds0 - 2 + j_offset, 0), ds_height - 1);
      int j_max = MIN(MAX(j_ds0 + 1 + j_offset, 0), ds_height - 1);
      float w = 0.0f;
      float a = 0.0f;
      float b = 0.0f;
      for(int jds = j_min; jds <= j_max; jds++)
      {
        for(int ids = i_min; ids <= i_max; ids++)
        {
          float value_ds = ds_image[jds * ds_width + ids];
          // we use the ratio as a measure of difference to get
          // a difference in exposure
          float diff_ds = current / value_ds;
          if(diff_ds < 1.0f) diff_ds = 1.0f / diff_ds;
          float weight_ds = 1.0f / (diff_ds - 0.9999999f);
          // square the weight to increase the weight difference
          // between similar and dissimilar pixels
          weight_ds *= weight_ds;
          w += weight_ds;
          a += ds_ab[(jds * ds_width + ids) * 2] * weight_ds;
          b += ds_ab[(jds * ds_width + ids) * 2 + 1] * weight_ds;
        }
      }
      ab[(j * width + i) * 2] = a / w;
      ab[(j * width + i) * 2 + 1] = b / w;
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
