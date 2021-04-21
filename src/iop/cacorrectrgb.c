/*
    This file is part of darktable,
    Copyright (C) 2010-2020 darktable developers.

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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
// our includes go first:
#include "bauhaus/bauhaus.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include "common/gaussian.h"
#include "common/fast_guided_filter.h"

#include <gtk/gtk.h>
#include <stdlib.h>

DT_MODULE_INTROSPECTION(1, dt_iop_cacorrectrgb_params_t)

typedef enum dt_iop_cacorrectrgb_guide_channel_t
{
  DT_CACORRECT_RGB_R = 0,    // $DESCRIPTION: "red"
  DT_CACORRECT_RGB_G = 1,    // $DESCRIPTION: "green"
  DT_CACORRECT_RGB_B = 2     // $DESCRIPTION: "blue"
} dt_iop_cacorrectrgb_guide_channel_t;

typedef enum dt_iop_cacorrectrgb_mode_t
{
  DT_CACORRECT_MODE_STANDARD = 0,  // $DESCRIPTION: "standard"
  DT_CACORRECT_MODE_DARKEN = 1,    // $DESCRIPTION: "darken only"
  DT_CACORRECT_MODE_BRIGHTEN = 2   // $DESCRIPTION: "brighten only"
} dt_iop_cacorrectrgb_mode_t;


typedef struct dt_iop_cacorrectrgb_params_t
{
  dt_iop_cacorrectrgb_guide_channel_t guide_channel; // $DEFAULT: DT_CACORRECT_RGB_G $DESCRIPTION: "guide"
  float radius; // $MIN: 1 $MAX: 400 $DEFAULT: 5 $DESCRIPTION: "radius"
  float strength; // $MIN: 0 $MAX: 5 $DEFAULT: 3 $DESCRIPTION: "strength"
  dt_iop_cacorrectrgb_mode_t mode; // $DEFAULT: DT_CACORRECT_MODE_STANDARD $DESCRIPTION: "correction mode"
} dt_iop_cacorrectrgb_params_t;

typedef struct dt_iop_cacorrectrgb_gui_data_t
{
  GtkWidget *guide_channel, *radius, *strength, *mode;
} dt_iop_cacorrectrgb_gui_data_t;

// this returns a translatable name
const char *name()
{
  // make sure you put all your translatable strings into _() !
  return _("chromatic aberrations rgb");
}

// some additional flags (self explanatory i think):
int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

// where does it appear in the gui?
int default_group()
{
  return IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL;
}

int default_colorspace(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  return iop_cs_rgb;
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, p1, self->params_size);
}

static void normalize_manifolds(float *const restrict blurred_in, float *const restrict blurred_manifold_lower, float *const restrict blurred_manifold_higher, const size_t width, const size_t height, const dt_iop_cacorrectrgb_guide_channel_t guide)
{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(blurred_in, blurred_manifold_lower, blurred_manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(blurred_in, blurred_manifold_lower, blurred_manifold_higher:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    // normalize
    const float weighth = fmaxf(blurred_manifold_higher[k * 4 + 3], 1E-6);
    const float weightl = fmaxf(blurred_manifold_lower[k * 4 + 3], 1E-6);
    for(size_t c = 0; c < 3; c++)
    {
      blurred_manifold_higher[k * 4 + c] /= weighth;
      blurred_manifold_lower[k * 4 + c] /= weightl;
    }
    // replace by average if weight is too small
    if(weighth < 0.05f)
    {
      for(size_t c = 0; c < 3; c++)
      {
        blurred_manifold_higher[k * 4 + c] = blurred_in[k * 4 + c];
      }
    }
    if(weightl < 0.05f)
    {
      for(size_t c = 0; c < 3; c++)
      {
        blurred_manifold_lower[k * 4 + c] = blurred_in[k * 4 + c];
      }
    }
  }
}

static void get_manifolds(const float* const restrict in, const size_t width, const size_t height,
                          const size_t ch, const float sigma,
                          const dt_iop_cacorrectrgb_guide_channel_t guide,
                          float* const restrict manifolds)
{
  float *const restrict blurred_in = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict manifold_higher = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict manifold_lower = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_manifold_higher = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_manifold_lower = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));

  float max[4] = {INFINITY, INFINITY, INFINITY, 1.0f};
  float min[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  dt_gaussian_t *g = dt_gaussian_init(width, height, 4, max, min, sigma, 0);
  if(!g) return;
  dt_gaussian_blur_4c(g, in, blurred_in);

  // construct the manifolds
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, blurred_in, manifold_lower, manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(in, blurred_in, manifold_lower, manifold_higher:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float pixelg = in[k * 4 + guide];
    const float avg = blurred_in[k * 4 + guide];
    const float weighth = pixelg >= avg;
    const float weightl = pixelg <= avg;
    for(size_t c = 0; c < 3; c++)
    {
      const float pixel = fmaxf(in[k * 4 + c], 1E-6);
      manifold_higher[k * 4 + c] = pixel * weighth;
      manifold_lower[k * 4 + c] = pixel * weightl;
    }
    manifold_higher[k * 4 + 3] = weighth;
    manifold_lower[k * 4 + 3] = weightl;
  }

  dt_gaussian_blur_4c(g, manifold_higher, blurred_manifold_higher);
  dt_gaussian_blur_4c(g, manifold_lower, blurred_manifold_lower);

  normalize_manifolds(blurred_in, blurred_manifold_lower, blurred_manifold_higher, width, height, guide);

  // note that manifolds were constructed based on the value and average
  // of the guide channel ONLY.
  // this implies that the "higher" manifold in the channel c may be
  // actually lower than the "lower" manifold of that channel.
  // This happens in the following example:
  // guide:  1_____
  //               |_____0
  // guided:        _____1
  //         0_____|
  // here the higher manifold of guide is equal to 1, its lower manifold is
  // equal to 0. The higher manifold of the guided channel is equal to 0
  // as it is the average of the values where the guide is higher than its
  // average, and the lower manifold of the guided channel is equal to 1.

for(size_t p = 0; p < 15; p++)
{
  // refine the manifolds
  // improve result especially on very degraded images
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, blurred_in, manifold_lower, manifold_higher, blurred_manifold_lower, blurred_manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(in, blurred_in, manifold_lower, manifold_higher, blurred_manifold_lower, blurred_manifold_higher:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    // in order to refine the manifolds, we will compute weights
    // for which all channel will have a contribution.
    // this will allow to avoid taking too much into account pixels
    // that have wrong values due to the chromatic aberration
    //
    // for example, here:
    // guide:  1_____
    //               |_____0
    // guided: 1______
    //                |____0
    //               ^ this pixel makes the estimated lower manifold erroneous
    // here, the higher and lower manifolds values computed are:
    // _______|_higher_|________lower_________|
    // guide  |    1   |   0                  |
    // guided |    1   |(1 + 4 * 0) / 5 = 0.2 |
    //
    // the lower manifold of the guided is 0.2 if we consider only the guide
    // to determine weighth and weightl.
    //
    // at this step of the algorithm, we know if the higher manifold of guided
    // is actually higher or lower than the lower manifold.
    //
    // we can refine the manifolds by computing weights that promote pixels that
    // stretch the interval between the manifolds.
    // i.e., in our case, we give higher weights to the pixels that are equal to
    // 0 than to the pixel that is equal to 1 for the computation of the lower
    // manifold
    const float pixelg = in[k * 4 + guide];
    const float avgg = blurred_in[k * 4 + guide];
    if(pixelg > avgg)
    {
      // high manifold
      float weighth = 1.0f;
      for(size_t c = 0; c < 3; c++)
      {
        // if pixel value is close to the low manifold, give it a smaller weight
        // than if it is close to the high manifold
        const float pixel = logf(fmaxf(in[k * 4 + c], 1E-6));
        const float highc = logf(fmaxf(blurred_manifold_higher[k * 4 + c], 1E-6));
        const float lowc = logf(fmaxf(blurred_manifold_lower[k * 4 + c], 1E-6));
        float dist_manifolds = lowc - highc;
        float sign = (dist_manifolds >= 0) - (dist_manifolds < 0);
        dist_manifolds = fmaxf(fabsf(dist_manifolds), 1.0f);
        float dist = fminf(fmaxf(1.0f - sign * (pixel - highc) / dist_manifolds, 0.0f), 1.0f);
        dist *= dist;
        dist *= dist;
        weighth *= dist;
      }
      for(size_t c = 0; c < 3; c++)
      {
        const float pixel = fmaxf(in[k * 4 + c], 1E-6);
        manifold_higher[k * 4 + c] = pixel * weighth;
      }
      manifold_higher[k * 4 + 3] = weighth;
    }
    else
    {
      float weightl = 1.0f;
      for(size_t c = 0; c < 3; c++)
      {
        // if pixel value is close to the high manifold, give it a smaller weight
        // than if it is close to the low manifold
        const float pixel = logf(fmaxf(in[k * 4 + c], 1E-6));
        const float highc = logf(fmaxf(blurred_manifold_higher[k * 4 + c], 1E-6));
        const float lowc = logf(fmaxf(blurred_manifold_lower[k * 4 + c], 1E-6));
        float dist_manifolds = highc - lowc;
        float sign = (dist_manifolds >= 0) - (dist_manifolds < 0);
        dist_manifolds = fmaxf(fabsf(dist_manifolds), 1.0f);
        float dist = fminf(fmaxf(1.0f - sign * (pixel - lowc) / dist_manifolds, 0.0f), 1.0f);
        dist *= dist;
        dist *= dist;
        weightl *= dist;
      }
      for(size_t c = 0; c < 3; c++)
      {
        const float pixel = fmaxf(in[k * 4 + c], 1E-6);
        manifold_lower[k * 4 + c] = pixel * weightl;
      }
      manifold_lower[k * 4 + 3] = weightl;
    }
  }

  dt_gaussian_blur_4c(g, manifold_higher, blurred_manifold_higher);
  dt_gaussian_blur_4c(g, manifold_lower, blurred_manifold_lower);
  normalize_manifolds(blurred_in, blurred_manifold_lower, blurred_manifold_higher, width, height, guide);
}
  dt_gaussian_free(g);
  dt_free_align(manifold_lower);
  dt_free_align(manifold_higher);


#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(manifolds, blurred_manifold_lower, blurred_manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(manifolds, blurred_manifold_lower, blurred_manifold_higher:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    for(size_t c = 0; c < 3; c++)
    {
      manifolds[k * 6 + c] = blurred_manifold_higher[k * 4 + c];
      manifolds[k * 6 + 3 + c] = blurred_manifold_lower[k * 4 + c];
    }
  }
  dt_free_align(blurred_in);
  dt_free_align(blurred_manifold_lower);
  dt_free_align(blurred_manifold_higher);
}

static void apply_correction(const float* const restrict in,
                          const float* const restrict manifolds,
                          const size_t width, const size_t height,
                          const size_t ch, const float sigma,
                          const dt_iop_cacorrectrgb_guide_channel_t guide,
                          const dt_iop_cacorrectrgb_mode_t mode,
                          const float strength,
                          float* const restrict out)

{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, width, height, guide, manifolds, out, sigma, mode, strength) \
  schedule(simd:static) aligned(in, manifolds, out)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float high_guide = fmaxf(manifolds[k * 6 + guide], 1E-6);
    const float low_guide = fmaxf(manifolds[k * 6 + 3 + guide], 1E-6);
    const float log_high = logf(high_guide);
    const float log_low = logf(low_guide);
    const float pixelg = fmaxf(in[k * 4 + guide], 0.0f);
    const float log_pixg = logf(fminf(fmaxf(pixelg, low_guide), high_guide));
    float dist = fabsf(log_high - log_pixg) / fmaxf(fabsf(log_high - log_low), 1E-6);
    dist = fminf(dist, 1.0f);

    for(size_t kc = 0; kc <= 1; kc++)
    {
      const size_t c = (guide + kc + 1) % 3;
      //const size_t c2 = (guide + (kc ^ 1) + 1) % 3;
      const float pixelc = fmaxf(in[k * 4 + c], 0.0f);

      const float xl = log2f(fmaxf(low_guide, 1E-6));
      const float yl = log2f(fmaxf(manifolds[k * 6 + 3 + c], 1E-6));
      const float xh = log2f(fmaxf(high_guide, 1E-6));
      const float yh = log2f(fmaxf(manifolds[k * 6 + c], 1E-6));

      const float x = log2f(fmaxf(pixelg, 1E-6));
      const float y = log2f(fmaxf(pixelc, 1E-6));

      // (xl,yl) and (xh,yh) are good.
      // (xl,yh) and (xh,yl) are bad points, result of chromatic aberration
      // if (x,y) is too close from a bad point, correct it.
      // const float dist_bad_good = fminf(fabsf(yh - yl), xh - xl);

      // float strength = 1.0f; //in EV
      // strength = fminf(strength, dist_bad_good * 0.9f);

      const float w_bad = 1.0f;

      const float dist_to_bad_1 = sqrtf((x - xl) * (x - xl) + (y - yh) * (y - yh));
      const float dist_bad_h1 = fabsf((x - xl) + (y - yh)) / sqrtf(2.0f);
      const float dist_bad_perp_1 = sqrtf(fmaxf(dist_to_bad_1 * dist_to_bad_1 - dist_bad_h1 * dist_bad_h1, 0.0f));

      const float dist_to_bad_2 = sqrtf((x - xh) * (x - xh) + (y - yl) * (y - yl));
      const float dist_bad_h2 = fabsf((x - xh) + (y - yl)) / sqrtf(2.0f);
      const float dist_bad_perp_2 = sqrtf(fmaxf(dist_to_bad_2 * dist_to_bad_2 - dist_bad_h2 * dist_bad_h2, 0.0f));

      const float dist_to_bad = fminf((1.0f - w_bad) * dist_bad_perp_1 + w_bad * dist_to_bad_1, (1.0f - w_bad) * dist_bad_perp_2 + w_bad * dist_to_bad_2);

      const float w_good = 1.0f;

      const float dist_to_good_1 = sqrtf((x - xl) * (x - xl) + (y - yl) * (y - yl));
      const float dist_good_h1 = fabsf((x - xl) + (y - yl)) / sqrtf(2.0f);
      const float dist_good_perp_1 = sqrtf(fmaxf(dist_to_good_1 * dist_to_good_1 - dist_good_h1 * dist_good_h1, 0.0f));

      const float dist_to_good_2 = sqrtf((x - xh) * (x - xh) + (y - yh) * (y - yh));
      const float dist_good_h2 = fabsf((x - xh) + (y - yh)) / sqrtf(2.0f);
      const float dist_good_perp_2 = sqrtf(fmaxf(dist_to_good_2 * dist_to_good_2 - dist_good_h2 * dist_good_h2, 0.0f));

      const float dist_to_good = fminf((1.0f - w_good) * dist_good_perp_1 + w_good * dist_to_good_1, (1.0f - w_good) * dist_good_perp_2 + w_good * dist_to_good_2);

      const float dist_lh_g = fabsf(xl - xh);
      const float dist_lh_c = fabsf(yl - yh);
      const float confidence = 1.0f + 0.0f * exp2f((1.0f - fmaxf(dist_lh_c, dist_lh_g) / dist_lh_g) * 4.0f);

      const float weight_corr = confidence * exp2f(-dist_to_bad / ((dist_to_good + 0.01f) * strength));

      //const float weight_corr = exp2f(-fmaxf(dist_to_bad - strength, 0.0f) / fmaxf(40.0f * strength, 1E-6));

      const float diff_high_manifolds = manifolds[k * 6 + c] - high_guide;
      const float diff_low_manifolds = manifolds[k * 6 + 3 + c] - low_guide;
      const float diff = (diff_low_manifolds * dist) + (diff_high_manifolds * fmaxf(1.0f - dist, 0.0f));
      const float estimate_d = pixelg + diff;

      const float ratio_high_manifolds = manifolds[k * 6 + c] / high_guide;
      const float ratio_low_manifolds = manifolds[k * 6 + 3 + c] / low_guide;
      const float ratio = powf(ratio_low_manifolds, dist) * powf(ratio_high_manifolds, fmaxf(1.0f - dist, 0.0f));
      const float estimate_r = pixelg * ratio;

      float dist_dr = 1.0f; //(pixelc - estimate_d) / (estimate_r - estimate_d);
      dist_dr = fminf(dist_dr, 1.0f);
      dist_dr = fmaxf(dist_dr, 0.0f);
      const float outp = powf(pixelc, fmaxf(1.0f - weight_corr, 0.0f)) * powf(estimate_d * fmaxf(1.0f - dist_dr, 0.0f) + estimate_r * dist_dr, weight_corr);

      switch(mode)
      {
        case DT_CACORRECT_MODE_STANDARD:
          out[k * 4 + c] = outp;
          break;
        case DT_CACORRECT_MODE_DARKEN:
          out[k * 4 + c] = fminf(outp, pixelc);
          break;
        case DT_CACORRECT_MODE_BRIGHTEN:
          out[k * 4 + c] = fmaxf(outp, pixelc);
          break;
      }
    }

    out[k * 4 + guide] = pixelg;
    out[k * 4 + 3] = in[k * 4 + 3];
  }
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_cacorrectrgb_params_t *d = (dt_iop_cacorrectrgb_params_t *)piece->data;
  const float scale = piece->iscale / roi_in->scale;
  const int ch = piece->colors;
  const size_t width = roi_out->width;
  const size_t height = roi_out->height;
  const float* in = (float*)ivoid;
  float* out = (float*)ovoid;
  const float sigma = MAX(d->radius / scale, 1.0f);

  if(ch != 4)
  {
    memcpy(out, in, width * height * ch * sizeof(float));
    return;
  }

  const dt_iop_cacorrectrgb_guide_channel_t guide = d->guide_channel;
  const dt_iop_cacorrectrgb_mode_t mode = d->mode;

  const float downsize = 3.0f;
  const size_t ds_width = width / downsize;
  const size_t ds_height = height / downsize;
  float *const restrict ds_in = dt_alloc_sse_ps(dt_round_size_sse(ds_width * ds_height * ch));
  // we use only one variable for both higher and lower manifolds in order
  // to save time by doing only one bilinear interpolation instead of 2.
  float *const restrict ds_manifolds = dt_alloc_sse_ps(dt_round_size_sse(ds_width * ds_height * 6));
  // Downsample the image for speed-up
  interpolate_bilinear(in, width, height, ds_in, ds_width, ds_height, 4);
  get_manifolds(ds_in, ds_width, ds_height, ch, sigma / downsize, guide, ds_manifolds);
  dt_free_align(ds_in);
  float *const restrict manifolds = dt_alloc_sse_ps(dt_round_size_sse(width * height * 6));
  // upscale manifolds
  interpolate_bilinear(ds_manifolds, ds_width, ds_height, manifolds, width, height, 6);
  dt_free_align(ds_manifolds);
  apply_correction(in, manifolds, width, height, ch, sigma, guide, mode, 1000.0f/*powf(10.0f, d->strength - 2.0f)*/, out);
  dt_free_align(manifolds);


  // reduce artifacts

  // contains 2 guided channels of in, and of out
  float *const restrict in_out = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, out, in_out, width, height, guide, ch) \
  schedule(simd:static) aligned(in, out, in_out:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    for(size_t kc = 0; kc <= 1; kc++)
    {
      size_t c = (guide + kc + 1) % 3;
      in_out[k * ch + kc * 2 + 0] = log2f(fmaxf(in[k * 4 + c], 0.000016f)) + 16.0f;
      in_out[k * ch + kc * 2 + 1] = log2f(fmaxf(out[k * 4 + c], 0.000016f)) + 16.0f;
    }
  }

  float *const restrict blurred_in_out = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float max[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
  float min[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  dt_gaussian_t *g = dt_gaussian_init(width, height, 4, max, min, sigma, 0);
  if(!g) return;
  dt_gaussian_blur_4c(g, in_out, blurred_in_out);
  dt_gaussian_free(g);
  dt_free_align(in_out);

  const float stren = powf(20.0f, 2.0f - d->strength);
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, out, blurred_in_out, width, height, guide, stren, ch) \
  schedule(simd:static) aligned(in, out, blurred_in_out:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    for(size_t kc = 0; kc <= 1; kc++)
    {
      size_t c = (guide + kc + 1) % 3;
      const float avg_in = blurred_in_out[k * ch + kc * 2 + 0];
      const float avg_out = blurred_in_out[k * ch + kc * 2 + 1];
      const float w = expf(-fmaxf(fabsf(avg_out - avg_in), 0.01f) * stren);
      out[k * ch + c] = fmaxf(1.0f - w, 0.0f) * fmaxf(in[k * ch + c], 0.0f) + w * fmaxf(out[k * ch + c], 0.0f);
    }
  }



}

/** gui setup and update, these are needed. */
void gui_update(dt_iop_module_t *self)
{
  dt_iop_cacorrectrgb_gui_data_t *g = (dt_iop_cacorrectrgb_gui_data_t *)self->gui_data;
  dt_iop_cacorrectrgb_params_t *p = (dt_iop_cacorrectrgb_params_t *)self->params;

  dt_bauhaus_combobox_set_from_value(g->guide_channel, p->guide_channel);
  dt_bauhaus_slider_set_soft(g->radius, p->radius);
  dt_bauhaus_slider_set_soft(g->strength, p->strength);
  dt_bauhaus_combobox_set_from_value(g->mode, p->mode);
}

/** optional: if this exists, it will be called to init new defaults if a new image is
 * loaded from film strip mode. */
void reload_defaults(dt_iop_module_t *module)
{
  dt_iop_cacorrectrgb_params_t *d = (dt_iop_cacorrectrgb_params_t *)module->default_params;

  d->guide_channel = DT_CACORRECT_RGB_G;
  d->radius = 5.0f;
  d->strength = 3.0f;
  d->mode = DT_CACORRECT_MODE_STANDARD;

  dt_iop_cacorrectrgb_gui_data_t *g = (dt_iop_cacorrectrgb_gui_data_t *)module->gui_data;
  if(g)
  {
    dt_bauhaus_combobox_set_default(g->guide_channel, d->guide_channel);
    dt_bauhaus_slider_set_default(g->radius, d->radius);
    dt_bauhaus_slider_set_soft_range(g->radius, 1.0, 20.0);
    dt_bauhaus_slider_set_default(g->strength, d->strength);
    dt_bauhaus_combobox_set_default(g->mode, d->mode);
  }
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_cacorrectrgb_gui_data_t *g = IOP_GUI_ALLOC(cacorrectrgb);
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);
  g->guide_channel = dt_bauhaus_combobox_from_params(self, "guide_channel");
  gtk_widget_set_tooltip_text(g->guide_channel, _("channel used as a reference to\n"
                                           "correct the other channels.\n"
                                           "use sharpest channel if some\n"
                                           "channels are blurry.\n"
                                           "try changing guide channel if you\n"
                                           "have artefacts."));
  g->radius = dt_bauhaus_slider_from_params(self, "radius");
  gtk_widget_set_tooltip_text(g->radius, _("increase for stronger correction\n"));
  g->strength = dt_bauhaus_slider_from_params(self, "strength");
  gtk_widget_set_tooltip_text(g->strength, _("increase for stronger correction\n"));

  gtk_box_pack_start(GTK_BOX(self->widget), dt_ui_label_new(_("advanced parameters:")), TRUE, TRUE, 0);
  g->mode = dt_bauhaus_combobox_from_params(self, "mode");
  gtk_widget_set_tooltip_text(g->mode, _("correction mode to use.\n"
                                         "can help with multiple\n"
                                         "instances for very damaged\n"
                                         "images.\n"
                                         "darken only is particularly\n"
                                         "efficient to correct blue\n"
                                         "chromatic aberration."));
}
