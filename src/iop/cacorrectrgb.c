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

typedef struct dt_iop_cacorrectrgb_params_t
{
  dt_iop_cacorrectrgb_guide_channel_t guide_channel; // $DEFAULT: DT_CACORRECT_RGB_G $DESCRIPTION: "guide"
  int radius; // $MIN: 1 $MAX: 100 $DEFAULT: 1 $DESCRIPTION: "radius"
} dt_iop_cacorrectrgb_params_t;

typedef struct dt_iop_cacorrectrgb_gui_data_t
{
  GtkWidget *guide_channel, *radius;
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

static void ca_correct_rgb(const float* const restrict in, const size_t width, const size_t height,
                          const size_t ch, const float sigma,
                          const dt_iop_cacorrectrgb_guide_channel_t guide,
                          float* const restrict out, float* const restrict ratio_manifolds_guide)
{
  //TODO do all computation with downscaled image

  float *const restrict blurred_in = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict manifold_higher = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict manifold_lower = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_manifold_higher = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_manifold_lower = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));

  float minr = 10000000.0f;
  float maxr = 0.0f;
  float ming = 10000000.0f;
  float maxg = 0.0f;
  float minb = 10000000.0f;
  float maxb = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, width, height) \
  schedule(simd:static) aligned(in:64) \
  reduction(max:maxr, maxg, maxb)\
  reduction(min:minr, ming, minb)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float pixelr = in[k * 4];
    if(pixelr < minr) minr = pixelr;
    if(pixelr > maxr) maxr = pixelr;
    const float pixelg = in[k * 4 + 1];
    if(pixelg < ming) ming = pixelg;
    if(pixelg > maxg) maxg = pixelg;
    const float pixelb = in[k * 4 + 2];
    if(pixelb < minb) minb = pixelb;
    if(pixelb > maxb) maxb = pixelb;
  }

  float max[4] = {maxr, maxg, maxb, 1.0f};
  float min[4] = {fminf(minr, 0.0f), fminf(ming, 0.0f), fminf(minb, 0.0f), 0.0f};
  dt_gaussian_t *g = dt_gaussian_init(width, height, 4, max, min, sigma, 0);
  if(!g) return;
  dt_gaussian_blur_4c(g, in, blurred_in);

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
      const float pixel = in[k * 4 + c];
      manifold_higher[k * 4 + c] = pixel * weighth;
      manifold_lower[k * 4 + c] = pixel * weightl;
    }
    manifold_higher[k * 4 + 3] = weighth;
    manifold_lower[k * 4 + 3] = weightl;
  }

  dt_gaussian_blur_4c(g, manifold_higher, blurred_manifold_higher);
  dt_gaussian_blur_4c(g, manifold_lower, blurred_manifold_lower);

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

// iterations to refine the manifolds. Usually useless.
// can improve result on VERY degraded images
for(int n = 0; n < 0; n++)
{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, blurred_in, manifold_lower, manifold_higher, blurred_manifold_lower, blurred_manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(in, blurred_in, manifold_lower, manifold_higher, blurred_manifold_lower, blurred_manifold_higher:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float pixelg = in[k * 4 + guide];
    const float avgg = blurred_in[k * 4 + guide];
    float weighth = fmaxf(pixelg >= avgg, 0.1f);
    float weightl = fmaxf(pixelg <= avgg, 0.1f);
    // const float log_range = logf(blurred_manifold_higher[k * 4 + guide] / fmaxf(blurred_manifold_lower[k * 4 + guide], 1E-6));
    // const float min_log_diff = fminf(log_range, 1.0f);
    // const float max_log_diff = fminf(log_range, 8.0f); //TODO not sure it is still needed

    for(size_t c = 0; c < 3; c++)
    {
      const float pixel = in[k * 4 + c];
      // const float avg = blurred_in[k * 4 + c];
      float high = blurred_manifold_higher[k * 4 + c];
      float low = blurred_manifold_lower[k * 4 + c];
      // if(high > low)
      // {
      //   low /= 2.0f;
      //   high *= 2.0f;
      // }
      // else
      // {
      //   high /= 2.0f;
      //   low *= 2.0f;
      // }
      // float log_diff_low = (pixel < fminf(low, high)) ? 1.0f : fmaxf(fminf(fabsf(logf(fmaxf(pixel, 1E-6) / fmaxf(low, 1E-6))), max_log_diff), min_log_diff) / max_log_diff;
      // float log_diff_high = (pixel > fmaxf(low, high)) ? 1.0f : fmaxf(fminf(fabsf(logf(fmaxf(pixel, 1E-6) / fmaxf(high, 1E-6))), max_log_diff), min_log_diff) / max_log_diff;
      float log_diff_low = (pixel < fminf(low, high)) ? 1.0f : fmaxf(pixel, 1E-6) / fmaxf(low, 1E-6);
      float log_diff_high = (pixel > fmaxf(low, high)) ? 1.0f : fmaxf(pixel, 1E-6) / fmaxf(high, 1E-6);
      // log_diff_low *= log_diff_low;
      // log_diff_high *= log_diff_high;
      if(high > low)
      {
        weighth /= log_diff_high;
        weightl /= log_diff_low;

        // // for h, we want to be as far from low as possible
        // if(pixel > low)
        //   weighth /= log_diff_low;//fminf(pixel / low / 4.0f, 1.0f);
        // else
        //   weighth *= 0.0001f;
        // if(pixel < high)
        //   weightl *= log_diff_high;//fminf(high / pixel / 4.0f, 1.0f);
        // else
        //   weightl *= 0.0001f;
      }
      else
      {
        weighth /= log_diff_low;
        weightl /= log_diff_high;

        // if(pixel > high)
        //   weighth *= log_diff_high;//fminf(pixel / high / 4.0f, 1.0f);
        // else
        //   weighth *= 0.0001f;
        // if(pixel < low)
        //   weightl *= log_diff_low;//fminf(low / pixel / 4.0f, 1.0f);
        // else
        //   weightl *= 0.0001f;
      }
    }
    // weighth = powf(weighth, 0.333f);
    // weightl = powf(weightl, 0.333f);
    for(size_t c = 0; c < 3; c++)
    {
      const float pixel = in[k * 4 + c];
      manifold_higher[k * 4 + c] = pixel * weighth;
      manifold_lower[k * 4 + c] = pixel * weightl;
    }
    manifold_higher[k * 4 + 3] = weighth;
    manifold_lower[k * 4 + 3] = weightl;
  }

  dt_gaussian_blur_4c(g, manifold_higher, blurred_manifold_higher);
  dt_gaussian_blur_4c(g, manifold_lower, blurred_manifold_lower);

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
  dt_gaussian_free(g);

  dt_free_align(manifold_lower);
  dt_free_align(manifold_higher);

  //TODO also compute manifolds guided by each channel

  //TODO upscale blurred_manifolds and blurred_in here
  // for this to be worth it we need upscaling to be faster than gaussian blur

#ifdef _OPENMP
#pragma omp parallel for default(none) \
dt_omp_firstprivate(in, width, height, guide, blurred_in, blurred_manifold_higher, blurred_manifold_lower, out) \
  schedule(static)
#endif
  for(size_t i = 0; i < height; i++)
  {
    for(size_t j = 0; j < width; j++)
    {
      for(size_t kc = 1; kc <= 2; kc++)
      {
        size_t c = (guide + kc) % 3;
        const float pixelg = in[(i * width + j) * 4 + guide];

        float dist = 0.0f;
        const float high_guide = fmaxf(blurred_manifold_higher[(i * width + j) * 4 + guide], 1E-6);
        const float low_guide = fmaxf(blurred_manifold_lower[(i * width + j) * 4 + guide], 1E-6);
        const float log_pixg = logf(fminf(fmaxf(pixelg, low_guide), high_guide));
        float ratio_high_manifolds = blurred_manifold_higher[(i * width + j) * 4 + c] / high_guide;
        float ratio_low_manifolds = blurred_manifold_lower[(i * width + j) * 4 + c] / low_guide;

        const float log_high = logf(high_guide);
        const float log_low = logf(low_guide);
        dist = fabsf(log_high - log_pixg) / fmaxf(fabsf(log_high - log_low), 1E-6);
        dist = fminf(dist, 1.0f);

        // if(dist < 0.5f) dist = dist * dist / 0.5f;
        // if(dist > 0.5f) dist = 1.0f - (1.0f - dist) * (1.0f - dist) / 0.5f;

        //float ratio = dist * ratio_means + (1.0f - dist) * ratio_means_manifold;
        float ratio = powf(ratio_low_manifolds, dist) * powf(ratio_high_manifolds, 1.0f - dist);
        out[(i * width + j) * 4 + c] = in[(i * width + j) * 4 + guide] * ratio;
      }
      out[(i * width + j) * 4 + guide] = in[(i * width + j) * 4 + guide];
    }
  }

  if(ratio_manifolds_guide != NULL)
  {
  #ifdef _OPENMP
  #pragma omp parallel for simd default(none) \
  dt_omp_firstprivate(ratio_manifolds_guide, blurred_manifold_lower, blurred_manifold_higher, width, height, guide) \
    schedule(simd:static) aligned(ratio_manifolds_guide, blurred_manifold_lower, blurred_manifold_higher:64)
  #endif
    for(size_t k = 0; k < width * height; k++)
    {
      ratio_manifolds_guide[k] = blurred_manifold_higher[k * 4 + guide] / fmaxf(blurred_manifold_lower[k * 4 + guide], 1E-6);
    }
  }

  dt_free_align(blurred_in);
  dt_free_align(blurred_manifold_lower);
  dt_free_align(blurred_manifold_higher);
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

  if(ch != 4/* || (piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW) == DT_DEV_PIXELPIPE_PREVIEW
     || (piece->pipe->type & DT_DEV_PIXELPIPE_THUMBNAIL) == DT_DEV_PIXELPIPE_THUMBNAIL*/)
  {
    memcpy(out, in, width * height * ch * sizeof(float));
    return;
  }

  float *const restrict out_s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out_4s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out_16s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict ratio_manifolds_guide_s = dt_alloc_sse_ps(dt_round_size_sse(width * height));
  float *const restrict ratio_manifolds_guide_4s = dt_alloc_sse_ps(dt_round_size_sse(width * height));
  float *const restrict ratio_manifolds_guide_16s = dt_alloc_sse_ps(dt_round_size_sse(width * height));
  float *const restrict guide_in = dt_alloc_sse_ps(dt_round_size_sse(width * height));
  float *const restrict blurred_guide_in = dt_alloc_sse_ps(dt_round_size_sse(width * height));

  const dt_iop_cacorrectrgb_guide_channel_t guide = d->guide_channel;
  const float force = d->radius;
  const float sigma = MAX(4.0f/*d->radius*/ / scale, 1);
  // we compute the correction 3 times and them blend them
  // in order to have an adaptative correction depending on
  // the amount of chromatic aberration in each part of the
  // image
  ca_correct_rgb(in, width, height, ch, d->radius, guide, out, ratio_manifolds_guide_s);
  return;
  ca_correct_rgb(in, width, height, ch, sigma, guide, out_s, ratio_manifolds_guide_s);
  ca_correct_rgb(in, width, height, ch, 4.0f * sigma, guide, out_4s, ratio_manifolds_guide_4s);
  ca_correct_rgb(in, width, height, ch, 16.0f * sigma, guide, out_16s, ratio_manifolds_guide_16s);

  float ming = 10000000.0f;
  float maxg = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, guide_in, width, height, guide) \
  schedule(simd:static) aligned(in, guide_in:64) \
  reduction(max:maxg)\
  reduction(min:ming)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float pixelg = in[k * 4 + guide];
    guide_in[k] = pixelg;
    if(pixelg < ming) ming = pixelg;
    if(pixelg > maxg) maxg = pixelg;
  }

  dt_gaussian_t *g = dt_gaussian_init(width, height, 1, &maxg, &ming, sigma / 4.0f, 0);
  if(!g) return;
  dt_gaussian_blur(g, guide_in, blurred_guide_in);
  dt_gaussian_free(g);

  ming = 10000000.0f;
  maxg = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(blurred_guide_in, guide_in, width, height, guide) \
  schedule(simd:static) aligned(blurred_guide_in, guide_in:64) \
  reduction(max:maxg)\
  reduction(min:ming)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float a = fmaxf(guide_in[k], 1E-4);
    const float b = fmaxf(blurred_guide_in[k], 1E-4);
    const float pixelg = powf(fminf(a / b, b / a), 256.0f);// - 1.0f;
    guide_in[k] = pixelg;
    if(pixelg < ming) ming = pixelg;
    if(pixelg > maxg) maxg = pixelg;
  }

  g = dt_gaussian_init(width, height, 1, &maxg, &ming, 32.0f * sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, guide_in, blurred_guide_in);
  dt_gaussian_free(g);
  dt_free_align(guide_in);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, blurred_guide_in, out_s, out_4s, out_16s, out, ratio_manifolds_guide_s, ratio_manifolds_guide_4s, ratio_manifolds_guide_16s, width, height, guide, sigma, force) \
  schedule(simd:static) aligned(in, blurred_guide_in, out_s, out_4s, out_16s, out, ratio_manifolds_guide_s, ratio_manifolds_guide_4s, ratio_manifolds_guide_16s:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    out[k * 4 + guide] = in[k * 4 + guide];
    out[k * 4 + 3] = in[k * 4 + 3];
    float blurry_vs_sharp_weight = blurred_guide_in[k];
    blurry_vs_sharp_weight *= blurry_vs_sharp_weight;
    blurry_vs_sharp_weight *= blurry_vs_sharp_weight;
    // out[k * 4 + 0] = blurry_vs_sharp_weight;
    // out[k * 4 + 1] = blurry_vs_sharp_weight;
    // out[k * 4 + 2] = blurry_vs_sharp_weight;
    // continue;

    float ratio = fmaxf(powf(ratio_manifolds_guide_s[k] * ratio_manifolds_guide_4s[k] * ratio_manifolds_guide_16s[k], 0.33333f), 1.0f);
    //ratio *= ratio;
    ratio = (ratio - 1.0f) * force * force * force * force * blurry_vs_sharp_weight;
    size_t c1 = (guide + 1) % 3;
    size_t c2 = (guide + 2) % 3;
    if(ratio < sigma)
    {
      float w = 1.0f - ratio / sigma;
      out[k * 4 + c1] = w * in[k * 4 + c1] + (1.0f - w) * out_s[k * 4 + c1];
      out[k * 4 + c2] = w * in[k * 4 + c2] + (1.0f - w) * out_s[k * 4 + c2];
    }
    else if(ratio < 4.0f * sigma)
    {
      float w = 1.0f - (ratio - sigma) / (3.0f * sigma);
      out[k * 4 + c1] = w * out_s[k * 4 + c1] + (1.0f - w) * out_4s[k * 4 + c1];
      out[k * 4 + c2] = w * out_s[k * 4 + c2] + (1.0f - w) * out_4s[k * 4 + c2];
    }
    else if(ratio < 16.0f * sigma)
    {
      float w = 1.0f - (ratio - 4.0f * sigma) / (12.0f * sigma);
      out[k * 4 + c1] = w * out_4s[k * 4 + c1] + (1.0f - w) * out_16s[k * 4 + c1];
      out[k * 4 + c2] = w * out_4s[k * 4 + c2] + (1.0f - w) * out_16s[k * 4 + c2];
    }
    else
    {
      out[k * 4 + c1] = out_16s[k * 4 + c1];
      out[k * 4 + c2] = out_16s[k * 4 + c2];
    }
  }

  dt_free_align(out_s);
  dt_free_align(out_4s);
  dt_free_align(out_16s);
  dt_free_align(ratio_manifolds_guide_s);
  dt_free_align(ratio_manifolds_guide_4s);
  dt_free_align(ratio_manifolds_guide_16s);
  dt_free_align(blurred_guide_in);
}

/** gui setup and update, these are needed. */
void gui_update(dt_iop_module_t *self)
{
  dt_iop_cacorrectrgb_gui_data_t *g = (dt_iop_cacorrectrgb_gui_data_t *)self->gui_data;
  dt_iop_cacorrectrgb_params_t *p = (dt_iop_cacorrectrgb_params_t *)self->params;

  dt_bauhaus_combobox_set_from_value(g->guide_channel, p->guide_channel);
  dt_bauhaus_slider_set(g->radius, p->radius);
}

/** optional: if this exists, it will be called to init new defaults if a new image is
 * loaded from film strip mode. */
void reload_defaults(dt_iop_module_t *module)
{
  dt_iop_cacorrectrgb_params_t *d = (dt_iop_cacorrectrgb_params_t *)module->default_params;

  d->guide_channel = DT_CACORRECT_RGB_G;
  d->radius = 1;

  dt_iop_cacorrectrgb_gui_data_t *g = (dt_iop_cacorrectrgb_gui_data_t *)module->gui_data;
  if(g)
  {
    dt_bauhaus_combobox_set_default(g->guide_channel, d->guide_channel);
    dt_bauhaus_slider_set_default(g->radius, d->radius);
  }
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_cacorrectrgb_gui_data_t *g = IOP_GUI_ALLOC(cacorrectrgb);
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);
  g->guide_channel = dt_bauhaus_combobox_from_params(self, "guide_channel");
  g->radius = dt_bauhaus_slider_from_params(self, "radius");
}
