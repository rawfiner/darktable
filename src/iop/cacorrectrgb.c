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

typedef enum dt_iop_cacorrectrgb_direction_t
{
  DT_CACORRECT_N_S = 0,    // $DESCRIPTION: "north south"
  DT_CACORRECT_E_W = 1,    // $DESCRIPTION: "east west"
  DT_CACORRECT_NE_SW = 2,  // $DESCRIPTION: "north-east south-west"
  DT_CACORRECT_NW_SE = 3   // $DESCRIPTION: "north-west south-east"
} dt_iop_cacorrectrgb_direction_t;

typedef struct dt_iop_cacorrectrgb_params_t
{
  dt_iop_cacorrectrgb_guide_channel_t guide_channel; // $DEFAULT: DT_CACORRECT_RGB_G $DESCRIPTION: "guide"
  int radius; // $MIN: 1 $MAX: 50 $DEFAULT: 1 $DESCRIPTION: "radius"
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
                          float* const restrict out)
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
  dt_gaussian_free(g);

  dt_free_align(manifold_lower);
  dt_free_align(manifold_higher);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(blurred_manifold_lower, blurred_manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(blurred_manifold_lower, blurred_manifold_higher:64)
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
  }

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
        const float log_pixg = logf(fmaxf(pixelg, 1E-6));
        const float avg = blurred_in[(i * width + j) * 4 + guide];
        const float log_avg = logf(fmaxf(avg, 1E-6));
        const float ratio_means = blurred_in[(i * width + j) * 4 + c] / fmaxf(avg, 1E-6);

        float dist = 0.0f;
        float ratio_means_manifold;
        if(pixelg >= avg)
        {
          const float avg_high = blurred_manifold_higher[(i * width + j) * 4 + guide];
          ratio_means_manifold = blurred_manifold_higher[(i * width + j) * 4 + c] / fmaxf(avg_high, 1E-6);
          const float log_avgh = logf(fmaxf(avg_high, 1E-6));
          dist = (log_avgh - fminf(log_pixg, log_avgh)) / fmaxf(log_avgh - log_avg, 1E-6);
        }
        else
        {
          const float avg_low = blurred_manifold_lower[(i * width + j) * 4 + guide];
          ratio_means_manifold = blurred_manifold_lower[(i * width + j) * 4 + c] / fmaxf(avg_low, 1E-6);
          const float log_avgl = logf(fmaxf(avg_low, 1E-6));
          dist = (fmaxf(log_pixg, log_avgl) - log_avgl) / fmaxf(log_avg - log_avgl, 1E-6);
        }

        float ratio = dist * ratio_means + (1.0f - dist) * ratio_means_manifold;
        out[(i * width + j) * 4 + c] = in[(i * width + j) * 4 + guide] * ratio;
      }
      out[(i * width + j) * 4 + guide] = in[(i * width + j) * 4 + guide];
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

  if(ch != 4 || (piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW) == DT_DEV_PIXELPIPE_PREVIEW
     || (piece->pipe->type & DT_DEV_PIXELPIPE_THUMBNAIL) == DT_DEV_PIXELPIPE_THUMBNAIL)
  {
    memcpy(out, in, width * height * ch * sizeof(float));
    return;
  }

  float *const restrict out_s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out_4s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out_16s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict correlation_1 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict correlation_2 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_correlation_1 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_correlation_2 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));

  const dt_iop_cacorrectrgb_guide_channel_t guide = d->guide_channel;
  const float sigma = MAX(d->radius / scale, 1);
  // we compute the correction 3 times and them blend them
  // in order to have an adaptative correction depending on
  // the amount of chromatic aberration in each part of the
  // image
  ca_correct_rgb(in, width, height, ch, sigma, guide, out_s);
  ca_correct_rgb(in, width, height, ch, 4.0f * sigma, guide, out_4s);
  ca_correct_rgb(in, width, height, ch, 16.0f * sigma, guide, out_16s);

  float min1 = 10000000.0f;
  float max1 = 0.0f;
  float min2 = 10000000.0f;
  float max2 = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, out_s, correlation_1, correlation_2, width, height, guide) \
  schedule(simd:static) aligned(in, out_s, correlation_1, correlation_2:64) \
  reduction(max:max1, max2)\
  reduction(min:min1, min2)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    size_t c = (guide + 1) % 3;
    float x = in[k * 4 + c];
    float y = out_s[k * 4 + c];
    x = fabsf(x - y) / x;
    correlation_1[k * 4] = x * y;
    correlation_1[k * 4 + 1] = x * x;
    correlation_1[k * 4 + 2] = y * y;
    if(x > max1) max1 = x;
    if(y > max1) max1 = y;
    if(x < min1) min1 = x;
    if(y < min1) min1 = y;

    c = (guide + 2) % 3;
    x = in[k * 4 + c];
    y = out_s[k * 4 + c];
    x = fabsf(x - y) / x;
    correlation_2[k * 4] = x * y;
    correlation_2[k * 4 + 1] = x * x;
    correlation_2[k * 4 + 2] = y * y;
    if(x > max2) max2 = x;
    if(y > max2) max2 = y;
    if(x < min2) min2 = x;
    if(y < min2) min2 = y;
  }
  max1 *= max1;
  max2 *= max2;
  min1 *= min1;
  min2 *= min2;

  float max[4] = {max1, max1, max1, 0.0f};
  float min[4] = {min1, min1, min1, 0.0f};
  dt_gaussian_t *g = dt_gaussian_init(width, height, 4, max, min, sigma * 16.0f, 0);
  if(!g) return;
  dt_gaussian_blur_4c(g, correlation_1, blurred_correlation_1);
  dt_gaussian_free(g);

  for(size_t c = 0; c < 4; c++)
  {
    max[c] = max2;
    min[c] = min2;
  }
  g = dt_gaussian_init(width, height, 4, max, min, sigma * 64.0f, 0);
  if(!g) return;
  dt_gaussian_blur_4c(g, correlation_2, blurred_correlation_2);
  dt_gaussian_free(g);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, out_s, out_4s, out_16s, out, blurred_correlation_1, blurred_correlation_2, width, height, guide) \
  schedule(simd:static) aligned(in, out_s, out_4s, out_16s, out, blurred_correlation_1, blurred_correlation_2:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    out[k * 4 + guide] = in[k * 4 + guide];
    out[k * 4 + 3] = in[k * 4 + 3];

    size_t c = (guide + 1) % 3;
    float corr1 = fabsf(blurred_correlation_1[k * 4]) / fmaxf(sqrtf(blurred_correlation_1[k * 4 + 1] * blurred_correlation_1[k * 4 + 2]), 1E-6);
    //printf("%f\n", corr1);
    if(corr1 > 0.7) out[k * 4 + c] = in[k * 4 + c];
    else if(corr1 > 0.6) out[k * 4 + c] = out_s[k * 4 + c];
    else if(corr1 > 0.5) out[k * 4 + c] = out_4s[k * 4 + c];
    else out[k * 4 + c] = out_16s[k * 4 + c];

    c = (guide + 2) % 3;
    float corr2 = fabsf(blurred_correlation_2[k * 4]) / fmaxf(sqrtf(blurred_correlation_2[k * 4 + 1] * blurred_correlation_2[k * 4 + 2]), 1E-6);
    if(corr2 > 0.7) out[k * 4 + c] = in[k * 4 + c];
    else if(corr2 > 0.6) out[k * 4 + c] = out_s[k * 4 + c];
    else if(corr2 > 0.5) out[k * 4 + c] = out_4s[k * 4 + c];
    else out[k * 4 + c] = out_16s[k * 4 + c];
  }

  dt_free_align(out_s);
  dt_free_align(out_4s);
  dt_free_align(out_16s);
  dt_free_align(correlation_1);
  dt_free_align(correlation_2);
  dt_free_align(blurred_correlation_1);
  dt_free_align(blurred_correlation_2);
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
