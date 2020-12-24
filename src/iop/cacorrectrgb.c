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

  float *const restrict out0_s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out0_4s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out0_16s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out1_s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out1_4s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out1_16s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out2_s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out2_4s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict out2_16s = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict diffs_1 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict diffs_2 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_diffs_1 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_diffs_2 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));

  const dt_iop_cacorrectrgb_guide_channel_t guide = d->guide_channel;
  const float sigma = MAX(d->radius / scale, 1);
  // we compute the correction 3 times and them blend them
  // in order to have an adaptative correction depending on
  // the amount of chromatic aberration in each part of the
  // image
  ca_correct_rgb(in, width, height, ch, sigma, guide, out0_s);
  ca_correct_rgb(in, width, height, ch, 4.0f * sigma, guide, out0_4s);
  ca_correct_rgb(in, width, height, ch, 16.0f * sigma, guide, out0_16s);
  ca_correct_rgb(out0_s, width, height, ch, sigma, (guide + 1) % 3, out1_s);
  ca_correct_rgb(out0_4s, width, height, ch, 4.0f * sigma, (guide + 1) % 3, out1_4s);
  ca_correct_rgb(out0_16s, width, height, ch, 16.0f * sigma, (guide + 1) % 3, out1_16s);
  ca_correct_rgb(out0_s, width, height, ch, sigma, (guide + 2) % 3, out2_s);
  ca_correct_rgb(out0_4s, width, height, ch, 4.0f * sigma, (guide + 2) % 3, out2_4s);
  ca_correct_rgb(out0_16s, width, height, ch, 16.0f * sigma, (guide + 2) % 3, out2_16s);
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, out0_s, out0_4s, out0_16s, out1_s, out1_4s, out1_16s, out2_s, out2_4s, out2_16s, diffs_1, diffs_2, width, height, guide) \
  schedule(simd:static) aligned(in, out0_s, out0_4s, out0_16s, out1_s, out1_4s, out1_16s, out2_s, out2_4s, out2_16s, diffs_1, diffs_2:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float ref_pixelg = fmaxf(in[k * 4 + guide], 1E-6);
    // out[k * 4 + guide] = ref_pixelg;
    float diff1_1 = fmaxf(fabsf(out1_s[k * 4 + guide] - ref_pixelg), 1E-6) / ref_pixelg;
    float diff1_4 = fmaxf(fabsf(out1_4s[k * 4 + guide] - ref_pixelg), 1E-6) / ref_pixelg;
    float diff1_16 = fmaxf(fabsf(out1_16s[k * 4 + guide] - ref_pixelg), 1E-6) / ref_pixelg;
    float w11 = fminf(diff1_1 * diff1_1, 4.0f);
    float w44 = fminf(diff1_4 * diff1_4, 4.0f);
    float w1616 = fminf(diff1_16 * diff1_16, 4.0f);
    diffs_1[k * 4] = w11;
    diffs_1[k * 4 + 1] = w44;
    diffs_1[k * 4 + 2] = w1616;
    // size_t c = (guide + 1) % 3;
    // out[k * 4 + c] = (in[k * 4 + c] * w00 + out0_s[k * 4 + c] * w11 + out0_4s[k * 4 + c] * w44 + out0_16s[k * 4 + c] * w1616) / (w00 + w11 + w44 + w1616);
    float diff2_1 = fmaxf(fabsf(out2_s[k * 4 + guide] - ref_pixelg), 1E-6) / ref_pixelg;
    float diff2_4 = fmaxf(fabsf(out2_4s[k * 4 + guide] - ref_pixelg), 1E-6) / ref_pixelg;
    float diff2_16 = fmaxf(fabsf(out2_16s[k * 4 + guide] - ref_pixelg), 1E-6) / ref_pixelg;
    w11 = fminf(diff2_1 * diff2_1, 4.0f);
    w44 = fminf(diff2_4 * diff2_4, 4.0f);
    w1616 = fminf(diff2_16 * diff2_16, 4.0f);
    diffs_2[k * 4] = w11;
    diffs_2[k * 4 + 1] = w44;
    diffs_2[k * 4 + 2] = w1616;
    // c = (guide + 2) % 3;
    // out[k * 4 + c] = (in[k * 4 + c] * w00 + out0_s[k * 4 + c] * w11 + out0_4s[k * 4 + c] * w44 + out0_16s[k * 4 + c] * w1616) / (w00 + w11 + w44 + w1616);
    // out[k * 4 + 3] = in[k * 4 + 3];
  }
  float max[4] = {4.0f, 4.0f, 4.0f, 0.0f};
  float min[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  dt_gaussian_t *g = dt_gaussian_init(width, height, 4, max, min, sigma * 16.0f, 0);
  if(!g) return;
  dt_gaussian_blur_4c(g, diffs_1, blurred_diffs_1);
  dt_gaussian_blur_4c(g, diffs_2, blurred_diffs_2);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, out0_s, out0_4s, out0_16s, out1_s, out1_4s, out1_16s, out2_s, out2_4s, out2_16s, blurred_diffs_1, blurred_diffs_2, out, width, height, guide) \
  schedule(simd:static) aligned(in, out0_s, out0_4s, out0_16s, out1_s, out1_4s, out1_16s, out2_s, out2_4s, out2_16s, blurred_diffs_1, blurred_diffs_2, out:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float ref_pixelg = fmaxf(in[k * 4 + guide], 1E-6);
    out[k * 4 + guide] = ref_pixelg;
    float w11 = 1.0f / fmaxf(blurred_diffs_1[k * 4], 1E-6);
    float w44 = 1.0f / fmaxf(blurred_diffs_1[k * 4 + 1], 1E-6);
    float w1616 = 1.0f / fmaxf(blurred_diffs_1[k * 4 + 2], 1E-6);
    float w00 = w11;
    w00 *= w00;
    w11 *= w11;
    w44 *= w44;
    w1616 *= w1616;
    size_t c = (guide + 1) % 3;
    out[k * 4 + c] = (in[k * 4 + c] * w00 + out0_s[k * 4 + c] * w11 + out0_4s[k * 4 + c] * w44 + out0_16s[k * 4 + c] * w1616) / (w00 + w11 + w44 + w1616);
    w11 = 1.0f / fmaxf(blurred_diffs_2[k * 4], 1E-6);
    w44 = 1.0f / fmaxf(blurred_diffs_2[k * 4 + 1], 1E-6);
    w1616 = 1.0f / fmaxf(blurred_diffs_2[k * 4 + 2], 1E-6);
    w00 = w11;
    w00 *= w00;
    w11 *= w11;
    w44 *= w44;
    w1616 *= w1616;
    c = (guide + 2) % 3;
    out[k * 4 + c] = (in[k * 4 + c] * w00 + out0_s[k * 4 + c] * w11 + out0_4s[k * 4 + c] * w44 + out0_16s[k * 4 + c] * w1616) / (w00 + w11 + w44 + w1616);
    out[k * 4 + 3] = in[k * 4 + 3];
  }

  dt_free_align(out0_s);
  dt_free_align(out0_4s);
  dt_free_align(out0_16s);
  dt_free_align(out1_s);
  dt_free_align(out1_4s);
  dt_free_align(out1_16s);
  dt_free_align(out2_s);
  dt_free_align(out2_4s);
  dt_free_align(out2_16s);
  dt_free_align(diffs_1);
  dt_free_align(diffs_2);
  dt_free_align(blurred_diffs_1);
  dt_free_align(blurred_diffs_2);
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
