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

static void upscale_2x_shift_nn(int* ds_shift, int* shift, const size_t width, const size_t height, const size_t ch)
{
  // upscale shift using nearest neighbour. Multiply all shifts by 2
  size_t ds_width = (width - 1) / 2 + 1;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(ds_shift, shift, ds_width, width, height, ch) \
  schedule(static)
#endif
  for(size_t i = 0; i < height; i++)
  {
    const size_t ds_i = i / 2;
    for(size_t j = 0; j < width; j++)
    {
      const size_t ds_j = j / 2;
      for(size_t c = 0; c < ch; c++)
      {
        shift[(i * width + j) * ch + c] = 2 * ds_shift[(ds_i * ds_width + ds_j) * ch + c];
      }
    }
  }
}

static inline int median(int current, int prev, int next)
{
  int median;
  if (next > current)
  {
    if (next < prev)
    {
      median = next;
    }
    else if (current > prev)
    {
      median = current;
    }
    else
    {
      median = prev;
    }
  }
  else
  {
    if (next > prev)
    {
      median = next;
    }
    else if (current < prev)
    {
      median = current;
    }
    else
    {
      median = prev;
    }
  }
  return median;
}

// RPATCH is the radius of the patch that is used to compare the
// channels and select the best shift
#define RPATCH 10
#define DIAMETERPATCH (2 * RPATCH + 1)
// number of pixels in a patch
#define NBELEMPATCH (DIAMETERPATCH * DIAMETERPATCH)

int comp(const void* p1, const void* p2)
{
  int shft1 = *((int*)p1);
  int shft2 = *((int*)p2);
  if(abs(shft1) < abs(shft2)) return -1;
  if(abs(shft2) < abs(shft1)) return 1;
  return 0;
}

static void compute_shift(const float* const in, int* shift_h, int* shift_v,
                          const size_t width, const size_t height,
                          const size_t ch, const size_t iterations,
                          const dt_iop_cacorrectrgb_guide_channel_t guide,
                          float* const out, gboolean apply_shift)
{

  if(FALSE && iterations > 2)
  {
    size_t ds_width = (width - 1) / 2 + 1;
    size_t ds_height = (height - 1) / 2 + 1;
    float *const restrict ds_in = dt_alloc_sse_ps(dt_round_size_sse(ds_width * ds_height * ch));
    int *const restrict ds_shift_h = dt_alloc_sse_ps(dt_round_size_sse(ds_width * ds_height * ch));
    int *const restrict ds_shift_v = dt_alloc_sse_ps(dt_round_size_sse(ds_width * ds_height * ch));
    interpolate_bilinear(in, width, height, ds_in, ds_width, ds_height, ch);
    compute_shift(ds_in, ds_shift_h, ds_shift_v, ds_width, ds_height, ch, iterations / 2, guide, NULL, FALSE);
    upscale_2x_shift_nn(ds_shift_h, shift_h, width, height, ch);
    upscale_2x_shift_nn(ds_shift_v, shift_v, width, height, ch);
    dt_free_align(ds_in);
    dt_free_align(ds_shift_h);
    dt_free_align(ds_shift_v);
  }

  if(height < RPATCH + iterations) return;
  if(width < RPATCH + iterations) return;

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

  const float sigma = iterations;
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

#ifdef _OPENMP
#pragma omp parallel for default(none) \
dt_omp_firstprivate(in, width, height, iterations, guide, blurred_in, blurred_manifold_higher, blurred_manifold_lower, apply_shift, out) \
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
    }
  }
  dt_free_align(blurred_in);
  dt_free_align(blurred_manifold_lower);
  dt_free_align(blurred_manifold_higher);

  if(!apply_shift)
  {
    // in place median approximation
    // /!\ not thread safe yet.
  #ifdef _OPENMP
  #pragma omp parallel for default(none) \
    dt_omp_firstprivate(shift_v, shift_h, width, height, ch, guide) \
    schedule(static)
  #endif
    for(size_t i = 1; i < height-1; i++)
    {
      for(size_t j = 1; j < width-1; j++)
      {
        for(size_t k = 1; k <= 2; k++)
        {
          size_t c = (guide + k) % 3;
          int currentv = shift_v[(i * width + j) * ch + c];
          int nextv = shift_v[((i + 1) * width + j) * ch + c];
          int prevv = shift_v[((i - 1) * width + j) * ch + c];
          int medianv = median(currentv, nextv, prevv);
          shift_v[(i * width + j) * ch + c] = medianv;
          int currenth = shift_h[(i * width + j) * ch + c];
          int nexth = shift_h[(i * width + (j + 1)) * ch + c];
          int prevh = shift_h[(i * width + (j - 1)) * ch + c];
          int medianh = median(currenth, nexth, prevh);
          shift_h[(i * width + j) * ch + c] = medianh;
        }
      }
    }
  }
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_cacorrectrgb_params_t *d = (dt_iop_cacorrectrgb_params_t *)piece->data;
  const float scale = piece->iscale / roi_in->scale;
  const int ch = piece->colors;
  memcpy(ovoid, ivoid, sizeof(float) * ch * roi_out->height * roi_in->width);
  if(ch != 4 || (piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW) == DT_DEV_PIXELPIPE_PREVIEW
     || (piece->pipe->type & DT_DEV_PIXELPIPE_THUMBNAIL) == DT_DEV_PIXELPIPE_THUMBNAIL)
  {
    return;
  }
  const size_t width = roi_out->width;
  const size_t height = roi_out->height;
  const dt_iop_cacorrectrgb_guide_channel_t guide = d->guide_channel;
  const size_t iter = MAX(d->radius / scale, 1);
  const float* in = (float*)ivoid;
  float* out = (float*)ovoid;
  int* shift_h = calloc(width * height * ch, sizeof(int));
  int* shift_v = calloc(width * height * ch, sizeof(int));

  compute_shift(in, shift_h, shift_v, width, height, ch, iter, guide, out, TRUE);
  dt_free_align(shift_h);
  dt_free_align(shift_v);
  return;
#if 0
  //propagate_shift(); // weighted gaussian blur to propagate the shift to apply
  apply_shift(in, out, shift_map, width, height, ch, iter);
#endif
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
