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
  int radius; // $MIN: 1 $MAX: 20 $DEFAULT: 1 $DESCRIPTION: "radius"
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

static void compute_shift(const float* const in, int* shift_h, int* shift_v,
                          const size_t width, const size_t height,
                          const size_t ch, const size_t iterations,
                          const dt_iop_cacorrectrgb_guide_channel_t guide,
                          float* const out, gboolean apply_shift)
{
  if(iterations > 2)
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
  const size_t iter = MIN(iterations, 2);

  // compute first-order derivatives in horizontal and vertical directions
  // we will use them for alignment
  float *const restrict average = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float maxb = 0.0f;
  float maxg = 0.0f;
  float maxr = 0.0f;
  float minb = 10000000.0f;
  float ming = 10000000.0f;
  float minr = 10000000.0f;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, width, height, ch) \
  schedule(static) \
  reduction(max:maxr, maxg, maxb)\
  reduction(min:minr, ming, minb)
#endif
  for(size_t i = 0; i < height * width; i++)
  {
    const float r = in[i * ch];
    const float g = in[i * ch + 1];
    const float b = in[i * ch + 2];
    if(r < minr) minr = r;
    if(r > maxr) maxr = r;
    if(g < ming) ming = g;
    if(g > maxg) maxg = g;
    if(b < minb) minb = b;
    if(b > maxb) maxb = b;
  }
  const float max[4] = {maxr, maxg, maxb, 0.0f};
  const float min[4] = {minr, ming, minb, 0.0f};
  float sigma = iter;
  dt_gaussian_t *g = dt_gaussian_init(width, height, 4, max, min, sigma, 0);
  if(!g) return;
  dt_gaussian_blur_4c(g, in, average);
  dt_gaussian_free(g);

#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, average, width, height, ch) \
  schedule(static)
#endif
  for(size_t i = 0; i < height * width * ch; i++)
  {
    average[i] = fabsf(in[i] - average[i]) / fmaxf(average[i], 1E-6);
  }

  // find horizontal and vertical shifts
  // rpatch is the radius of the patch that is used to compare the
  // channels and select the best shift
  const int64_t rpatch = 2;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, width, height, ch, iter, iterations, rpatch, guide, average, shift_h, shift_v, apply_shift) \
  schedule(static)
#endif
  for(size_t i = rpatch + iterations; i < height - rpatch - iterations; i++)
  {
    for(size_t j = rpatch + iterations; j < width - rpatch - iterations; j++)
    {
      for(size_t c = 0; c < ch; c++)
      {
        if(c == guide || c == 4) continue;

        //TODO continue if MAX(average[0], average[1], average[2]) is very
        // small

        // find best shift in transformed image
        // in transformed image, the center is at coordinate [radius, radius]
        int shft_i = shift_v[(i * width + j) * ch + c];
        int shft_j = shift_h[(i * width + j) * ch + c];
        for(int k = 0; k < iter; k++)
        {
          // compute alignment score between guide[radius,radius]
          // and c[shft_ii, shft_jj] with
          // shft_ii and shft_jj in shft_i+[-1,1] and shft_j+[-1,1].
          // then, update shft_i and shft_j with the shft_ii and
          // shft_jj that delivered the best score (i.e. the smaller
          // mean squared error)

          // to compute alignment, we use a correlation between the derivatives
          float max_correlation = 0.0f;
          int best_shft_ii = shft_i;
          int best_shft_jj = shft_j;
          for(int shft_ii = shft_i - 1; shft_ii <= shft_i + 1; shft_ii++)
          {
            for(int shft_jj = shft_j - 1; shft_jj <= shft_j + 1; shft_jj++)
            {
              float correlation = 0.0f;
              float gg = 0.0f;
              float cc = 0.0f;
              for(int pi = -rpatch; pi <= rpatch; pi++)
              {
                for(int pj = -rpatch; pj <= rpatch; pj++)
                {
                  float value_ij = average[((i + pi) * width + (j + pj)) * ch + guide];
                  float value_shft_ij = average[((i + shft_ii + pi) * width + (j + shft_jj + pj)) * ch + c];
                  correlation += value_ij * value_shft_ij;
                  gg += value_ij * value_ij;
                  cc += value_shft_ij * value_shft_ij;
                }
              }
              correlation *= correlation;
              correlation /= fmaxf(gg * cc, 1E-6);
              if(correlation > max_correlation)
              {
                max_correlation = correlation;
                best_shft_ii = shft_ii;
                best_shft_jj = shft_jj;
              }
            }
          }
          shft_i = best_shft_ii;
          shft_j = best_shft_jj;
          if(shft_i == i && shft_j == j) break;
        }
        shift_v[(i * width + j) * ch + c] = shft_i;
        shift_h[(i * width + j) * ch + c] = shft_j;
        //TODO remove this: (only for testing purpose)
        //if(apply_shift)
        //  out[(i * width + j) * ch + c] = in[((i + shft_i) * width + (j + shft_j)) * ch + c];
      }
    }
  }

  if(apply_shift)
  {
    float *const restrict shifts = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
    float *const restrict weights = dt_alloc_sse_ps(dt_round_size_sse(width * height * 2));
    float *const restrict blurred_shifts = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
    float *const restrict blurred_weights = dt_alloc_sse_ps(dt_round_size_sse(width * height * 2));
    const size_t c1 = (guide + 1) % 3;
    const size_t c2 = (guide + 2) % 3;
    float maxshift = 0.0f;
    float minshift = 10000000.0f;
    float maxw = 0.0f;
    float minw = 10000000.0f;
  #ifdef _OPENMP
  #pragma omp parallel for default(none) \
    dt_omp_firstprivate(average, weights, shifts, shift_h, shift_v, width, height, ch, guide, c1, c2) \
    schedule(static) \
    reduction(max:maxshift, maxw)\
    reduction(min:minshift, minw)
  #endif
    for(size_t i = 0; i < height * width; i++)
    {
      // float weight = average[i * ch + guide];
      // float weight_c1 = fmaxf(weight, average[i * ch + c1]);
      // float weight_c2 = fmaxf(weight, average[i * ch + c2]);
      float shift_h_c1 = shift_h[i * ch + c1];
      float shift_v_c1 = shift_v[i * ch + c1];
      float weight_c1 = ((fabsf(shift_h_c1) > 0) || (fabsf(shift_v_c1) > 0)) + 0.01f;
      shift_h_c1 *= weight_c1;
      shift_v_c1 *= weight_c1;
      shifts[i * ch] = shift_h_c1;
      shifts[i * ch + 1] = shift_v_c1;
      float shift_h_c2 = shift_h[i * ch + c2];
      float shift_v_c2 = shift_v[i * ch + c2];
      float weight_c2 = ((fabsf(shift_h_c2) > 0) || (fabsf(shift_v_c2) > 0)) + 0.01f;
      shift_h_c2 *= weight_c2;
      shift_v_c2 *= weight_c2;
      shifts[i * ch + 2] = shift_h_c2;
      shifts[i * ch + 3] = shift_v_c2;

      weights[i * 2] = weight_c1;
      weights[i * 2 + 1] = weight_c2;

      if(shift_h_c1 > maxshift) maxshift = shift_h_c1;
      if(shift_h_c1 < minshift) minshift = shift_h_c1;
      if(shift_h_c2 > maxshift) maxshift = shift_h_c2;
      if(shift_h_c2 < minshift) minshift = shift_h_c2;
      if(shift_v_c1 > maxshift) maxshift = shift_v_c1;
      if(shift_v_c1 < minshift) minshift = shift_v_c1;
      if(shift_v_c2 > maxshift) maxshift = shift_v_c2;
      if(shift_v_c2 < minshift) minshift = shift_v_c2;

      if(weight_c1 > maxw) maxw = weight_c1;
      if(weight_c1 < minw) minw = weight_c1;
      if(weight_c2 > maxw) maxw = weight_c2;
      if(weight_c2 < minw) minw = weight_c2;
    }
    const float maxs[4] = {maxshift, maxshift, maxshift, maxshift};
    const float mins[4] = {minshift, minshift, minshift, minshift};
    sigma = 10.0f;
    g = dt_gaussian_init(width, height, 4, maxs, mins, sigma, 0);
    if(!g) return;
    dt_gaussian_blur_4c(g, shifts, blurred_shifts);
    dt_gaussian_free(g);
    const float maxws[2] = {maxw, maxw};
    const float minws[2] = {minw, minw};
    g = dt_gaussian_init(width, height, 2, maxws, minws, sigma, 0);
    if(!g) return;
    dt_gaussian_blur(g, weights, blurred_weights);
    dt_gaussian_free(g);

  #ifdef _OPENMP
  #pragma omp parallel for default(none) \
    dt_omp_firstprivate(blurred_shifts, blurred_weights, width, height) \
    schedule(static)
  #endif
    for(size_t i = 0; i < height * width; i++)
    {
      const float weight_c1 = blurred_weights[i * 2];
      const float weight_c2 = blurred_weights[i * 2 + 1];
      blurred_shifts[i * 4] /= weight_c1;
      blurred_shifts[i * 4 + 1] /= weight_c1;
      blurred_shifts[i * 4 + 2] /= weight_c2;
      blurred_shifts[i * 4 + 3] /= weight_c2;
    }

    // apply final transformation
  #ifdef _OPENMP
  #pragma omp parallel for default(none) \
    dt_omp_firstprivate(blurred_shifts, in, out, width, height, ch, guide) \
    schedule(static)
  #endif
    for(size_t i = 0; i < height * width; i++)
    {
      for(size_t k = 1; k <= 2; k++)
      {
        size_t c = (guide + k) % 3;
        // TODO compute float vertical and horizontal correction, and
        // correct with a 4 points interpolation (cf code of bilin interpol)
        float x = (i % width) + blurred_shifts[i * 4 + (k-1) * 2];
        float y = (i / width) + blurred_shifts[i * 4 + (k-1) * 2 + 1];
        size_t x_prev = (size_t)floorf(x);
        size_t x_next = x_prev + 1;
        size_t y_prev = (size_t)floorf(y);
        size_t y_next = y_prev + 1;

        x_prev = (x_prev < width) ? x_prev : width - 1;
        x_next = (x_next < width) ? x_next : width - 1;
        y_prev = (y_prev < height) ? y_prev : height - 1;
        y_next = (y_next < height) ? y_next : height - 1;

        // Nearest pixels in input array (nodes in grid)
        const size_t Y_prev = y_prev * width;
        const size_t Y_next =  y_next * width;
        const float *const Q_NW = (float *)in + (Y_prev + x_prev) * ch;
        const float *const Q_NE = (float *)in + (Y_prev + x_next) * ch;
        const float *const Q_SE = (float *)in + (Y_next + x_next) * ch;
        const float *const Q_SW = (float *)in + (Y_next + x_prev) * ch;

        // Spatial differences between nodes
        const float Dy_next = (float)y_next - y;
        const float Dy_prev = 1.f - Dy_next; // because next - prev = 1
        const float Dx_next = (float)x_next - x;
        const float Dx_prev = 1.f - Dx_next; // because next - prev = 1

        out[i * ch + c] = Dy_prev * (Q_SW[c] * Dx_next + Q_SE[c] * Dx_prev) +
                          Dy_next * (Q_NW[c] * Dx_next + Q_NE[c] * Dx_prev);
      }
    }
    dt_free_align(blurred_shifts);
    dt_free_align(blurred_weights);
  }
  dt_free_align(average);

#if 0
  //TODO
  median_vert(shift_h);
  median_horiz(shift_v);
#endif
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
