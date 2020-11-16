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

#if 0
static void upscale_2x_shift_nn(int* ds_shift, int* shift, const size_t width, const size_t height)
{
  // upscale shift using nearest neighbour. Multiply all shifts by 2
}
#endif

static void compute_shift(const float* const in, int* shift_h, int* shift_v,
                          const size_t width, const size_t height,
                          const size_t ch, const size_t iterations,
                          const dt_iop_cacorrectrgb_guide_channel_t guide,
                          float* const out)
{
  if(iterations > 2)
  {
    // size_t width_ds = width / 2;
    // size_t height_ds = height / 2;
    // float *const restrict ds_in = dt_alloc_sse_ps(dt_round_size_sse(width_ds * height_ds * ch));
    // int *const restrict ds_shift_h = dt_alloc_sse_ps(dt_round_size_sse(width_ds * height_ds * ch));
    // int *const restrict ds_shift_v = dt_alloc_sse_ps(dt_round_size_sse(width_ds * height_ds * ch));
    // interpolate_bilinear(in, width, height, ds_in, width_ds, height_ds, ch);
    // compute_shift(ds_in, ds_shift_h, ds_shift_v, ds_width, ds_height, ch, iterations / 2);
    // upscale_2x_shift_nn(ds_shift_h, shift_h, width, height);
    // upscale_2x_shift_nn(ds_shift_v, shift_v, width, height);
    // dt_free_align(ds_in);
    // dt_free_align(ds_shift_h);
    // dt_free_align(ds_shift_v);
  }
  const size_t iter = iterations; // MIN(iterations, 2);

  // compute first-order derivatives in horizontal and vertical directions
  // we will use them for alignment
  float *const restrict derivative_h = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict derivative_v = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, width, height, ch, derivative_h) \
  schedule(static)
#endif
  for(size_t i = 0; i < height; i++)
  {
    for(size_t c = 0; c < 3; c++)
    {
      derivative_h[(i * width) * ch + c] = 0.0f;
    }
    for(size_t j = 1; j < width-1; j++)
    {
      for(size_t c = 0; c < 3; c++)
      {
        derivative_h[(i * width + j) * ch + c] = fabsf(in[(i * width + j - 1) * ch + c]
                                               + in[(i * width + j + 1) * ch + c]
                                               - 2.0f * in[(i * width + j) * ch + c]);
      }
    }
    for(size_t c = 0; c < 3; c++)
    {
      derivative_h[(i * width + width - 1) * ch + c] = 0.0f;
    }
  }
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, width, height, ch, derivative_v) \
  schedule(static)
#endif
  for(size_t j = 0; j < width; j++)
  {
    for(size_t c = 0; c < 3; c++)
    {
      derivative_v[j * ch + c] = 0.0f;
    }
    for(size_t i = 1; i < height - 1; i++)
    {
      for(size_t c = 0; c < 3; c++)
      {
        derivative_v[(i * width + j) * ch + c] = fabsf(in[((i - 1) * width + j) * ch + c]
                                               + in[((i + 1) * width + j) * ch + c]
                                               - 2.0f * in[(i * width + j) * ch + c]);
      }
    }
    for(size_t c = 0; c < 3; c++)
    {
      derivative_v[((height - 1) * width + j) * ch + c] = 0.0f;
    }
  }

  // find horizontal and vertical shifts
  // rpatch is the radius of the patch that is used to compare the
  // channels and select the best shift
  const int64_t rpatch = 2;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, width, height, ch, iter, rpatch, guide, derivative_h, derivative_v, shift_h, shift_v, out) \
  schedule(static)
#endif
  for(size_t i = rpatch + iter; i < height - rpatch - iter; i++)
  {
    for(size_t j = rpatch + iter; j < width - rpatch - iter; j++)
    {
      for(size_t c = 0; c < ch; c++)
      {
        if(c == guide || c == 4) continue;
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
              // use reflective correlation
              // TODO try with pearson correlation instead
              float correlation_h = 0.0f;
              float correlation_v = 0.0f;
              float hh_g = 0.0f;
              float hh_c = 0.0f;
              float vv_g = 0.0f;
              float vv_c = 0.0f;
              for(int pi = -rpatch; pi <= rpatch; pi++)
              {
                for(int pj = -rpatch; pj <= rpatch; pj++)
                {
                  float deriv_h_ij = derivative_h[((i + pi) * width + (j + pj)) * ch + guide];
                  float deriv_h_shft_ij = derivative_h[((i + shft_ii + pi) * width + (j + shft_jj + pj)) * ch + c];
                  correlation_h += deriv_h_ij * deriv_h_shft_ij;
                  hh_g += deriv_h_ij * deriv_h_ij;
                  hh_c += deriv_h_shft_ij * deriv_h_shft_ij;
                  float deriv_v_ij = derivative_v[((i + pi) * width + (j + pj)) * ch + guide];
                  float deriv_v_shft_ij = derivative_v[((i + shft_ii + pi) * width + (j + shft_jj + pj)) * ch + c];
                  correlation_v += deriv_v_ij * deriv_v_shft_ij;
                  vv_g += deriv_v_ij * deriv_v_ij;
                  vv_c += deriv_v_shft_ij * deriv_v_shft_ij;
                }
              }
              correlation_h *= correlation_h;
              correlation_v *= correlation_v;
              correlation_h /= fmaxf(hh_g * hh_c, 1E-6);
              correlation_v /= fmaxf(vv_g * vv_c, 1E-6);
              float correlation = correlation_h * correlation_v;
              // printf("%f\n", correlation);
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
        out[(i * width + j) * ch + c] = in[((i + shft_i) * width + (j + shft_j)) * ch + c];
      }
    }
  }

  dt_free_align(derivative_h);
  dt_free_align(derivative_v);
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
  float* transformed = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  int* shift_h = calloc(width * height * ch, sizeof(int));
  int* shift_v = calloc(width * height * ch, sizeof(int));

  compute_shift(in, shift_h, shift_v, width, height, ch, iter, guide, out);
  dt_free_align(shift_h);
  dt_free_align(shift_v);
#if 0
  //propagate_shift(); // weighted gaussian blur to propagate the shift to apply
  apply_shift(in, out, shift_map, width, height, ch, iter);
#endif

  // rpatch is the radius of the patch that is used to compare the
  // channels and select the best shift
  const int64_t rpatch = 2;
  // radius of the local box in which we need to find the minimum and maximum
  // to compare patches
  const int64_t radius = iter + rpatch;
  // radius for min and max search
  const int64_t radiusm = MIN(20, iter);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, width, height, ch, radiusm, guide, transformed) \
  schedule(static)
#endif
  for(size_t i = radiusm; i < height - radiusm; i++)
  {
    for(size_t j = radiusm; j < width - radiusm; j++)
    {
      // find local maximum and minimum on all channels in
      // i+-(radius + rpatch), j+-(radius + rpatch)
      // then, express every pixel as (1 - a) * min + a * max
      // then, compute b = 2.0f * fabs(a - 0.5f)
      // we will use these values to compare channels between them.
      // these values should be close to 1 everywhere except near edges or gradients
      // TODO optimise this.
      float max[3] = {0.0f, 0.0f, 0.0f};
      float min[3] = {1E6f, 1E6f, 1E6f};
      for(size_t ii = i - radiusm; ii <= i + radiusm; ii++)
      {
        for(size_t jj = j - radiusm; jj <= j + radiusm; jj++)
        {
          for(size_t c = 0; c < 3; c++)
          {
            const float inc = in[(ii * width + jj) * ch + c];
            if(inc < min[c]) min[c] = inc;
            if(inc > max[c]) max[c] = inc;
          }
        }
      }
      for(size_t c = 0; c < 3; c++)
      {
        const float inc = in[(i * width + j) * ch + c];
        // in = a * max + (1-a) * min
        float a = (inc - min[c]) / fmaxf(max[c] - min[c], 1E-6);
        transformed[(i * width + j) * ch + c] = -2.0f * fabs(a - 0.5f) + 1.0f;
      }
    }
  }


#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, out, width, height, ch, iter, rpatch, radius, guide, transformed) \
  schedule(static)
#endif
  for(size_t i = radius; i < height - radius; i++)
  {
    for(size_t j = radius; j < width - radius; j++)
    {
      for(size_t c = 0; c < ch; c++)
      {
        if(c == guide || c == 4) continue;
        // find best shift in transformed image
        // in transformed image, the center is at coordinate [radius, radius]
        size_t shft_i = i;
        size_t shft_j = j;
        for(int k = 0; k < iter; k++)
        {
          if(MAX(fabs((int64_t)shft_i - i), fabs((int64_t)shft_j - j)) < k) break;
          // compute alignment score between guide[radius,radius]
          // and c[shft_ii, shft_jj] with
          // shft_ii and shft_jj in shft_i+[-1,1] and shft_j+[-1,1].
          // then, update shft_i and shft_j with the shft_ii and
          // shft_jj that delivered the best score (i.e. the smaller
          // mean squared error)
          // TODO optimise this.
          float min_error = 1E6f;
          float best_shft_ii = shft_i;
          float best_shft_jj = shft_j;
          for(int shft_ii = shft_i - 1; shft_ii <= shft_i + 1; shft_ii++)
          {
            for(int shft_jj = shft_j - 1; shft_jj <= shft_j + 1; shft_jj++)
            {
              float error = 0.0f;
              for(int pi = -rpatch; pi <= rpatch; pi++)
              {
                for(int pj = -rpatch; pj <= rpatch; pj++)
                {
                  float guide_value = transformed[((i + pi) * width + (j + pj)) * ch + guide];
                  float channel_value = transformed[((shft_ii + pi) * width + (shft_jj + pj)) * ch + c];
                  float diff = guide_value - channel_value;
                  error += diff * diff;
                }
              }
              if(error < min_error)
              {
                min_error = error;
                best_shft_ii = shft_ii;
                best_shft_jj = shft_jj;
              }
            }
          }
          shft_i = best_shft_ii;
          shft_j = best_shft_jj;
          if(shft_i == i && shft_j == j) break;
        }
        out[(i * width + j) * ch + c] = in[(shft_i * width + shft_j) * ch + c];
      }
    }
  }
  free(transformed);
//#endif
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
