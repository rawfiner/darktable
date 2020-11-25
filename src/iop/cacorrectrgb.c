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
#define RPATCH 4
#define DIAMETERPATCH (2 * RPATCH + 1)
// number of pixels in a patch
#define NBELEMPATCH (DIAMETERPATCH * DIAMETERPATCH)
#define RBILAT 4

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
  const int iter = MIN(iterations, 2);

  if(height < RPATCH + iterations) return;
  if(width < RPATCH + iterations) return;

  float *const restrict log = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));

#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(log, in, height, width, ch) \
  schedule(static)
#endif
  for(int j = 0; j < width * height * ch; j++)
  {
    log[j] = log2f(fmaxf(in[j], 1.0f/1024));
  }

  // find horizontal and vertical shifts
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, log, width, height, ch, iter, iterations, guide, shift_h, shift_v, apply_shift, out) \
  schedule(static)
#endif
  for(size_t i = RPATCH + RBILAT + iterations; i < height - RPATCH - RBILAT - iterations; i++)
  {
    for(size_t j = RPATCH + RBILAT + iterations; j < width - RPATCH - RBILAT - iterations; j++)
    {
      for(size_t kc = 1; kc <= 2; kc++)
      {
        size_t c = (guide + kc) % 3;

        //float res = 0;
        // find best shift in transformed image
        // in transformed image, the center is at coordinate [radius, radius]
        int shft_i = shift_v[(i * width + j) * ch + c];
        int shft_j = shift_h[(i * width + j) * ch + c];
        for(int k = 0; k < /*iter*/1; k++)
        {
          // compute alignment score between guide[radius,radius]
          // and c[shft_ii, shft_jj] with
          // shft_ii and shft_jj in shft_i+[-1,1] and shft_j+[-1,1].
          // then, update shft_i and shft_j with the shft_ii and
          // shft_jj that delivered the best score (i.e. the smaller
          // mean squared error)
          float best_score = 1000000000.0f; // the lower the better
          int best_shft_ii = shft_i;
          int best_shft_jj = shft_j;
          for(int shft_ii = shft_i - iter; shft_ii <= shft_i + iter; shft_ii++)
          {
            for(int shft_jj = shft_j - iter; shft_jj <= shft_j + iter; shft_jj++)
            {
              float total_err_g = 0.0f;
              float total_err_c = 0.0f;
              float total_err_g_ref = 0.0f;
              float total_err_c_ref = 0.0f;
              // compute bilateral filters with small radius
              // guide the filtering of color with the guide, and
              // guide the filtering of guide with color
              for(int pi = -RPATCH; pi <= RPATCH; pi++)
              {
                for(int pj = -RPATCH; pj <= RPATCH; pj++)
                {
                  float log_center_g = log[((i + pi) * width + (j + pj)) * ch + guide];
                  float log_center_c = log[((i + shft_ii + pi) * width + (j + shft_jj + pj)) * ch + c];
                  float center_g = log[((i + pi) * width + (j + pj)) * ch + guide];
                  float center_c = log[((i + shft_ii + pi) * width + (j + shft_jj + pj)) * ch + c];
                  float avg_g = 0.0f;
                  float avg_c = 0.0f;
                  float avg_g2 = 0.0f;
                  float avg_c2 = 0.0f;
                  float sum_w_g = 0.0f;
                  float sum_w_c = 0.0f;
                  int tii[13] = {-RBILAT+1, 0, RBILAT-1, -RBILAT, 0, RBILAT, -RBILAT+1, 0, RBILAT-1, 0, 0, 1, -1};
                  int tjj[13] = {-RBILAT+1, -RBILAT, -RBILAT+1, 0, 0, 0, RBILAT-1, RBILAT, RBILAT-1, 1, -1, 0, 0};
                  for(size_t p = 0; p < 13; p++)
                  {
                    int pii = tii[p];
                    int pjj = tjj[p];
                    float log_iijj_g = log[((i + pi + pii) * width + (j + pj + pjj)) * ch + guide];
                    float log_iijj_c = log[((i + shft_ii + pi + pii) * width + (j + shft_jj + pj + pjj)) * ch + c];
                    float iijj_g = log[((i + pi + pii) * width + (j + pj + pjj)) * ch + guide];
                    float iijj_c = log[((i + shft_ii + pi + pii) * width + (j + shft_jj + pj + pjj)) * ch + c];
                    float w_g = 0.01f / fmaxf(fabsf(log_center_g - log_iijj_g), 0.01f);
                    float w_c = 0.01f / fmaxf(fabsf(log_center_c - log_iijj_c), 0.01f);
                    w_c *= w_c;
                    w_g *= w_g;
                    // w_c *= w_c;
                    // w_g *= w_g;
                    avg_g += w_c * iijj_g;
                    avg_g2 += w_g * iijj_g;
                    sum_w_g += w_c;
                    avg_c += w_g * iijj_c;
                    avg_c2 += w_c * iijj_c;
                    sum_w_c += w_g;
                  }
                  //if(sum_w_g > 9.0f) printf("bilat filter incorrect!\n");
                  //if(sum_w_c > 9.0f) printf("bilat filter incorrect!\n");
                  avg_g /= sum_w_g;
                  avg_c /= sum_w_c;
                  avg_g2 /= sum_w_c;
                  avg_c2 /= sum_w_g;

                  float err_g = center_g - avg_g;
                  total_err_g += err_g * err_g;
                  float err_c = center_c - avg_c;
                  total_err_c += err_c * err_c;
                  err_g = center_g - avg_g2;
                  total_err_g_ref += err_g * err_g;
                  err_c = center_c - avg_c2;
                  total_err_c_ref += err_c * err_c;

                  //if(pi == 0 && pj == 0) res = avg_c;
                }
              }
              float err_squared = /*fmaxf(*/fmaxf(total_err_g, total_err_c)/* - fmaxf(total_err_g_ref, total_err_c_ref), 0.00001f)*/ + sqrtf(shft_i * shft_i + shft_j * shft_j) / 4;
              // lower variance means that we are closer to being able
              // to express the values of channel c as a function of the
              // values of channel guide in our patch.
              //err_squared *= (10 + shft_ii * shft_ii + shft_jj * shft_jj);
              if(err_squared < best_score)
              {
                best_score = err_squared;
                best_shft_ii = shft_ii;
                best_shft_jj = shft_jj;
              }
            }
          }
          // if(shft_i == best_shft_ii && shft_j == best_shft_jj) break;
          shft_i = best_shft_ii;
          shft_j = best_shft_jj;
          // if(shft_i == 0 && shft_j == 0) break;
        }
        shift_v[(i * width + j) * ch + c] = shft_i;
        shift_h[(i * width + j) * ch + c] = shft_j;
        //TODO remove this: (only for testing purpose)
        if(apply_shift)
          out[(i * width + j) * ch + c] = /*exp2f(res);*/in[((i + shft_i) * width + (j + shft_j)) * ch + c];
      }
    }
  }
  dt_free_align(log);

  if(FALSE && !apply_shift)
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
