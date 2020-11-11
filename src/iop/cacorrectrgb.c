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
  // rpatch is the radius of the patch that is used to compare the
  // channels and select the best shift
  const int64_t rpatch = 2;
  // radius of the local box in which we need to find the minimum and maximum
  // to compare patches
  const int64_t radius = iter + rpatch;
  const float* in = (float*)ivoid;
  float* out = (float*)ovoid;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(in, out, width, height, ch, iter, rpatch, radius, guide) \
  schedule(static)
#endif
  for(size_t i = radius; i < height - radius; i++)
  {
    const size_t width_transformed = 2 * radius + 1;
    float* transformed = malloc(sizeof(float) * width_transformed * width_transformed * ch);
    for(size_t j = radius; j < width - radius; j++)
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
      for(size_t ii = i - radius; ii <= i + radius; ii++)
      {
        for(size_t jj = j - radius; jj <= j + radius; jj++)
        {
          for(size_t c = 0; c < 3; c++)
          {
            const float inc = in[(ii * width + jj) * ch + c];
            if(inc < min[c]) min[c] = inc;
            if(inc > max[c]) max[c] = inc;
          }
        }
      }
      if(max[guide] / fmaxf(min[guide], 1E-6) < 1.5f) continue;

      size_t ri = 0;
      for(size_t ii = i - radius; ii <= i + radius; ii++)
      {
        size_t rj = 0;
        for(size_t jj = j - radius; jj <= j + radius; jj++)
        {
          for(size_t c = 0; c < 3; c++)
          {
            const float inc = in[(ii * width + jj) * ch + c];
            // in = a * max + (1-a) * min
            float a = (inc - min[c]) / fmaxf(max[c] - min[c], 1E-6);
            transformed[(ri * width_transformed + rj) * ch + c] = 2.0f * fabs(a - 0.5f);
          }
          rj++;
        }
        ri++;
      }

      for(size_t c = 0; c < ch; c++)
      {
        if(c == guide || c == 4) continue;
        // find best shift in transformed image
        // in transformed image, the center is at coordinate [radius, radius]
        size_t shft_i = radius;
        size_t shft_j = radius;
        for(int k = 0; k < iter; k++)
        {
          if(MAX(fabs((int64_t)shft_i - radius), fabs((int64_t)shft_j - radius)) < k) break;
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
                  float guide_value = transformed[((radius + pi) * width_transformed + (radius + pj)) * ch + guide];
                  float channel_value = transformed[((shft_ii + pi) * width_transformed + (shft_jj + pj)) * ch + c];
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
          if(shft_i == radius && shft_j == radius) break;
        }
        // write output channel
        size_t ci = shft_i + i - radius;
        size_t cj = shft_j + j - radius;
        out[(i * width + j) * ch + c] = in[(ci * width + cj) * ch + c];
      }
    }
  }
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
