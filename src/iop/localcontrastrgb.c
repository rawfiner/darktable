/*
    This file is part of darktable,
    Copyright (C) 2021 darktable developers.

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

#include "bauhaus/bauhaus.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include "common/eigf.h"
#include "common/rgb_norms.h"

#include <gtk/gtk.h>
#include <stdlib.h>

DT_MODULE_INTROSPECTION(1, dt_iop_localcontrastrgb_params_t)

typedef struct dt_iop_localcontrastrgb_params_t
{
  dt_iop_rgb_norms_t norm; // $DEFAULT: DT_RGB_NORM_LUMINANCE $DESCRIPTION: "rgb norm"
  float blending; // $MIN: 0.01 $MAX: 1000.0 $DEFAULT: 5.0 $DESCRIPTION: "smoothing diameter"
  float feathering; // $MIN: 0.01 $MAX: 10000.0 $DEFAULT: 1.0 $DESCRIPTION: "edges refinement/feathering"
  float strength; // $MIN: 0 $MAX: 400 $DEFAULT: 100 $DESCRIPTION: "strength"
} dt_iop_localcontrastrgb_params_t;

typedef struct dt_iop_localcontrastrgb_gui_data_t
{
  GtkWidget *norm, *blending, *feathering, *strength;
} dt_iop_localcontrastrgb_gui_data_t;

const char *name()
{
  return _("local contrast rgb");
}

const char *description(struct dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("enhance local contrast"),
                                      _("creative"),
                                      _("linear, raw, scene-referred"),
                                      _("linear, raw"),
                                      _("linear, raw, scene-referred"));
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

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
  dt_iop_localcontrastrgb_params_t *d = (dt_iop_localcontrastrgb_params_t *)self->params;
  const int ch = piece->colors;
  const size_t width = roi_out->width;
  const size_t height = roi_out->height;
  const float* in = (float*)ivoid;
  float* out = (float*)ovoid;
  const dt_iop_rgb_norms_t norm = d->norm;
  const float strength = d->strength / 100.0f;
  const dt_iop_order_iccprofile_info_t *const work_profile = dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  const int max_size = (piece->iwidth > piece->iheight) ? piece->iwidth : piece->iheight;
  const float scale = piece->iscale / roi_in->scale;
  const float diameter = d->blending / scale + 0.0f * max_size * roi_in->scale;
  const float radius = (diameter - 1.0f) / 2.0f;
  const float feathering = 1.0f / d->feathering;

  float *const restrict ratios = dt_alloc_align_float(width * height * ch);
  float *const restrict norm_in = dt_alloc_align_float(width * height);
  float *const restrict norm_blurred = dt_alloc_align_float(width * height);

  // compute rgb norm, and rgb ratios
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, norm_in, norm_blurred, ratios, norm, work_profile, width, height, ch) \
  schedule(simd:static) aligned(in, norm_in, norm_blurred, ratios:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float n = fmaxf(dt_rgb_norm(in + 4 * k, norm, work_profile), 1E-6);
    norm_in[k] = n;
    norm_blurred[k] = n;
    //TODO: use forall loop
    for(size_t c = 0; c < 3; c++)
    {
      ratios[k * 4 + c] = in[k * 4 + c] / n;
    }
  }

  // eigf
  fast_eigf_surface_blur(norm_blurred, width, height, radius, feathering, 1,
                 DT_GF_BLENDING_LINEAR, 1.0f, 0.0f, exp2f(-14.0f), 4.0f);

  // compute (log?) difference between norm and eigf
  // amplify difference by strength
  // add it back to eigf result
  // compute rgb image from this and rgb ratios
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, norm_in, norm_blurred, ratios, out, width, height, ch, strength) \
  schedule(simd:static) aligned(in, norm_in, norm_blurred, out, ratios:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float result_norm = exp2f(log2f(norm_in[k] / norm_blurred[k]) * strength) * norm_blurred[k];
    //const float result_norm = (norm_in[k] - norm_blurred[k]) * strength + norm_blurred[k];
    //TODO: use forall loop
    for(size_t c = 0; c < 3; c++)
    {
      out[k * 4 + c] = ratios[k * 4 + c] * result_norm;
    }
    out[k * 4 + 3] = in[k * 4 + 3];
  }


  //memcpy(out, in, width * height * ch * sizeof(float));
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_localcontrastrgb_gui_data_t *g = (dt_iop_localcontrastrgb_gui_data_t *)self->gui_data;
  dt_iop_localcontrastrgb_params_t *p = (dt_iop_localcontrastrgb_params_t *)self->params;

  dt_bauhaus_combobox_set_from_value(g->norm, p->norm);
  dt_bauhaus_slider_set_soft(g->blending, p->blending);
  dt_bauhaus_slider_set_soft(g->feathering, p->feathering);
  dt_bauhaus_slider_set_soft(g->strength, p->strength);
}

void reload_defaults(dt_iop_module_t *module)
{
  dt_iop_localcontrastrgb_params_t *d = (dt_iop_localcontrastrgb_params_t *)module->default_params;

  d->norm = DT_RGB_NORM_LUMINANCE;
  d->blending = 5.0f;
  d->feathering = 1.0f;
  d->strength = 100.0f;

  dt_iop_localcontrastrgb_gui_data_t *g = (dt_iop_localcontrastrgb_gui_data_t *)module->gui_data;
  if(g)
  {
    dt_bauhaus_combobox_set_default(g->norm, d->norm);
    dt_bauhaus_slider_set_default(g->blending, d->blending);
    dt_bauhaus_slider_set_soft_range(g->blending, 1.0, 20.0);
    dt_bauhaus_slider_set_default(g->feathering, d->feathering);
    dt_bauhaus_slider_set_soft_range(g->feathering, 0.1, 50.0);
    dt_bauhaus_slider_set_default(g->strength, d->strength);
  }
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_localcontrastrgb_gui_data_t *g = IOP_GUI_ALLOC(localcontrastrgb);
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);
  g->norm = dt_bauhaus_combobox_from_params(self, "norm");
  gtk_widget_set_tooltip_text(g->norm, _("method to preserve colors when applying local contrast"));
  g->blending = dt_bauhaus_slider_from_params(self, "blending");
  gtk_widget_set_tooltip_text(g->blending, _("blending of the blur to use"));
  g->feathering = dt_bauhaus_slider_from_params(self, "feathering");
  gtk_widget_set_tooltip_text(g->feathering, _("precision of the feathering :\n"
                                               "increase in case of halos"));
  g->strength = dt_bauhaus_slider_from_params(self, "strength");
  gtk_widget_set_tooltip_text(g->strength, _("strength of the effect"));
}
