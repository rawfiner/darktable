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
  float blending; // $MIN: 0.01 $MAX: 100.0 $DEFAULT: 5.0 $DESCRIPTION: "smoothing diameter"
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
  const int ch = piece->colors;
  const size_t width = roi_out->width;
  const size_t height = roi_out->height;
  const float* in = (float*)ivoid;
  float* out = (float*)ovoid;
  memcpy(out, in, width * height * ch * sizeof(float));
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
