/*
    This file is part of darktable,
    Copyright (C) 2010-2021 darktable developers.

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


DT_MODULE_INTROSPECTION(1, dt_iop_deblur_params_t)


typedef struct dt_iop_deblur_params_t
{
  int radius; // $MIN: 1 $MAX: 30 $DEFAULT: 1 $DESCRIPTION: "blur radius"
} dt_iop_deblur_params_t;

typedef struct dt_iop_deblur_gui_data_t
{
  GtkWidget *radius;
} dt_iop_deblur_gui_data_t;

const char *name()
{
  return _("deblur");
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

const char *description(struct dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("deblur an image"),
                                      _("corrective"),
                                      _("linear, raw, scene-referred"),
                                      _("linear, raw"),
                                      _("linear, raw, scene-referred"));
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

int legacy_params(dt_iop_module_t *self, const void *const old_params, const int old_version,
                  void *new_params, const int new_version)
{
  return 1;
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, p1, self->params_size);
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  //dt_iop_deblur_params_t *d = (dt_iop_deblur_params_t *)piece->data;
  const size_t ch = piece->colors;
  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  memcpy(ovoid, ivoid, ch * width * height * sizeof(float));
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_deblur_gui_data_t *g = (dt_iop_deblur_gui_data_t *)self->gui_data;
  dt_iop_deblur_params_t *p = (dt_iop_deblur_params_t *)self->params;

  dt_bauhaus_slider_set(g->radius, p->radius);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_deblur_gui_data_t *g = IOP_GUI_ALLOC(deblur);
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  g->radius = dt_bauhaus_slider_from_params(self, "radius");
}

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
