/*
    This file is part of darktable,
    copyright (c) 2020 rawfiner.

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
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <stdlib.h>

// this is the version of the modules parameters,
// and includes version information about compile-time dt
DT_MODULE_INTROSPECTION(2, dt_iop_blind_denoise_params_t)

// TODO: some build system to support dt-less compilation and translation!

typedef struct dt_iop_blind_denoise_params_t
{
  // these are stored in db.
  // make sure everything is in here does not
  // depend on temporary memory (pointers etc)
  // stored in self->params and self->default_params
  // also, since this is stored in db, you should keep changes to this struct
  // to a minimum. if you have to change this struct, it will break
  // users data bases, and you should increment the version
  // of DT_MODULE(VERSION) above!
  int checker_scale;
  float factor;
} dt_iop_blind_denoise_params_t;

typedef struct dt_iop_blind_denoise_gui_data_t
{
  // whatever you need to make your gui happy.
  // stored in self->gui_data
  GtkWidget *scale, *factor; // this is needed by gui_update
} dt_iop_blind_denoise_gui_data_t;

typedef struct dt_iop_blind_denoise_global_data_t
{
} dt_iop_blind_denoise_global_data_t;

const char *name()
{
  return _("blinddenoise");
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING;
}

int default_group()
{
  return IOP_GROUP_CORRECT;
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

// decompose image in 2 layers: each pixel of out is a 4 pixels mean, and scaling up 2x out and adding details gives back in
// out width is (width+1)/2
// out height is (height+1)/2
static void decompose(const float* in, float* out, float* details, unsigned width, unsigned height)
{
  const unsigned widthout = (width + 1) / 2;
  for(unsigned j = 0; j < height; j+=2)
  {
    unsigned jout = (j+1)/2;
    for(unsigned i = 0; i < width; i+=2)
    {
      unsigned iout =(i+1)/2;
      for(unsigned c = 0; c < 3; c++)
      {
        float tmp00 = in[(j*width+i)*4+c];
        float tmp01 = in[(j*width+MIN(i+1,width-1))*4+c];
        float tmp10 = in[(MIN(j+1,height-1)*width+i)*4+c];
        float tmp11 = in[(MIN(j+1,height-1)*width+MIN(i+1,width-1))*4+c];
        float mean = (tmp00 + tmp01 + tmp10 + tmp11) / 4.0f;
        out[(jout * widthout + iout) * 4 + c] = mean;
        details[(j*width+i)*4+c] = tmp00 - mean;
        details[(j*width+MIN(i+1,width-1))*4+c] = tmp01 - mean;
        details[(MIN(j+1,height-1)*width+i)*4+c] = tmp10 - mean;
        details[(MIN(j+1,height-1)*width+MIN(i+1,width-1))*4+c] = tmp11 - mean;
      }
    }
  }
}

// recompose image from 2 layers
// width and height are the dimensions of out
static void recompose(float* in, float* out, float* details, unsigned width, unsigned height)
{
  const unsigned widthin = (width + 1) / 2;
  for(unsigned j = 0; j < height; j+=2)
  {
    unsigned jin = (j+1)/2;
    for(unsigned i = 0; i < width; i+=2)
    {
      unsigned iin =(i+1)/2;
      for(unsigned c = 0; c < 3; c++)
      {
        float mean = in[(jin * widthin + iin) * 4 + c];
        out[(j*width+i)*4+c] = mean + details[(j*width+i)*4+c];
        out[(j*width+MIN(i+1,width-1))*4+c] = mean + details[(j*width+MIN(i+1,width-1))*4+c];
        out[(MIN(j+1,height-1)*width+i)*4+c] = mean + details[(MIN(j+1,height-1)*width+i)*4+c];
        out[(MIN(j+1,height-1)*width+MIN(i+1,width-1))*4+c] = mean + details[(MIN(j+1,height-1)*width+MIN(i+1,width-1))*4+c];
      }
    }
  }
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  // dt_iop_blind_denoise_params_t *d = (dt_iop_blind_denoise_params_t *)piece->data;
  float* out = (float*)ovoid;
  const float* in = (float*)ivoid;
  const unsigned width = roi_out->width;
  const unsigned height = roi_out->height;
  float* tmp = (float*)malloc(sizeof(float) * 4 * width * height);
  float* tmp1 = (float*)malloc(sizeof(float) * 4 * width * height);
  float* tmp2 = (float*)calloc(sizeof(float), 4 * width * height);
  decompose(in, tmp, tmp1, width, height);
  recompose(tmp, out, tmp2, width, height);
  free(tmp);
  free(tmp1);
  free(tmp2);
}

void reload_defaults(dt_iop_module_t *module)
{
}

void init(dt_iop_module_t *module)
{
  module->global_data = NULL;
  module->params = calloc(1, sizeof(dt_iop_blind_denoise_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_blind_denoise_params_t));
  // our module is disabled by default
  // by default:
  module->default_enabled = 0;
  module->params_size = sizeof(dt_iop_blind_denoise_params_t);
  module->gui_data = NULL;
  // init defaults:
  dt_iop_blind_denoise_params_t tmp = (dt_iop_blind_denoise_params_t){ .checker_scale = 50, .factor = 0.5 };

  memcpy(module->params, &tmp, sizeof(dt_iop_blind_denoise_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_blind_denoise_params_t));
}

void init_global(dt_iop_module_so_t *module)
{
  module->data = malloc(sizeof(dt_iop_blind_denoise_global_data_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
  free(module->default_params);
  module->default_params = NULL;
}

void cleanup_global(dt_iop_module_so_t *module)
{
  free(module->data);
  module->data = NULL;
}

static void scale_callback(GtkWidget *w, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_blind_denoise_params_t *p = (dt_iop_blind_denoise_params_t *)self->params;
  p->checker_scale = dt_bauhaus_slider_get(w);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void factor_callback(GtkWidget *w, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_blind_denoise_params_t *p = (dt_iop_blind_denoise_params_t *)self->params;
  p->factor = dt_bauhaus_slider_get(w);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_blind_denoise_gui_data_t *g = (dt_iop_blind_denoise_gui_data_t *)self->gui_data;
  dt_iop_blind_denoise_params_t *p = (dt_iop_blind_denoise_params_t *)self->params;
  dt_bauhaus_slider_set(g->scale, p->checker_scale);
  dt_bauhaus_slider_set(g->factor, p->factor);
}

void gui_init(dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_blind_denoise_gui_data_t));
  dt_iop_blind_denoise_gui_data_t *g = (dt_iop_blind_denoise_gui_data_t *)self->gui_data;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  g->scale = dt_bauhaus_slider_new_with_range(self, 1, 100, 1, 50, 0);
  dt_bauhaus_widget_set_label(g->scale, NULL, _("size"));
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->scale), TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->scale), "value-changed", G_CALLBACK(scale_callback), self);

  g->factor = dt_bauhaus_slider_new_with_range(self, 0.0, 1.0, 0.1, 0.5, 2);
  dt_bauhaus_widget_set_label(g->factor, NULL, _("factor"));
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->factor), TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->factor), "value-changed", G_CALLBACK(factor_callback), self);
}

void gui_cleanup(dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
