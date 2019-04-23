/*
    This file is part of darktable,

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
#include "common/darktable.h"
#include "common/guided_filter.h"
#include "develop/imageop.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <stdlib.h>

DT_MODULE_INTROSPECTION(1, dt_iop_fastdenoise_params_t)

typedef struct dt_iop_fastdenoise_params_t
{
  unsigned radius;
  float strength;
  float strengthluma;
} dt_iop_fastdenoise_params_t;

typedef struct dt_iop_fastdenoise_gui_data_t
{
  GtkWidget *radius;
  GtkWidget *strength;
  GtkWidget *strengthluma;
} dt_iop_fastdenoise_gui_data_t;

const char *name()
{
  return _("guided denoise");
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

int default_group()
{
  return IOP_GROUP_CORRECT;
}

int default_colorspace(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  return iop_cs_rgb;
}

/** process, all real work is done here. */
void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_fastdenoise_params_t *d = (dt_iop_fastdenoise_params_t *)piece->data;
  //const float scale = piece->iscale / roi_in->scale;
  //const int ch = piece->colors;
  const float wb[3] = {piece->pipe->dsc.temperature.coeffs[0], piece->pipe->dsc.temperature.coeffs[1]
                         , piece->pipe->dsc.temperature.coeffs[2]};
  float *norm = dt_alloc_align(64, (size_t)4 * sizeof(float) * roi_in->width * roi_in->height);
  float *normluma = dt_alloc_align(64, (size_t)4 * sizeof(float) * roi_in->width * roi_in->height);
  float *norm1 = dt_alloc_align(64, (size_t)sizeof(float) * roi_in->width * roi_in->height);
  const float *in = (float*)ivoid;
  float *rratios = dt_alloc_align(64, (size_t)sizeof(float) * roi_in->width * roi_in->height);
  float *gratios = dt_alloc_align(64, (size_t)sizeof(float) * roi_in->width * roi_in->height);
  float *bratios = dt_alloc_align(64, (size_t)sizeof(float) * roi_in->width * roi_in->height);
  const float thrs = 0.000001f;

  #ifdef _OPENMP
  #pragma omp parallel for default(none) schedule(static) shared(d) firstprivate(norm, normluma, norm1, rratios, gratios, bratios, in)
  #endif
  for(int j = 0; j < 4*roi_out->height*roi_out->width; j+=4)
  {
    float inr = in[j];
    float ing = in[j+1];
    float inb = in[j+2];
    if(inr < 0.0f) inr = 0.0f;
    if(ing < 0.0f) ing = 0.0f;
    if(inb < 0.0f) inb = 0.0f;
    float inr2 = powf(inr/wb[0], 1.0);//d->strengthluma / 1000.0);
    float ing2 = powf(ing/wb[1], 1.0);//d->strengthluma / 1000.0);
    float inb2 = powf(inb/wb[2], 1.0);//d->strengthluma / 1000.0);
    float res = inr2*(inr/wb[0]+thrs)+ing2*(ing/wb[1]+thrs)+inb2*(inb/wb[2]+thrs);
    res /= (inr/wb[0]+ing/wb[1]+inb/wb[2]+3*thrs);
    // float res = inr/wb[0]*(inr/wb[0]+thrs)+ing/wb[1]*(ing/wb[1]+thrs)+inb/wb[2]*(inb/wb[2]+thrs);
    // res /= (inr/wb[0]+ing/wb[1]+inb/wb[2]+3*thrs);
    float max = inr/wb[0];
    if(ing/wb[1] > max)
      max = ing/wb[1];
    if(inb/wb[2] > max)
      max = inb/wb[2];
    if(max < thrs)
      max = thrs;
    float res2 = inr2*(inr2/wb[0]+thrs)+ing2*(ing2/wb[1]+thrs)+inb2*(inb2/wb[2]+thrs);
    res2 /= (inr2/wb[0]+ing2/wb[1]+inb2/wb[2]+3*thrs);
    float res3 = inr2+ing2+inb2;
    res3 /= 3;
    if (res < thrs)
      res = thrs;
    if (res2 < thrs)
      res2 = thrs;
    if(res3 < thrs)
      res3 = thrs;
    //res *= 10.0;
    //res2 *= 3.0;
    //res3 *= 10.0;
    //max *= 10.0;
    norm[j]   = inr/res;//inr /wb[0] / d->strengthluma;
    norm[j+1] = inb/res;//ing /wb[1] / d->strengthluma;
    norm[j+2] = ing/res;//inb /wb[2] / d->strengthluma;
    norm[j+3] = max/res;
    norm1[j/4] = res;
    normluma[j]   = pow(res,0.7);//inr /wb[0] / d->strengthluma;
    normluma[j+1] = pow(res2,0.7);//ing /wb[1] / d->strengthluma;
    normluma[j+2] = pow(res3,0.7);//inb /wb[2] / d->strengthluma;
    normluma[j+3] = pow(max,0.7);
    rratios[j/4] = inr/wb[0] / res;
    gratios[j/4] = ing/wb[1] / res;
    bratios[j/4] = inb/wb[2] / res;
  }

  float *rratios_out = dt_alloc_align(64, (size_t)sizeof(float) * roi_in->width * roi_in->height);
  float *gratios_out = dt_alloc_align(64, (size_t)sizeof(float) * roi_in->width * roi_in->height);
  float *bratios_out = dt_alloc_align(64, (size_t)sizeof(float) * roi_in->width * roi_in->height);
  float *norm1_out = dt_alloc_align(64, (size_t)sizeof(float) * roi_in->width * roi_in->height);

  const float w = 1.0f;
  guided_filter(norm, rratios, rratios_out, roi_in->width, roi_in->height, 4, d->radius,
                    d->strength / 500.0, w, 0.0f, 1000000000.0f);
  guided_filter(norm, gratios, gratios_out, roi_in->width, roi_in->height, 4, d->radius,
                    d->strength / 500.0, w, 0.0f, 1000000000.0f);
  guided_filter(norm, bratios, bratios_out, roi_in->width, roi_in->height, 4, d->radius,
                    d->strength / 500.0, w, 0.0f, 1000000000.0f);
  guided_filter(normluma, norm1, norm1_out, roi_in->width, roi_in->height, 4, d->radius,
                    d->strengthluma / 10.0, 1000.0, 0.0f, 1000000000.0f);

  g_free(rratios);
  g_free(gratios);
  g_free(bratios);
  g_free(norm1);

  #ifdef _OPENMP
  #pragma omp parallel for default(none) schedule(static) shared(d) firstprivate(norm1_out, rratios_out, gratios_out, bratios_out, in)
  #endif
  for(int j = 0; j < 4*roi_out->height*roi_out->width; j+=4)
  {
    float* out = (float*)ovoid;
    out[j] = rratios_out[j/4]*wb[0]*norm1_out[j/4];
    out[j+1] = gratios_out[j/4]*wb[1]*norm1_out[j/4];
    out[j+2] = bratios_out[j/4]*wb[2]*norm1_out[j/4];
    out[j+3] = in[j+3];
  }
  g_free(rratios_out);
  g_free(gratios_out);
  g_free(bratios_out);
  g_free(norm1_out);
  g_free(norm);
  g_free(normluma);
}

// void reload_defaults(dt_iop_module_t *module)
// {
//   //TODO
// }

void init(dt_iop_module_t *module)
{
  // we don't need global data:
  module->data = calloc(1, sizeof(dt_iop_fastdenoise_params_t)); // malloc(sizeof(dt_iop_useless_global_data_t));
  module->params = calloc(1, sizeof(dt_iop_fastdenoise_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_fastdenoise_params_t));
  module->params_size = sizeof(dt_iop_fastdenoise_params_t);
  module->gui_data = NULL;

  dt_iop_fastdenoise_params_t tmp = (dt_iop_fastdenoise_params_t){ 15, 1, 1 };

  memcpy(module->params, &tmp, sizeof(dt_iop_fastdenoise_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_fastdenoise_params_t));
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *params, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_fastdenoise_params_t *p = (dt_iop_fastdenoise_params_t *)params;
  dt_iop_fastdenoise_params_t *d = (dt_iop_fastdenoise_params_t *)piece->data;
  d->radius = p->radius;
  d->strength = p->strength;
  d->strengthluma = p->strengthluma;
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

static void radius_callback(GtkWidget *w, dt_iop_module_t *self)
{
  // this is important to avoid cycles!
  if(darktable.gui->reset) return;
  dt_iop_fastdenoise_params_t *p = (dt_iop_fastdenoise_params_t *)self->params;
  p->radius = dt_bauhaus_slider_get(w);
  // let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void strength_callback(GtkWidget *w, dt_iop_module_t *self)
{
  // this is important to avoid cycles!
  if(darktable.gui->reset) return;
  dt_iop_fastdenoise_params_t *p = (dt_iop_fastdenoise_params_t *)self->params;
  p->strength = dt_bauhaus_slider_get(w);
  // let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void strengthluma_callback(GtkWidget *w, dt_iop_module_t *self)
{
  // this is important to avoid cycles!
  if(darktable.gui->reset) return;
  dt_iop_fastdenoise_params_t *p = (dt_iop_fastdenoise_params_t *)self->params;
  p->strengthluma = dt_bauhaus_slider_get(w);
  // let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_update(dt_iop_module_t *self)
{
  // let gui slider match current parameters:
  dt_iop_fastdenoise_gui_data_t *g = (dt_iop_fastdenoise_gui_data_t *)self->gui_data;
  dt_iop_fastdenoise_params_t *p = (dt_iop_fastdenoise_params_t *)self->params;
  dt_bauhaus_slider_set(g->radius, p->radius);
  dt_bauhaus_slider_set(g->strength, p->strength);
  dt_bauhaus_slider_set(g->strengthluma, p->strengthluma);
}

void gui_init(dt_iop_module_t *self)
{
  // init the slider (more sophisticated layouts are possible with gtk tables and boxes):
  self->gui_data = malloc(sizeof(dt_iop_fastdenoise_gui_data_t));
  GtkWidget* widget = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));
  dt_iop_fastdenoise_gui_data_t *g = (dt_iop_fastdenoise_gui_data_t *)self->gui_data;
  g->radius = dt_bauhaus_slider_new_with_range(self, 0.0f, 80.0f, 1.f, 15.f, 0);
  dt_bauhaus_widget_set_label(g->radius, NULL, _("radius"));
  g_signal_connect(G_OBJECT(g->radius), "value-changed", G_CALLBACK(radius_callback), self);
  g->strength = dt_bauhaus_slider_new_with_range(self, 0.001f, 1000.0f, .05, 1.f, 3);
  dt_bauhaus_widget_set_label(g->strength, NULL, _("color denoise"));
  g_signal_connect(G_OBJECT(g->strength), "value-changed", G_CALLBACK(strength_callback), self);
  g->strengthluma = dt_bauhaus_slider_new_with_range(self, 0.001f, 1000.0f, .05, 1.f, 3);
  dt_bauhaus_widget_set_label(g->strengthluma, NULL, _("luma-like norm denoise"));
  g_signal_connect(G_OBJECT(g->strengthluma), "value-changed", G_CALLBACK(strengthluma_callback), self);
  gtk_box_pack_start(GTK_BOX(widget), GTK_WIDGET(g->radius), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(widget), GTK_WIDGET(g->strength), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(widget), GTK_WIDGET(g->strengthluma), TRUE, TRUE, 0);
  gtk_widget_show_all(widget);
  self->widget = widget;
}

void gui_cleanup(dt_iop_module_t *self)
{
  // nothing else necessary, gtk will clean up the slider.
  free(self->gui_data);
  self->gui_data = NULL;
}
