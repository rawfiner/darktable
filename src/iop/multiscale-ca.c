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
#include "develop/imageop.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <stdlib.h>

DT_MODULE_INTROSPECTION(1, dt_iop_msca_params_t)

typedef enum dt_iop_msca_guiding_channel_t
{
  RED = 0,
  GREEN = 1,
  BLUE = 2
} dt_iop_msca_guiding_channel_t;

typedef struct dt_iop_msca_params_t
{
  unsigned nb_of_scales;
  float edge_threshold;
  float correction_margin;
} dt_iop_msca_params_t;

typedef struct dt_iop_msca_gui_data_t
{
  GtkWidget *nb_of_scales;
  GtkWidget *edge_threshold;
  GtkWidget *correction_margin;
} dt_iop_msca_gui_data_t;

const char *name()
{
  return _("msca");
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

static void ca_correct(float* inout, float threshold, unsigned width, unsigned height)
{
  // in all this function, variables named with a 0 refer to the guiding channel,
  // variables named with a 1 to the first following channel, and variables named
  // with a 2 to the second following channel.
  for (int i = 0; i < height; ++i)
  {
    for (int j = 1; j < width - 2; ++j)
    {
      float edge_ratio = inout[(i * width + j + 1) * 4 + 1] / inout[(i * width + j - 1) * 4 + 1];
      gboolean inverse = FALSE;
      if (edge_ratio < threshold)
      {
        edge_ratio = 1.0f / edge_ratio;
        inverse = TRUE;
      }

      if (edge_ratio >= threshold)
      {
        // find edge boundaries
        int left_bound;
        int right_bound;
        for (left_bound = j-1; left_bound > 0; left_bound--)
        {
          float ratio0 = (inout[(i * width + left_bound + 1) * 4 + 1] / inout[(i * width + left_bound - 1) * 4 + 1]);
          float ratio1 = (inout[(i * width + left_bound + 1) * 4 + 2] / inout[(i * width + left_bound - 1) * 4 + 2]);
          float ratio2 = (inout[(i * width + left_bound + 1) * 4 + 0] / inout[(i * width + left_bound - 1) * 4 + 0]);
          if(inverse)
          {
            ratio0 = 1.0f / ratio0;
            ratio1 = 1.0f / ratio1;
            ratio2 = 1.0f / ratio2;
          }
          if(MAX(MAX(ratio1, ratio0), ratio2) < threshold) break;
          if(j - left_bound > 10) break;
        }
        for (right_bound = j+1; right_bound < width - 2; right_bound++)
        {
          float ratio0 = (inout[(i * width + right_bound + 1) * 4 + 1] / inout[(i * width + right_bound - 1) * 4 + 1]);
          float ratio1 = (inout[(i * width + right_bound + 1) * 4 + 2] / inout[(i * width + right_bound - 1) * 4 + 2]);
          float ratio2 = (inout[(i * width + right_bound + 1) * 4 + 0] / inout[(i * width + right_bound - 1) * 4 + 0]);
          if(inverse)
          {
            ratio0 = 1.0f / ratio0;
            ratio1 = 1.0f / ratio1;
            ratio2 = 1.0f / ratio2;
          }
          if(MAX(MAX(ratio1, ratio0), ratio2) < threshold) break;
          if(right_bound - j > 10) break;
        }

        // find maximum ratio between guiding and following channels at boundaries
        float left_ratio_1_0 = inout[(i * width + left_bound) * 4 + 2] / inout[(i * width + left_bound) * 4 + 1];
        float right_ratio_1_0 = inout[(i * width + right_bound) * 4 + 2] / inout[(i * width + right_bound) * 4 + 1];
        if(left_ratio_1_0 < 1.0f) left_ratio_1_0 = 1.0f / left_ratio_1_0;
        if(right_ratio_1_0 < 1.0f) right_ratio_1_0 = 1.0f / right_ratio_1_0;
        float max_ratio_1_0 = MAX(left_ratio_1_0, right_ratio_1_0);
        float inv_max_ratio_1_0 = 1.0f / max_ratio_1_0;

        float left_ratio_2_0 = inout[(i * width + left_bound) * 4 + 0] / inout[(i * width + left_bound) * 4 + 1];
        float right_ratio_2_0 = inout[(i * width + right_bound) * 4 + 0] / inout[(i * width + right_bound) * 4 + 1];
        if(left_ratio_2_0 < 1.0f) left_ratio_2_0 = 1.0f / left_ratio_2_0;
        if(right_ratio_2_0 < 1.0f) right_ratio_2_0 = 1.0f / right_ratio_2_0;
        float max_ratio_2_0 = MAX(left_ratio_2_0, right_ratio_2_0);
        float inv_max_ratio_2_0 = 1.0f / max_ratio_2_0;

        for (int k = left_bound; k <= right_bound; ++k)
        {
          float ratio_1_0 = inout[(i * width + k) * 4 + 2] / inout[(i * width + k) * 4 + 1];
          float ratio_2_0 = inout[(i * width + k) * 4 + 0] / inout[(i * width + k) * 4 + 1];

          float lead = inout[(i * width + k) * 4 + 1];
          float follow1 = inout[(i * width + k) * 4 + 2];
          float follow2 = inout[(i * width + k) * 4 + 0];

          // fix the following channels values if the ratios between
          // them and the leading channel are too big
          if(ratio_1_0 > max_ratio_1_0)
          {
            inout[(i * width + k) * 4 + 2] = max_ratio_1_0 * lead;
          }
          else if(ratio_1_0 < inv_max_ratio_1_0)
          {
            inout[(i * width + k) * 4 + 2] = inv_max_ratio_1_0 * lead;
          }
          else
          {
            inout[(i * width + k) * 4 + 2] = follow1;
          }

          if(ratio_2_0 > max_ratio_2_0)
          {
            inout[(i * width + k) * 4 + 0] = max_ratio_2_0 * lead;
          }
          else if(ratio_2_0 < inv_max_ratio_2_0)
          {
            inout[(i * width + k) * 4 + 0] = inv_max_ratio_2_0 * lead;
          }
          else
          {
            inout[(i * width + k) * 4 + 0] = follow2;
          }
        }
        j = right_bound;
      }
    }
  }
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
            const void *const ivoid, void *const ovoid, const dt_iop_roi_t *const roi_in,
            const dt_iop_roi_t *const roi_out)
{
  const dt_iop_msca_params_t *const d = piece->data;
  size_t img_size = (size_t)4 * sizeof(float) * roi_in->width * roi_in->height;
  memcpy(ovoid, ivoid, img_size);
  ca_correct(ovoid, 1.0f + d->edge_threshold / 20.0f, roi_in->width, roi_in->height);
}

// void reload_defaults(dt_iop_module_t *module)
// {
//   //TODO
// }

void init(dt_iop_module_t *module)
{
  module->params = calloc(1, sizeof(dt_iop_msca_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_msca_params_t));
  module->params_size = sizeof(dt_iop_msca_params_t);
  module->gui_data = NULL;

  dt_iop_msca_params_t tmp = (dt_iop_msca_params_t){ 2, 1, 0 };

  memcpy(module->params, &tmp, sizeof(dt_iop_msca_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_msca_params_t));
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *params, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_msca_params_t *p = (dt_iop_msca_params_t *)params;
  dt_iop_msca_params_t *d = (dt_iop_msca_params_t *)piece->data;
  d->nb_of_scales = p->nb_of_scales;
  d->edge_threshold = p->edge_threshold;
  d->correction_margin = p->correction_margin;
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

static void nb_of_scales_callback(GtkWidget *w, dt_iop_module_t *self)
{
  // this is important to avoid cycles!
  if(darktable.gui->reset) return;
  dt_iop_msca_params_t *p = (dt_iop_msca_params_t *)self->params;
  p->nb_of_scales = dt_bauhaus_slider_get(w);
  // let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void edge_threshold_callback(GtkWidget *w, dt_iop_module_t *self)
{
  // this is important to avoid cycles!
  if(darktable.gui->reset) return;
  dt_iop_msca_params_t *p = (dt_iop_msca_params_t *)self->params;
  p->edge_threshold = dt_bauhaus_slider_get(w);
  // let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void correction_margin_callback(GtkWidget *w, dt_iop_module_t *self)
{
  // this is important to avoid cycles!
  if(darktable.gui->reset) return;
  dt_iop_msca_params_t *p = (dt_iop_msca_params_t *)self->params;
  p->correction_margin = dt_bauhaus_slider_get(w);
  // let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_update(dt_iop_module_t *self)
{
  // let gui slider match current parameters:
  dt_iop_msca_gui_data_t *g = (dt_iop_msca_gui_data_t *)self->gui_data;
  dt_iop_msca_params_t *p = (dt_iop_msca_params_t *)self->params;
  dt_bauhaus_slider_set(g->nb_of_scales, p->nb_of_scales);
  dt_bauhaus_slider_set(g->edge_threshold, p->edge_threshold);
  dt_bauhaus_slider_set(g->correction_margin, p->correction_margin);
}

void gui_init(dt_iop_module_t *self)
{
  // init the slider (more sophisticated layouts are possible with gtk tables and boxes):
  self->gui_data = malloc(sizeof(dt_iop_msca_gui_data_t));
  GtkWidget* widget = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));
  dt_iop_msca_gui_data_t *g = (dt_iop_msca_gui_data_t *)self->gui_data;
  g->nb_of_scales = dt_bauhaus_slider_new_with_range(self, 1.0f, 8.0f, 1.f, 2.f, 0);
  dt_bauhaus_widget_set_label(g->nb_of_scales, NULL, _("number of scales"));
  gtk_widget_set_tooltip_text(g->nb_of_scales,  _("number of scale used to correct the chromatic aberation\n"
                                                  "increase if you have large chromatic aberations"));
  g_signal_connect(G_OBJECT(g->nb_of_scales), "value-changed", G_CALLBACK(nb_of_scales_callback), self);
  g->edge_threshold = dt_bauhaus_slider_new_with_range(self, 0.0f, 10.f, .1, 1.0f, 1);
  dt_bauhaus_widget_set_label(g->edge_threshold, NULL, _("edge detection threshold"));
  gtk_widget_set_tooltip_text(g->edge_threshold, _("threshold to detect edges\n"
                                                  "decrease if chromatic aberations are uncorrected"));
  g_signal_connect(G_OBJECT(g->edge_threshold), "value-changed", G_CALLBACK(edge_threshold_callback), self);
  g->correction_margin = dt_bauhaus_slider_new_with_range(self, 0.0f, 1.0f, .05, 0.f, 3);
  dt_bauhaus_widget_set_label(g->correction_margin, NULL, _("correction margin"));
  gtk_widget_set_tooltip_text(g->correction_margin,  _("amount of correction on detected edges"));
  g_signal_connect(G_OBJECT(g->correction_margin), "value-changed", G_CALLBACK(correction_margin_callback), self);
  gtk_box_pack_start(GTK_BOX(widget), GTK_WIDGET(g->nb_of_scales), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(widget), GTK_WIDGET(g->edge_threshold), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(widget), GTK_WIDGET(g->correction_margin), TRUE, TRUE, 0);
  gtk_widget_show_all(widget);
  self->widget = widget;
}

void gui_cleanup(dt_iop_module_t *self)
{
  // nothing else necessary, gtk will clean up the slider.
  free(self->gui_data);
  self->gui_data = NULL;
}
