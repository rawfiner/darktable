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

static void var(float* details, float* variance, unsigned width, unsigned height, unsigned radius, float* wb, unsigned max_var)
{
  float* tmp = calloc(sizeof(float), width * height * 4);
  //FIXME number of elements is not the same for all pixels
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(details, variance, width, height, radius, tmp, wb) \
    schedule(static)
#endif
  for(unsigned i = 0; i < height; i++)
  {
    for(unsigned j = 0; j < width; j++)
    {
      const size_t begin_convol = (i < radius) ? 0 : i - radius;
      size_t end_convol = i + radius;
      end_convol = (end_convol < height) ? end_convol : height - 1;
      // const float num_elem = 1.0f / ((float)end_convol - (float)begin_convol + 1.0f);
      for(size_t s = begin_convol; s <= end_convol; s++)
      {
        float sum = 0.0f;
        for(unsigned c = 0; c < 3; c++)
        {
          float value = details[(s * width + j) * 4 + c];// / wb[c];
          sum += value;
          tmp[(s * width + j) * 4 + c] += value * value;
        }
        sum /= 3.0f;
        tmp[(s * width + j) * 4 + 3] += sum * sum;
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(details, variance, width, height, radius, tmp, wb, max_var) \
    schedule(static)
#endif
  for(unsigned i = 0; i < height; i++)
  {
    for(unsigned j = 0; j < width; j++)
    {
      const size_t begin_convol = (j < radius) ? 0 : j - radius;
      size_t end_convol = j + radius;
      end_convol = (end_convol < width) ? end_convol : width - 1;
      for(size_t s = begin_convol; s <= end_convol; s+=1)
      {
        for(unsigned c = 0; c < 4; c++)
        {
          float value = tmp[(i * width + s) * 4 + c];
          variance[(i * width + j) * 4 + c] += value / (float)((2 * radius + 1) * (2 * radius + 1) - 1);
        }
      }
      float alpha = variance[(i * width + j) * 4] + variance[(i * width + j) * 4 + 1] + variance[(i * width + j) * 4 + 2];
      alpha /= 3.0f;
      float beta = variance[(i * width + j) * 4 + 3];
      for(unsigned c = 0; c < 4; c++)
      {
        variance[(i * width + j) * 4 + c] = 1.5f * (alpha - beta);// * wb[c] * wb[c];
      }
    }
  }
  free(tmp);
}

#define SWAP(x,y) if (diff[y] < diff[x]) { float tmp = diff[x]; diff[x] = diff[y]; diff[y] = tmp; float* tmpdir = dir[x]; dir[x] = dir[y]; dir[y] = tmpdir; }

// for each pixel, direction[0] is the best direction, direction[1] the second best, etc
static void get_details_and_direction(const float* in, float* mean, float* details, unsigned width, unsigned height, float** direction, float* wb)
{
  const unsigned widthmean = (width + 1) / 2;
  const unsigned heightmean = (height + 1) / 2;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(in, mean, details, direction, width, height, widthmean, heightmean, wb) \
    schedule(static)
#endif
  for(unsigned j = 0; j < height; j++)
  {
    unsigned j0 = MAX(MIN(j / 2 + (j & 1) - 1, heightmean - 1), 0);
    unsigned j1 = MIN(j / 2 + (j & 1), heightmean - 1);
    for(unsigned i = 0; i < width; i++)
    {
      unsigned i0 = MAX(MIN(i / 2 + (i & 1) - 1, widthmean - 1), 0);
      unsigned i1 = MIN(i / 2 + (i & 1), widthmean - 1);
      float diff[4] = {0.0f};
      float** dir = &(direction[(j * width + i) * 4]);
      for(unsigned c = 0; c < 3; c++)
      {
        diff[0] += fabs(mean[(j0 * widthmean + i0) * 4 + c] - in[(j * width + i) * 4 + c]) / wb[c];
        diff[1] += fabs(mean[(j1 * widthmean + i0) * 4 + c] - in[(j * width + i) * 4 + c]) / wb[c];
        diff[2] += fabs(mean[(j0 * widthmean + i1) * 4 + c] - in[(j * width + i) * 4 + c]) / wb[c];
        diff[3] += fabs(mean[(j1 * widthmean + i1) * 4 + c] - in[(j * width + i) * 4 + c]) / wb[c];
      }
      dir[0] = &(mean[(j0 * widthmean + i0) * 4]);
      dir[1] = &(mean[(j1 * widthmean + i0) * 4]);
      dir[2] = &(mean[(j0 * widthmean + i1) * 4]);
      dir[3] = &(mean[(j1 * widthmean + i1) * 4]);

      // sort diff and dir jointly
      SWAP(0, 1);
      SWAP(2, 3);
      SWAP(0, 2);
      SWAP(1, 3);
      SWAP(1, 2);

      for(unsigned c = 0; c < 3; c++)
      {
        details[(j * width + i) * 4 + c] = in[(j * width + i) * 4 + c] - 0.75f * dir[0][c] - 0.25f * dir[1][c];
      }
    }
  }
}

// decompose image in 2 layers: each pixel of out is a 4 pixels mean, and scaling up 2x out and adding details gives back in
// out width is (width+1)/2
// out height is (height+1)/2
static void decompose(const float* in, float* out, unsigned width, unsigned height)
{
  const unsigned widthout = (width + 1) / 2;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(in, out, width, widthout, height) \
    schedule(static)
#endif
  for(unsigned j = 0; j < height; j+=2)
  {
    unsigned jout = j / 2;
    for(unsigned i = 0; i < width; i+=2)
    {
      unsigned iout = i / 2;
      for(unsigned c = 0; c < 3; c++)
      {
        float tmp00 = in[(j*width+i)*4+c];
        float tmp01 = in[(j*width+MIN(i+1,width-1))*4+c];
        float tmp10 = in[(MIN(j+1,height-1)*width+i)*4+c];
        float tmp11 = in[(MIN(j+1,height-1)*width+MIN(i+1,width-1))*4+c];
        float mean = (tmp00 + tmp01 + tmp10 + tmp11) / 4.0f;
        out[(jout * widthout + iout) * 4 + c] = mean;
      }
    }
  }
}

static int sign(float a)
{
  return (a >= 0.0f) - (a < 0.0f);
}

static void thresholding(float* details, unsigned width, unsigned height, float** direction, float threshold, float* wb, float* var)
{
#if 0
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(details, direction, width, height, threshold, wb, var) \
    schedule(static)
#endif
#endif
  for(unsigned j = 0; j < height; j++)
  {
    for(unsigned i = 0; i < width; i++)
    {
      for(unsigned c = 0; c < 3; c++)
      {
        float det = details[(j * width + i) * 4 + c];
        // float thrs = threshold * sqrt((0.75f * direction[(j * width + i) * 4][c]
        //                           + 0.25f * direction[(j * width + i) * 4 + 1][c]) + 0.05f) * wb[c];
        float thrs = threshold * sqrt(var[(j * width + i) * 4 + c]);
        details[(j * width + i) * 4 + c] = sign(det) * MAX(fabs(det) - thrs, 0.0f);
      }
    }
  }
}

// recompose image from 2 layers
// width and height are the dimensions of out
static void recompose(float* in, float* out, float* details, unsigned width, unsigned height, float** direction)
{
  // const unsigned widthin = (width + 1) / 2;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(in, out, details, direction, width, height) \
    schedule(static)
#endif
  for(unsigned j = 0; j < height; j++)
  {
    for(unsigned i = 0; i < width; i++)
    {
      for(unsigned c = 0; c < 3; c++)
      {
        out[(j * width + i) * 4 + c] = 0.75f * direction[(j * width + i) * 4][c];
        out[(j * width + i) * 4 + c] += 0.25f * direction[(j * width + i) * 4 + 1][c];
        out[(j * width + i) * 4 + c] += details[(j * width + i) * 4 + c];
      }
    }
  }
}

#define NB_SCALES 8
#define RADIUS 5

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_blind_denoise_params_t *d = (dt_iop_blind_denoise_params_t *)piece->data;
  float* out = (float*)ovoid;
  float* in = (float*)ivoid;
  float* means[NB_SCALES];
  float* vars[NB_SCALES];
  float* details[NB_SCALES];
  float** direction[NB_SCALES];
  float threshold[NB_SCALES] = {0.7, 1.0, 0.9, 0.6, 0.3, 0.1, 0.05, 0.01};
//  float threshold[NB_SCALES] = {0.20, 0.20, 0.20, 0.15, 0.10, 0.05, 0.01, 0.007};
  unsigned width[NB_SCALES];
  unsigned height[NB_SCALES];
  float wb[3];
  for(int i = 0; i < 3; i++) wb[i] = piece->pipe->dsc.temperature.coeffs[i];

  // init
  means[0] = in;
  width[0] = roi_out->width;
  height[0] = roi_out->height;
  details[0] = (float*)malloc(sizeof(float) * 4 * width[0] * height[0]);
  vars[0] = (float*)calloc(sizeof(float), 4 * width[0] * height[0]);
  direction[0] = (float**)malloc(sizeof(float*) * 4 * width[0] * height[0]);
  for(int i = 1; i < NB_SCALES; i++)
  {
    width[i] = (width[i-1] + 1) / 2;
    height[i] = (height[i-1] + 1) / 2;
    means[i] = (float*)malloc(sizeof(float) * 4 * width[i] * height[i]);
    vars[i] = (float*)calloc(sizeof(float), 4 * width[i] * height[i]);
    details[i] = (float*)malloc(sizeof(float) * 4 * width[i] * height[i]);
    direction[i] = (float**)malloc(sizeof(float*) * 4 * width[i] * height[i]);
  }

  // threshold[0] *= d->checker_scale;
  // threshold[1] *= d->checker_scale;
  // threshold[2] *= d->checker_scale;
  for(int k = 0; k < NB_SCALES; k++)
  {
    threshold[k] *= d->factor;
  }

  for(int i = 0; i < NB_SCALES-1; i++)
  {
    decompose(means[i], means[i+1], width[i], height[i]);
  }
  for(int i = NB_SCALES-1; i > 1; i--)
  {
    get_details_and_direction(means[i-1], means[i], details[i-1], width[i-1], height[i-1], direction[i-1], wb);
    var(details[i-1], vars[i-1], width[i-1], height[i-1], NB_SCALES - i, wb, d->checker_scale);
    thresholding(details[i-1], width[i-1], height[i-1], direction[i-1], threshold[i-1], wb, vars[i-1]);
    recompose(means[i], means[i-1], details[i-1], width[i-1], height[i-1], direction[i-1]);
  }
  get_details_and_direction(means[0], means[1], details[0], width[0], height[0], direction[0], wb);
  var(details[0], vars[0], width[0], height[0], NB_SCALES, wb, d->checker_scale);
  thresholding(details[0], width[0], height[0], direction[0], threshold[0], wb, vars[0]);
  recompose(means[1], out, details[0], width[0], height[0], direction[0]);

  // cleanup
  for(int i = 1; i < NB_SCALES; i++)
  {
    free(means[i]);
    free(vars[i]);
    free(details[i]);
    free(direction[i]);
  }
  free(vars[0]);
  free(details[0]);
  free(direction[0]);
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

  g->factor = dt_bauhaus_slider_new_with_range(self, 0.0, 100.0, 0.1, 0.5, 2);
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
