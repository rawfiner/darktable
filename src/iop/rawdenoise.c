/*
    This file is part of darktable,
    copyright (c) 2011 bruce guenter
    copyright (c) 2012 henrik andersson


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
#include "common/noiseprofiles.h"
#include "control/control.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <stdlib.h>
#include <strings.h>

DT_MODULE_INTROSPECTION(2, dt_iop_rawdenoise_params_t)

typedef enum dt_iop_rawdenoise_profile_mode_t { UNPROFILED = 0, PROFILED = 1 } dt_iop_rawdenoise_profile_mode_t;

typedef struct dt_iop_rawdenoise_params_t
{
  float threshold;                               // threshold for wavelets
  dt_iop_rawdenoise_profile_mode_t profile_mode; // whether to use anscombe transform or not
  float strength;                                // strength for anscombe transform
  float a[3], b[3];                              // fit for poissonian-gaussian noise per color channel
} dt_iop_rawdenoise_params_t;

typedef struct dt_iop_rawdenoise_gui_data_t
{
  GtkWidget *stack;
  GtkWidget *box_raw;
  GtkWidget *profile_mode;
  GtkWidget *strength;
  GtkWidget *profile;
  dt_noiseprofile_t interpolated;
  GList *profiles;
  GtkWidget *threshold;
  GtkWidget *label_non_raw;
} dt_iop_rawdenoise_gui_data_t;

typedef struct dt_iop_rawdenoise_data_t
{
  float threshold;                               // threshold for wavelets
  dt_iop_rawdenoise_profile_mode_t profile_mode; // whether to use anscombe transform or not
  float strength;                                // strength for anscombe transform
  float a[3], b[3];                              // fit for poissonian-gaussian noise per color channel
} dt_iop_rawdenoise_data_t;

typedef struct dt_iop_rawdenoise_global_data_t
{
} dt_iop_rawdenoise_global_data_t;

const char *name()
{
  return _("raw denoise");
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING;
}

int groups()
{
  return IOP_GROUP_CORRECT;
}

void init_key_accels(dt_iop_module_so_t *self)
{
  dt_accel_register_slider_iop(self, FALSE, NC_("accel", "noise threshold"));
}

void connect_key_accels(dt_iop_module_t *self)
{
  dt_iop_rawdenoise_gui_data_t *g = (dt_iop_rawdenoise_gui_data_t *)self->gui_data;

  dt_accel_connect_slider_iop(self, "noise threshold", GTK_WIDGET(g->threshold));
}

int legacy_params(dt_iop_module_t *self, const void *const old_params, const int old_version, void *new_params,
                  const int new_version)
{
  dt_iop_rawdenoise_params_t *o = (dt_iop_rawdenoise_params_t *)old_params;
  dt_iop_rawdenoise_params_t *n = (dt_iop_rawdenoise_params_t *)new_params;
  if((old_version == 1) && (new_version == 2))
  {
    n->threshold = o->threshold;
    n->profile_mode = UNPROFILED;
    return 0;
  }
  return 1;
}

static inline void precondition(const float *const in, float *const buf, const int wd, const int ht,
                                const float a[3], const float b[3], const dt_iop_roi_t *const roi_in,
                                const uint32_t filters, const uint8_t (*const xtrans)[6])
{
  const float sigma2[3]
      = { (b[0] / a[0]) * (b[0] / a[0]), (b[1] / a[1]) * (b[1] / a[1]), (b[2] / a[2]) * (b[2] / a[2]) };

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(a)
#endif
  for(int j = 0; j < ht; j++)
  {
    float *buf2 = buf + (size_t)j * wd;
    const float *in2 = in + (size_t)j * wd;
    for(int i = 0; i < wd; i++)
    {
      int c;
      if(filters != 9u)
        c = FC(j, i, filters);
      else
        c = FCxtrans(j, i, roi_in, xtrans);
      *buf2 = *in2 / a[c];
      const float d = fmaxf(0.0f, *buf2 + 3. / 8. + sigma2[c]);
      *buf2 = 2.0f * sqrtf(d);
      buf2++;
      in2++;
    }
  }
}

static inline void backtransform(float *const buf, const int wd, const int ht, const float a[3], const float b[3],
                                 const dt_iop_roi_t *const roi_in, const uint32_t filters,
                                 const uint8_t (*const xtrans)[6])
{
  const float sigma2[3]
      = { (b[0] / a[0]) * (b[0] / a[0]), (b[1] / a[1]) * (b[1] / a[1]), (b[2] / a[2]) * (b[2] / a[2]) };

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(a)
#endif
  for(int j = 0; j < ht; j++)
  {
    float *buf2 = buf + (size_t)j * wd;
    for(int i = 0; i < wd; i++)
    {
      int c;
      if(filters != 9u)
        c = FC(j, i, filters);
      else
        c = FCxtrans(j, i, roi_in, xtrans);
      const float x = *buf2;
      // closed form approximation to unbiased inverse (input range was 0..200 for fit, not 0..1)
      if(x < .5f)
        *buf2 = 0.0f;
      else
        *buf2 = 1. / 4. * x * x + 1. / 4. * sqrtf(3. / 2.) / x - 11. / 8. * 1.0 / (x * x)
                + 5. / 8. * sqrtf(3. / 2.) * 1.0 / (x * x * x) - 1. / 8. - sigma2[c];
      // asymptotic form:
      // buf2[c] = fmaxf(0.0f, 1./4.*x*x - 1./8. - sigma2[c]);
      *buf2 *= a[c];
      buf2++;
    }
  }
}

// transposes image, it is faster to read columns than to write them.
static void hat_transform(float *temp, const float *const base, int stride, int size, int scale)
{
  int i;
  const float *basep0;
  const float *basep1;
  const float *basep2;
  const size_t stxsc = (size_t)stride * scale;

  basep0 = base;
  basep1 = base + stxsc;
  basep2 = base + stxsc;

  for(i = 0; i < scale; i++, basep0 += stride, basep1 -= stride, basep2 += stride)
    temp[i] = (*basep0 + *basep0 + *basep1 + *basep2) * 0.25f;

  for(; i < size - scale; i++, basep0 += stride)
    temp[i] = ((*basep0) * 2 + *(basep0 - stxsc) + *(basep0 + stxsc)) * 0.25f;

  basep1 = basep0 - stxsc;
  basep2 = base + stride * (size - 2);

  for(; i < size; i++, basep0 += stride, basep1 += stride, basep2 -= stride)
    temp[i] = (*basep0 + *basep0 + *basep1 + *basep2) * 0.25f;
}

#define BIT16 65536.0

static void wavelet_denoise(const float *const in, float *const out, const dt_iop_roi_t *const roi,
                            float threshold, uint32_t filters)
{
  int lev;
  // static float noise[] = { 1.0, 0.2735, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0 };
  static float noise_ref[] = { 0.8002, 0.2735, 0.1202, 0.0585, 0.0291, 0.0152, 0.0080, 0.0044 };

  const size_t size = (size_t)(roi->width / 2 + 1) * (roi->height / 2 + 1);
#if 0
  float maximum = 1.0;		/* FIXME */
  float black = 0.0;		/* FIXME */
  maximum *= BIT16;
  black *= BIT16;
  for (c=0; c<4; c++)
    cblack[c] *= BIT16;
#endif
  float *const fimg = calloc(size * 4, sizeof *fimg);


  const int nc = 4;
  for(int c = 0; c < nc; c++) /* denoise R,G1,B,G3 individually */
  {
    static float noise[8];
    if(c != 0 && c != 3)
    {
      // green pixels
      for(int i = 0; i < 3; i++) noise[i] = noise_ref[i] / 2.0;
      for(int i = 3; i < 8; i++) noise[i] = noise_ref[i] / 3.0;
    }
    else
    {
      for(int i = 0; i < 8; i++) noise[i] = noise_ref[i];
    }
    // zero lowest quarter part
    memset(fimg, 0, size * sizeof(float));

    // adjust for odd width and height
    const int halfwidth = roi->width / 2 + (roi->width & (~(c >> 1)) & 1);
    const int halfheight = roi->height / 2 + (roi->height & (~c) & 1);

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(c, filters) schedule(static)
#endif
    for(int row = c & 1; row < roi->height; row += 2)
    {
      float *fimgp = fimg + size + (size_t)row / 2 * halfwidth;
      int col = (c & 2) >> 1;
      const float *inp = in + (size_t)row * roi->width + col;
      for(; col < roi->width; col += 2, fimgp++, inp += 2) *fimgp = sqrt(MAX(0, *inp));
    }

    int lastpass;

    for(lev = 0; lev < 5; lev++)
    {
      const size_t pass1 = size * ((lev & 1) * 2 + 1);
      const size_t pass2 = 2 * size;
      const size_t pass3 = 4 * size - pass1;

// filter horizontally and transpose
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(lev) schedule(static)
#endif
      for(int col = 0; col < halfwidth; col++)
      {
        hat_transform(fimg + pass2 + (size_t)col * halfheight, fimg + pass1 + col, halfwidth, halfheight,
                      1 << lev);
      }
// filter vertically and transpose back
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(lev) schedule(static)
#endif
      for(int row = 0; row < halfheight; row++)
      {
        hat_transform(fimg + pass3 + (size_t)row * halfwidth, fimg + pass2 + row, halfheight, halfwidth,
                      1 << lev);
      }

      const float thold = threshold * noise[lev];
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(lev)
#endif
      for(size_t i = 0; i < (size_t)halfwidth * halfheight; i++)
      {
        float *fimgp = fimg + i;
        const float diff = fimgp[pass1] - fimgp[pass3];
        fimgp[0] += copysignf(fmaxf(fabsf(diff) - thold, 0), diff);
      }

      lastpass = pass3;
    }
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(c, lastpass) schedule(static)
#endif
    for(int row = c & 1; row < roi->height; row += 2)
    {
      const float *fimgp = fimg + (size_t)row / 2 * halfwidth;
      int col = (c & 2) >> 1;
      float *outp = out + (size_t)row * roi->width + col;
      for(; col < roi->width; col += 2, fimgp++, outp += 2)
      {
        float d = fimgp[0] + fimgp[lastpass];
        *outp = d * d;
      }
    }
  }
#if 0
  /* FIXME: Haven't ported this part yet */
  if (filters && colors == 3)	/* pull G1 and G3 closer together */
  {
    float *window[4];
    int wlast, blk[2];
    float mul[2];
    float thold = threshold/512;
    for (row=0; row < 2; row++)
    {
      mul[row] = 0.125 * pre_mul[FC(row+1,0) | 1] / pre_mul[FC(row,0) | 1];
      blk[row] = cblack[FC(row,0) | 1];
    }
    for (i=0; i < 4; i++)
      window[i] = fimg + width*i;
    for (wlast=-1, row=1; row < height-1; row++)
    {
      while (wlast < row+1)
      {
        for (wlast++, i=0; i < 4; i++)
          window[(i+3) & 3] = window[i];
        for (col = FC(wlast,1) & 1; col < width; col+=2)
          window[2][col] = BAYER(wlast,col);
      }
      for (col = (FC(row,0) & 1)+1; col < width-1; col+=2)
      {
        float avg = ( window[0][col-1] + window[0][col+1] +
                      window[2][col-1] + window[2][col+1] - blk[~row & 1]*4 )
                    * mul[row & 1] + (window[1][col] + blk[row & 1]) * 0.5;
        avg = avg < 0 ? 0 : sqrt(avg);
        float diff = sqrt(BAYER(row,col)) - avg;
        if      (diff < -thold) diff += thold;
        else if (diff >  thold) diff -= thold;
        else diff = 0;
        BAYER(row,col) = SQR(avg+diff);
      }
    }
  }
#endif
  free(fimg);
}

static void wavelet_denoise_xtrans(const float *const in, float *out, const dt_iop_roi_t *const roi,
                                   float threshold, const uint8_t (*const xtrans)[6])
{
  // note that these constants are the same for X-Trans and Bayer, as
  // they are proportional to image detail on each channel, not the
  // sensor pattern
  static float noise_ref[] = { 0.8002, 0.2735, 0.1202, 0.0585, 0.0291, 0.0152, 0.0080, 0.0044 };

  const int width = roi->width;
  const int height = roi->height;
  const size_t size = (size_t)width * height;
  float *const fimg = malloc((size_t)size * 4 * sizeof(float));

  for(int c = 0; c < 3; c++)
  {
    memset(fimg, 0, size * sizeof(float));
    static float noise[8];
    if(c != 0 && c != 2)
    {
      // green pixels
      for(int i = 0; i < 3; i++) noise[i] = noise_ref[i] / 2.0;
      for(int i = 3; i < 8; i++) noise[i] = noise_ref[i] / 3.0;
    }
    else
    {
      for(int i = 0; i < 8; i++) noise[i] = noise_ref[i];
    }

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(c) schedule(static) firstprivate(noise)
#endif
    for(int row = (c != 1); row < height - 1; row++)
    {
      int col = (c != 1);
      const float *inp = in + (size_t)row * width + col;
      float *fimgp = fimg + size + (size_t)row * width + col;
      for(; col < width - 1; col++, inp++, fimgp++)
        if(FCxtrans(row, col, roi, xtrans) == c)
        {
          float d = sqrt(MAX(0, *inp));
          *fimgp = d;
          // cheap nearest-neighbor interpolate
          if(c == 1)
            fimgp[1] = fimgp[width] = d;
          else
          {
            fimgp[-width - 1] = fimgp[-width] = fimgp[-width + 1] = fimgp[-1] = fimgp[1] = fimgp[width - 1]
                = fimgp[width] = fimgp[width + 1] = d;
          }
        }
    }

    int lastpass;

    for(int lev = 0; lev < 5; lev++)
    {
      const size_t pass1 = size * ((lev & 1) * 2 + 1);
      const size_t pass2 = 2 * size;
      const size_t pass3 = 4 * size - pass1;

// filter horizontally and transpose
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(lev) schedule(static)
#endif
      for(int col = 0; col < width; col++)
        hat_transform(fimg + pass2 + (size_t)col * height, fimg + pass1 + col, width, height, 1 << lev);
// filter vertically and transpose back
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(lev) schedule(static)
#endif
      for(int row = 0; row < height; row++)
        hat_transform(fimg + pass3 + (size_t)row * width, fimg + pass2 + row, height, width, 1 << lev);

      const float thold = threshold * noise[lev];
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(lev)
#endif
      for(size_t i = 0; i < size; i++)
      {
        float *fimgp = fimg + i;
        const float diff = fimgp[pass1] - fimgp[pass3];
        fimgp[0] += /*0.4 **/ copysignf(fmaxf(fabsf(diff) - thold * 10.0, 0), diff);
        // + 0.3 * copysignf(fmaxf(fabsf(diff) - thold * 2.0, 0), diff)
        // + 0.2 * copysignf(fmaxf(fabsf(diff) - thold * 4.0, 0), diff)
        // + 0.1 * copysignf(fmaxf(fabsf(diff) - thold * 8.0, 0), diff);
      }

      lastpass = pass3;
    }

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(c, lastpass, out) schedule(static)
#endif
    for(int row = 0; row < height; row++)
    {
      const float *fimgp = fimg + (size_t)row * width;
      float *outp = out + (size_t)row * width;
      for(int col = 0; col < width; col++, outp++, fimgp++)
        if(FCxtrans(row, col, roi, xtrans) == c)
        {
          float d = fimgp[0] + fimgp[lastpass];
          *outp = d * d;
        }
    }
  }

  free(fimg);
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_rawdenoise_data_t *d = (dt_iop_rawdenoise_data_t *)piece->data;

  const int width = roi_in->width;
  const int height = roi_in->height;

  if(!(d->threshold > 0.0f))
  {
    memcpy(ovoid, ivoid, (size_t)sizeof(float)*width*height);
  }
  else
  {
    const uint32_t filters = piece->pipe->dsc.filters;
    const uint8_t(*const xtrans)[6] = (const uint8_t(*const)[6])piece->pipe->dsc.xtrans;

    if(d->profile_mode == UNPROFILED)
    {
      if(filters != 9u)
        wavelet_denoise(ivoid, ovoid, roi_in, d->threshold, filters);
      else
        wavelet_denoise_xtrans(ivoid, ovoid, roi_in, d->threshold, xtrans);
    }
    else
    {
      const float wb[3] = { d->strength, d->strength, d->strength };
      const float aa[3] = { d->a[1] * wb[0], d->a[1] * wb[1], d->a[1] * wb[2] };
      const float bb[3] = { d->b[1] * wb[0], d->b[1] * wb[1], d->b[1] * wb[2] };
      printf("%f, %f\n\n", d->a[1], d->b[1]);
      float *in = malloc(sizeof(float) * roi_in->width * roi_in->height);
      precondition((float *)ivoid, in, roi_in->width, roi_in->height, aa, bb, roi_in, filters, xtrans);
      if(filters != 9u)
        wavelet_denoise(in, ovoid, roi_in, d->threshold * 10.0, filters);
      else
        wavelet_denoise_xtrans(in, ovoid, roi_in, d->threshold * 10.0, xtrans);
      free(in);
      backtransform((float *)ovoid, roi_in->width, roi_in->height, aa, bb, roi_in, filters, xtrans);
    }
  }
}

// FIXME code shared with denoiseprofile.c -> refactor at a common place
static dt_noiseprofile_t dt_iop_denoiseprofile_get_auto_profile(dt_iop_module_t *self)
{
  GList *profiles = dt_noiseprofile_get_matching(&self->dev->image_storage);
  dt_noiseprofile_t interpolated = dt_noiseprofile_generic; // default to generic poissonian

  const int iso = self->dev->image_storage.exif_iso;
  dt_noiseprofile_t *last = NULL;
  for(GList *iter = profiles; iter; iter = g_list_next(iter))
  {
    dt_noiseprofile_t *current = (dt_noiseprofile_t *)iter->data;
    if(current->iso == iso)
    {
      interpolated = *current;
      break;
    }
    if(last && last->iso < iso && current->iso > iso)
    {
      dt_noiseprofile_interpolate(last, current, &interpolated);
      break;
    }
    last = current;
  }
  g_list_free_full(profiles, dt_noiseprofile_free);
  return interpolated;
}

void reload_defaults(dt_iop_module_t *module)
{
  // init defaults:
  ((dt_iop_rawdenoise_params_t *)module->default_params)->threshold = 0.01;
  ((dt_iop_rawdenoise_params_t *)module->default_params)->strength = 1.0f;
  ((dt_iop_rawdenoise_params_t *)module->default_params)->profile_mode = PROFILED;

  // we might be called from presets update infrastructure => there is no image
  if(!module->dev)
  {
    memcpy(module->params, module->default_params, sizeof(dt_iop_rawdenoise_params_t));
    return;
  }

  // can't be switched on for non-raw images:
  if(dt_image_is_raw(&module->dev->image_storage))
    module->hide_enable_button = 0;
  else
    module->hide_enable_button = 1;

  module->default_enabled = 0;
  dt_iop_rawdenoise_gui_data_t *g = (dt_iop_rawdenoise_gui_data_t *)module->gui_data;
  if(g)
  {
    dt_bauhaus_combobox_clear(g->profile);

    // get matching profiles:
    char name[512];
    // FIXME if(g->profiles) g_list_free_full(g->profiles, dt_noiseprofile_free);
    g->profiles = dt_noiseprofile_get_matching(&module->dev->image_storage);
    g->interpolated = dt_noiseprofile_generic; // default to generic poissonian
    g_strlcpy(name, _(g->interpolated.name), sizeof(name));

    const int iso = module->dev->image_storage.exif_iso;
    dt_noiseprofile_t *last = NULL;
    for(GList *iter = g->profiles; iter; iter = g_list_next(iter))
    {
      dt_noiseprofile_t *current = (dt_noiseprofile_t *)iter->data;

      if(current->iso == iso)
      {
        g->interpolated = *current;
        // signal later autodetection in commit_params:
        g->interpolated.a[0] = -1.0;
        snprintf(name, sizeof(name), _("found match for ISO %d"), iso);
        break;
      }
      if(last && last->iso < iso && current->iso > iso)
      {
        dt_noiseprofile_interpolate(last, current, &g->interpolated);
        // signal later autodetection in commit_params:
        g->interpolated.a[0] = -1.0;
        snprintf(name, sizeof(name), _("interpolated from ISO %d and %d"), last->iso, current->iso);
        break;
      }
      last = current;
    }

    dt_bauhaus_combobox_add(g->profile, name);
    for(GList *iter = g->profiles; iter; iter = g_list_next(iter))
    {
      dt_noiseprofile_t *profile = (dt_noiseprofile_t *)iter->data;
      dt_bauhaus_combobox_add(g->profile, profile->name);
    }

    ((dt_iop_rawdenoise_params_t *)module->default_params)->strength = 1.0f;
    ((dt_iop_rawdenoise_params_t *)module->default_params)->profile_mode = PROFILED;
    for(int k = 0; k < 3; k++)
    {
      ((dt_iop_rawdenoise_params_t *)module->default_params)->a[k] = g->interpolated.a[k];
      ((dt_iop_rawdenoise_params_t *)module->default_params)->b[k] = g->interpolated.b[k];
    }
    memcpy(module->params, module->default_params, sizeof(dt_iop_rawdenoise_params_t));
  }
}

void init(dt_iop_module_t *module)
{
  module->data = NULL;
  module->params = calloc(1, sizeof(dt_iop_rawdenoise_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_rawdenoise_params_t));
  module->default_enabled = 0;

  // raw denoise must come just before demosaicing.
  module->priority = 102; // module order created by iop_dependencies.py, do not edit!
  module->params_size = sizeof(dt_iop_rawdenoise_params_t);
  module->gui_data = NULL;
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
  free(module->data);
  module->data = NULL;
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *params, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_rawdenoise_params_t *p = (dt_iop_rawdenoise_params_t *)params;
  dt_iop_rawdenoise_data_t *d = (dt_iop_rawdenoise_data_t *)piece->data;

  d->threshold = p->threshold;
  d->profile_mode = p->profile_mode;
  d->strength = p->strength;

  // compare if a[0] in params is set to "magic value" -1.0 for autodetection
  if(p->a[0] == -1.0)
  {
    // autodetect matching profile again, the same way as detecting their names,
    // this is partially duplicated code and data because we are not allowed to access
    // gui_data here ..
    dt_noiseprofile_t interpolated = dt_iop_denoiseprofile_get_auto_profile(self);
    for(int k = 0; k < 3; k++)
    {
      d->a[k] = interpolated.a[k];
      d->b[k] = interpolated.b[k];
    }
  }

  if (!(pipe->image.flags & DT_IMAGE_RAW))
    piece->enabled = 0;
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_rawdenoise_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_rawdenoise_gui_data_t *g = (dt_iop_rawdenoise_gui_data_t *)self->gui_data;
  dt_iop_rawdenoise_params_t *p = (dt_iop_rawdenoise_params_t *)self->params;

  dt_bauhaus_slider_set(g->threshold, p->threshold);
  dt_bauhaus_slider_set(g->strength, p->strength);
  dt_bauhaus_combobox_set(g->profile_mode, p->profile_mode);
  dt_bauhaus_combobox_set(g->profile, -1);
  if(p->profile_mode == UNPROFILED)
  {
    gtk_widget_set_visible(g->profile, FALSE);
    gtk_widget_set_visible(g->strength, FALSE);
  }
  else
  {
    gtk_widget_set_visible(g->profile, TRUE);
    gtk_widget_set_visible(g->strength, TRUE);
  }
  if(p->a[0] == -1.0)
  {
    dt_bauhaus_combobox_set(g->profile, 0);
  }
  else
  {
    int i = 1;
    for(GList *iter = g->profiles; iter; iter = g_list_next(iter), i++)
    {
      dt_noiseprofile_t *profile = (dt_noiseprofile_t *)iter->data;
      if(!memcmp(profile->a, p->a, sizeof(float) * 3) && !memcmp(profile->b, p->b, sizeof(float) * 3))
      {
        dt_bauhaus_combobox_set(g->profile, i);
        break;
      }
    }
  }
  gtk_stack_set_visible_child_name(GTK_STACK(g->stack), self->hide_enable_button ? "non_raw" : "raw");
}

static void threshold_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_rawdenoise_params_t *p = (dt_iop_rawdenoise_params_t *)self->params;
  p->threshold = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void profile_callback(GtkWidget *w, dt_iop_module_t *self)
{
  int i = dt_bauhaus_combobox_get(w);
  dt_iop_rawdenoise_params_t *p = (dt_iop_rawdenoise_params_t *)self->params;
  dt_iop_rawdenoise_gui_data_t *g = (dt_iop_rawdenoise_gui_data_t *)self->gui_data;
  const dt_noiseprofile_t *profile = &(g->interpolated);
  if(i > 0) profile = (dt_noiseprofile_t *)g_list_nth_data(g->profiles, i - 1);
  for(int k = 0; k < 3; k++)
  {
    p->a[k] = profile->a[k];
    p->b[k] = profile->b[k];
  }
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void profile_mode_callback(GtkWidget *w, dt_iop_module_t *self)
{
  dt_iop_rawdenoise_params_t *p = (dt_iop_rawdenoise_params_t *)self->params;
  dt_iop_rawdenoise_gui_data_t *g = (dt_iop_rawdenoise_gui_data_t *)self->gui_data;
  p->profile_mode = dt_bauhaus_combobox_get(w);
  if(p->profile_mode == UNPROFILED)
  {
    gtk_widget_set_visible(g->profile, FALSE);
    gtk_widget_set_visible(g->strength, FALSE);
  }
  else
  {
    gtk_widget_set_visible(g->profile, TRUE);
    gtk_widget_set_visible(g->strength, TRUE);
  }
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void strength_callback(GtkWidget *w, dt_iop_module_t *self)
{
  dt_iop_rawdenoise_params_t *p = (dt_iop_rawdenoise_params_t *)self->params;
  p->strength = dt_bauhaus_slider_get(w);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}


void gui_init(dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_rawdenoise_gui_data_t));
  dt_iop_rawdenoise_gui_data_t *g = (dt_iop_rawdenoise_gui_data_t *)self->gui_data;
  dt_iop_rawdenoise_params_t *p = (dt_iop_rawdenoise_params_t *)self->params;

  self->widget = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));

  g->stack = gtk_stack_new();
  gtk_stack_set_homogeneous(GTK_STACK(g->stack), FALSE);
  gtk_box_pack_start(GTK_BOX(self->widget), g->stack, TRUE, TRUE, 0);

  g->box_raw = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  g->profile_mode = dt_bauhaus_combobox_new(self);
  g->profile = dt_bauhaus_combobox_new(self);
  g->strength = dt_bauhaus_slider_new_with_range(self, 0.001f, 4.0f, .05, 1.f, 3);
  gtk_box_pack_start(GTK_BOX(g->box_raw), g->profile_mode, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->box_raw), g->profile, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->box_raw), g->strength, TRUE, TRUE, 0);
  dt_bauhaus_widget_set_label(g->profile, NULL, _("profile"));
  dt_bauhaus_widget_set_label(g->profile_mode, NULL, _("denoising mode"));
  dt_bauhaus_widget_set_label(g->strength, NULL, _("strength"));
  dt_bauhaus_combobox_add(g->profile_mode, _("unprofiled"));
  dt_bauhaus_combobox_add(g->profile_mode, _("profiled"));
  gtk_widget_set_tooltip_text(g->profile, _("profile used for variance stabilization"));
  gtk_widget_set_tooltip_text(g->profile_mode,
                              _("whether to use a camera profile for variance stabilization, or not."));
  gtk_widget_set_tooltip_text(g->strength, _("finetune denoising strength"));
  g_signal_connect(G_OBJECT(g->profile), "value-changed", G_CALLBACK(profile_callback), self);
  g_signal_connect(G_OBJECT(g->profile_mode), "value-changed", G_CALLBACK(profile_mode_callback), self);
  g_signal_connect(G_OBJECT(g->strength), "value-changed", G_CALLBACK(strength_callback), self);

  /* threshold */
  g->threshold = dt_bauhaus_slider_new_with_range(self, 0.0, 0.1, 0.001, p->threshold, 3);
  gtk_box_pack_start(GTK_BOX(g->box_raw), GTK_WIDGET(g->threshold), TRUE, TRUE, 0);
  dt_bauhaus_widget_set_label(g->threshold, NULL, _("noise threshold"));
  g_signal_connect(G_OBJECT(g->threshold), "value-changed", G_CALLBACK(threshold_callback), self);



  gtk_widget_show_all(g->box_raw);
  gtk_stack_add_named(GTK_STACK(g->stack), g->box_raw, "raw");

  g->label_non_raw = gtk_label_new(_("raw denoising\nonly works for raw images."));
  gtk_widget_set_halign(g->label_non_raw, GTK_ALIGN_START);

  gtk_widget_show_all(g->label_non_raw);
  gtk_stack_add_named(GTK_STACK(g->stack), g->label_non_raw, "non_raw");

  gtk_stack_set_visible_child_name(GTK_STACK(g->stack), self->hide_enable_button ? "non_raw" : "raw");
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_rawdenoise_gui_data_t *g = (dt_iop_rawdenoise_gui_data_t *)self->gui_data;
  g_list_free_full(g->profiles, dt_noiseprofile_free);
  free(self->gui_data);
  self->gui_data = NULL;
}
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
