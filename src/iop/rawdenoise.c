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
#include "control/control.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include <gtk/gtk.h>
#include <stdlib.h>
#include <strings.h>
#include <stdbool.h>

DT_MODULE_INTROSPECTION(2, dt_iop_rawdenoise_params_t)

typedef enum dt_iop_rawdenoise_mode_t
{
  MODE_NLMEANS = 0,
  MODE_WAVELETS = 1
} dt_iop_rawdenoise_mode_t;

typedef struct dt_iop_rawdenoise_params_t
{
  float threshold;
  float details;
  dt_iop_rawdenoise_mode_t mode;
} dt_iop_rawdenoise_params_t;

typedef struct dt_iop_rawdenoise_gui_data_t
{
  GtkWidget *stack;
  GtkWidget *box_raw;
  GtkWidget *mode;
  GtkWidget *threshold;
  GtkWidget *details;
  GtkWidget *label_non_raw;
} dt_iop_rawdenoise_gui_data_t;

typedef struct dt_iop_rawdenoise_data_t
{
  float threshold;
  float details;
  dt_iop_rawdenoise_mode_t mode;
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

int legacy_params(dt_iop_module_t *self, const void *const old_params, const int old_version,
                  void *new_params, const int new_version)
{
  dt_iop_rawdenoise_params_t *o = (dt_iop_rawdenoise_params_t *)old_params;
  dt_iop_rawdenoise_params_t *n = (dt_iop_rawdenoise_params_t *)new_params;
  if((old_version == 1) && new_version == 2)
  {
    n->threshold = o->threshold;
    n->details = 0.0f;
    n->mode = MODE_WAVELETS;
    return 0;
  }
  return 1;
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
  static const float noise[] = { 0.8002, 0.2735, 0.1202, 0.0585, 0.0291, 0.0152, 0.0080, 0.0044 };

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
    // zero lowest quarter part
    memset(fimg, 0, size * sizeof(float));

    // adjust for odd width and height
    const int halfwidth = roi->width / 2 + (roi->width & (~(c >> 1)) & 1);
    const int halfheight = roi->height / 2 + (roi->height & (~c) & 1);

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(c) schedule(static)
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
        fimgp[0] += copysignf(fmaxf(fabsf(diff) - thold, 0.0f), diff);
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
  static const float noise[] = { 0.8002, 0.2735, 0.1202, 0.0585, 0.0291, 0.0152, 0.0080, 0.0044 };

  const int width = roi->width;
  const int height = roi->height;
  const size_t size = (size_t)width * height;
  float *const fimg = malloc((size_t)size * 4 * sizeof(float));

  for(int c = 0; c < 3; c++)
  {
    memset(fimg, 0, size * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(c) schedule(static)
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
        fimgp[0] += copysignf(fmaxf(fabsf(diff) - thold, 0.0f), diff);
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
      float *inp = (float*)in + (size_t)row * width;
      for(int col = 0; col < width; col++, outp++, fimgp++, inp++)
        if(FCxtrans(row, col, roi, xtrans) == c)
        {
          float d = fimgp[0] + fimgp[lastpass];
          *outp = d * d;
        }
    }
  }

  free(fimg);
}

void *const downscale_bilinear_bayer_cfa(const void *const ivoid, int in_width, int in_height, int out_width,
                                         int out_height, const uint32_t filters)
{
  float *scaled_ivoid = (float *)malloc(sizeof(float) * out_width * out_height);
  float scale_factor = (float)in_width / (float)out_width;
  float *in = (float *)ivoid;
#pragma omp parallel for schedule(static) firstprivate(scale_factor, in) shared(scaled_ivoid)
  for(int j = 0; j < out_height; j++)
  {
    for(int i = 0; i < out_width; i++)
    {
      int color = FC(j, i, filters);
      // compute point coordinates in the original scale_factor
      // scale_factor /2 is here to give the coordinate of the middle of the pixel
      float oj = j * scale_factor + scale_factor / 2;
      float oi = i * scale_factor + scale_factor / 2;
      oi -= 0.5;
      oj -= 0.5;
      // find closest pixel that has the same color in the original grid
      int cj = MIN(MAX((int)(oj), 1), in_height - 2);
      int ci = MIN(MAX((int)(oi), 1), in_width - 2);
      int color_orig = FC(cj, ci, filters);

      if(color == 1)
      {
        if(color_orig != color)
        {
          // find nearest square
          float dtl = (oi - ci) * (oi - ci) + (oj - cj) * (oj - cj); // compare to top left
          float dbr
              = (oi - (ci + 1)) * (oi - (ci + 1)) + (oj - (cj + 1)) * (oj - (cj + 1)); // compare to bottom right
          if(dtl < dbr)
          {
            ci--;
            cj--;
          }
        }
        else
        {
          // find nearest square
          float db = (oi - ci) * (oi - ci) + (oj - (cj + 1)) * (oj - (cj + 1)); // compare to bottom
          float dr = (oi - (ci + 1)) * (oi - (ci + 1)) + (oj - cj) * (oj - cj); // compare to right
          if(db < dr)
            ci--;
          else
            cj--;
        }
        // A = (ci+1, cj)
        // B = (ci, cj+1)
        // C = (ci+2, cj+1)
        // D = (ci+1, cj+2)
        oi -= ci;
        oj -= cj;
        float fa = in[cj * in_width + ci + 1];
        float fb = in[(cj + 1) * in_width + ci];
        float fc = in[(cj + 1) * in_width + ci + 2];
        float fd = in[(cj + 2) * in_width + ci + 1];
        float a0 = 1.f / 4.f * (3.f * fa + 3.f * fb - fc - fd);
        float a1 = 1.f / 4.f * (fc + fd - fa - fb);
        float a2 = 1.f / 4.f * (3.f * fa - 3.f * fb - fc + fd);
        float a3 = 1.f / 4.f * (fb - fa + fc - fd);
        scaled_ivoid[j * out_width + i] = a0 + oi * (a1 + a2) + oj * (a1 - a2) + (oi * oi - oj * oj) * a3;
      }
      else
      {
        if(color_orig == 1)
        {
          int color_left = FC(cj, ci - 1, filters);
          if(color_left == color)
          {
            ci--;
          }
          else
          {
            cj--;
          }
        }
        else if(color_orig != color)
        {
          ci--;
          cj--;
        }
        // point is between (ci, cj), (ci+2,cj), (ci,cj+2), and (ci+2,cj+2)
        // interpolate horizontally
        float top = (ci + 2 - oi) * in[cj * in_width + ci] / 2 + (oi - ci) * in[cj * in_width + ci + 2] / 2;
        float bottom
            = (ci + 2 - oi) * in[(cj + 2) * in_width + ci] / 2 + (oi - ci) * in[(cj + 2) * in_width + ci + 2] / 2;
        // interpolate vertically
        scaled_ivoid[j * out_width + i] = (cj + 2 - oj) * top / 2 + (oj - cj) * bottom / 2;
      }
    }
  }
  return scaled_ivoid;
}


void *const halfscale_cfa(const void *const ivoid, dt_iop_roi_t *roi_in, dt_iop_roi_t *roi_out,
                          const uint32_t filters, const uint8_t (*const xtrans)[6], float scale_factor)
{
  int out_width = roi_in->width;
  int out_height = roi_in->height;
  if(scale_factor < 1.0) scale_factor = 1.0;
  float *half_ivoid = (float *)calloc(sizeof(float), out_width * out_height);
  out_width = (int)(out_width / scale_factor);
  out_height = (int)(out_height / scale_factor);
  float* in = (float*)ivoid;
#pragma omp parallel for schedule(static)
  for (int j = 0; j < out_height; j++)
  {
    for (int i = 0; i < out_width; i++)
    {
      int radius = 1;
      if(i < 4 || out_width - i < 4 || j < 4 || out_height - j < 4) // TODO 4 is probably not optimum
      {
        radius = 2;
      }
      int left = -MIN(i, radius);
      int right = MIN(out_width - i, radius);
      int up = -MIN(j, radius);
      int down = MIN(out_height - j, radius);
      double norm = 0;
      double value = 0;
      int color_big_pixel;
      if (filters == 9u)
        color_big_pixel = FCxtrans(j, i, roi_out, xtrans);
      else
        color_big_pixel = FC(j, i, filters);
      for(int jj = (int)((j + up) * scale_factor); jj <= (int)ceil((j + down) * scale_factor); jj++)
      {
        for(int ii = (int)((i + left) * scale_factor); ii <= (int)ceil((i + right) * scale_factor); ii++)
        {
          if(jj >= roi_in->height || ii >= roi_in->width) continue;
          int color;
          if (filters == 9u)
            color = FCxtrans(jj, ii, roi_in, xtrans);
          else
            color = FC(jj, ii, filters);

          if (color == color_big_pixel)
          {
            // add 0.5f to position to place the big pixel in the center of a
            // block of 4 pixels
            float j_pos_big_pixel = j * scale_factor + scale_factor / 2.0;
            float i_pos_big_pixel = i * scale_factor + scale_factor / 2.0;
            // compute distance between pixel and big_pixel
            // the "+0.5" are here to take the center of the small pixels
            float distance = sqrt((jj + 0.5 - j_pos_big_pixel) * (jj + 0.5 - j_pos_big_pixel)
                                  + (ii + 0.5 - i_pos_big_pixel) * (ii + 0.5 - i_pos_big_pixel));

            // add normalized value to big_pixel

            float distance_max = scale_factor;
            // handle cases where the distance_max can be too small to find any pixel
            if((scale_factor < 1.6) && (color != 1))
            {
              if(filters == 9u)
              {
                distance_max = 1.6;
              }
              else
              {
                if(distance_max < 1.42) distance_max = 1.42;
              }
              if(i < 2 || out_width - i < 2 || j < 2 || out_height - j < 2)
              {
                // close to the image ends, the minimal distance to closer pixel
                // is bigger (2.12 for xtrans, and 1.58 for bayer)
                distance_max = 2.13;
              }
            }
            if((distance_max < 1.12) && (filters == 9u))
            {
              // the point on the xtrans grid which is the farrest from the
              // green pixels is at a distance of sqrt(1^2+0.5^2) to the
              // nearest green pixel center.
              distance_max = 1.12;
            }
            if(distance < distance_max)
            {
              /* as we divide by the distance, we have to check if the distance is getting
               * very close to zero, and to prevent a divide by zero.
               * In addition, the min_dist parameter allow to have a better average when the
               * scale_factor is high, to prevent artefacts */
              const float min_dist = 0.05 * scale_factor * scale_factor * scale_factor * scale_factor;
              if(distance <= min_dist)
              {
                // big pixel is just over a small pixel
                // give a huge weight to this small pixel
                value += in[jj * roi_in->width + ii] / min_dist;
                norm += 1 / min_dist;
              }
              else
              {
                value += in[jj * roi_in->width + ii] / distance;
                norm += 1 / distance;
              }
            }
          }
        }
      }
      half_ivoid[j * roi_in->width + i] = value / norm;
    }
  }
  return (void *const)half_ivoid;
}

void modify_roi_in(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece, const dt_iop_roi_t *roi_out,
                   dt_iop_roi_t *roi_in)
{
  dt_iop_rawdenoise_data_t *d = (dt_iop_rawdenoise_data_t *)piece->data;
  float scale = 1.0 / d->threshold;
  roi_in->width = (int)(roi_in->width * scale);
  roi_in->height = (int)(roi_in->height * scale);
}

void modify_roi_out(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece, dt_iop_roi_t *roi_out,
                    const dt_iop_roi_t *roi_in)
{
  dt_iop_rawdenoise_data_t *d = (dt_iop_rawdenoise_data_t *)piece->data;
  float scale = 1.0 / d->threshold;
  roi_out->x = roi_in->x;
  roi_out->y = roi_in->y;
  roi_out->scale = roi_in->scale; // * scale;
  roi_out->width = (int)(roi_in->width / scale);
  roi_out->height = (int)(roi_in->height / scale);
}
void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_rawdenoise_data_t *d = (dt_iop_rawdenoise_data_t *)piece->data;

  const int width = roi_in->width;
  const int height = roi_in->height;
  const uint32_t filters = piece->pipe->dsc.filters;
  const uint8_t(*const xtrans)[6] = (const uint8_t(*const)[6])piece->pipe->dsc.xtrans;

  if (d->mode == MODE_WAVELETS)
  {
    if(!(d->threshold > 0.0f))
    {
      memcpy(ovoid, ivoid, (size_t)sizeof(float)*width*height);
    }
    else
    {
      if (filters != 9u)
        wavelet_denoise(ivoid, ovoid, roi_in, d->threshold, filters);
      else
        wavelet_denoise_xtrans(ivoid, ovoid, roi_in, d->threshold, xtrans);
    }
  }
  else
  {
    printf("%d\n", dt_control_get_dev_closeup());
    printf("%f\n", dt_control_get_dev_zoom_x());
    dt_dev_zoom_t zoom = dt_control_get_dev_zoom();
    int closeup = dt_control_get_dev_closeup();
    if (piece->pipe->type == DT_DEV_PIXELPIPE_FULL)
      printf("%f\n", dt_dev_get_zoom_scale(self->dev, zoom, closeup ? 2.0 : 1.0, 0));
    else if (piece->pipe->type == DT_DEV_PIXELPIPE_PREVIEW)
      printf("%f\n", dt_dev_get_zoom_scale(self->dev, zoom, closeup ? 2.0 : 1.0, 1));

    int target_w = roi_in->width * d->threshold;
    int target_h = roi_in->height * d->threshold;
    // float *const half_ivoid = (float *const)raw_downscale(ivoid, (dt_iop_roi_t*)roi_in, (dt_iop_roi_t*)roi_out,
    // filters, xtrans, target_w, target_h);
    float *half_ivoid;
    if(filters == 9u)
    {
      half_ivoid = (float *const)halfscale_cfa(ivoid, (dt_iop_roi_t *)roi_in, (dt_iop_roi_t *)roi_out, filters,
                                               xtrans, 1.0 / d->threshold);
      float *out = (float *)ovoid;
      assert(out != NULL);
      for(int j = 0; j < target_h; j++)
      {
        for(int i = 0; i < target_w; i++)
        {
          out[j * roi_out->width + i] = half_ivoid[j * roi_in->width + i];
        }
      }
      free(half_ivoid);
    }
    else
    {
      half_ivoid = (float *const)downscale_bilinear_bayer_cfa(ivoid, roi_in->width, roi_in->height,
                                                              roi_in->width * d->threshold,
                                                              roi_in->height * d->threshold, filters);
      float *out = (float *)ovoid;
      assert(out != NULL);
      for(int j = 0; j < target_h; j++)
      {
        for(int i = 0; i < target_w; i++)
        {
          out[j * roi_out->width + i] = half_ivoid[j * target_w + i];
        }
      }
      free(half_ivoid);
    }
    // nlm_denoise(half_ivoid, ovoid, roi_in, roi_out, d->threshold, filters, piece, xtrans);
    // dt_iop_clip_and_zoom_mosaic_half_size_f(ovoid, ivoid, roi_out, roi_in,
    //                                         roi_out->width, roi_in->width, filters);
    // dt_iop_clip_and_zoom_mosaic_third_size_xtrans_f(ovoid, ivoid, roi_out, roi_in,
    //                                           roi_out->width, roi_in->width, xtrans);
  }
}

void reload_defaults(dt_iop_module_t *module)
{
  // init defaults:
  dt_iop_rawdenoise_params_t tmp = (dt_iop_rawdenoise_params_t){ .threshold = 0.01, .mode = MODE_NLMEANS };

  // we might be called from presets update infrastructure => there is no image
  if(!module->dev) goto end;

  // can't be switched on for non-raw images:
  if(dt_image_is_raw(&module->dev->image_storage))
    module->hide_enable_button = 0;
  else
    module->hide_enable_button = 1;
  module->default_enabled = 0;

end:
  memcpy(module->params, &tmp, sizeof(dt_iop_rawdenoise_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_rawdenoise_params_t));
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
  d->details = p->details;
  d->mode = p->mode;

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
  dt_bauhaus_slider_set(g->details, p->details);
  dt_bauhaus_combobox_set(g->mode, p->mode);

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

static void details_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_rawdenoise_params_t *p = (dt_iop_rawdenoise_params_t *)self->params;
  p->details = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void mode_callback(GtkWidget *w, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_rawdenoise_params_t *p = (dt_iop_rawdenoise_params_t *)self->params;
  p->mode = dt_bauhaus_combobox_get(w);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_rawdenoise_gui_data_t));
  dt_iop_rawdenoise_gui_data_t *g = (dt_iop_rawdenoise_gui_data_t *)self->gui_data;

  self->widget = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));

  g->stack = gtk_stack_new();
  gtk_stack_set_homogeneous(GTK_STACK(g->stack), FALSE);
  gtk_box_pack_start(GTK_BOX(self->widget), g->stack, TRUE, TRUE, 0);

  g->box_raw = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  /* mode */
  g->mode = dt_bauhaus_combobox_new(self);
  gtk_box_pack_start(GTK_BOX(g->box_raw), GTK_WIDGET(g->mode), TRUE, TRUE, 0);
  dt_bauhaus_widget_set_label(g->mode, NULL, _("mode"));
  dt_bauhaus_combobox_add(g->mode, _("raw downscaling"));
  dt_bauhaus_combobox_add(g->mode, _("wavelets"));
  gtk_widget_set_tooltip_text(g->mode, _("method used in the denoising core."));
  g_signal_connect(G_OBJECT(g->mode), "value-changed", G_CALLBACK(mode_callback), self);

  /* threshold */
  g->threshold = dt_bauhaus_slider_new_with_range(self, 0.0, 1.0f, 0.001, 0.2f, 3);
  gtk_box_pack_start(GTK_BOX(g->box_raw), GTK_WIDGET(g->threshold), TRUE, TRUE, 0);
  dt_bauhaus_widget_set_label(g->threshold, NULL, _("scaling factor"));
  g_signal_connect(G_OBJECT(g->threshold), "value-changed", G_CALLBACK(threshold_callback), self);

  /* details */
  g->details = dt_bauhaus_slider_new_with_range(self, 0.0, 1.0f, 0.01f, 0.5f, 3);
  gtk_box_pack_start(GTK_BOX(g->box_raw), GTK_WIDGET(g->details), TRUE, TRUE, 0);
  dt_bauhaus_widget_set_label(g->details, NULL, _("details reconstruction"));
  g_signal_connect(G_OBJECT(g->details), "value-changed", G_CALLBACK(details_callback), self);
  gtk_widget_set_tooltip_text(g->details, _("restore fine details from input after the denoising step"));


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
  free(self->gui_data);
  self->gui_data = NULL;
}
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
