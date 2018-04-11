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

DT_MODULE_INTROSPECTION(1, dt_iop_rawdenoise_params_t)

typedef struct dt_iop_rawdenoise_params_t
{
  float threshold;
} dt_iop_rawdenoise_params_t;

typedef struct dt_iop_rawdenoise_gui_data_t
{
  GtkWidget *stack;
  GtkWidget *box_raw;
  GtkWidget *threshold;
  GtkWidget *label_non_raw;
} dt_iop_rawdenoise_gui_data_t;

typedef struct dt_iop_rawdenoise_data_t
{
  float threshold;
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

#if 0
static void median_denoise(const float *const in, float *const out, const dt_iop_roi_t *const roi,
                            float threshold, uint32_t filters)
{
  int radius = 1;
  for (int row = 0; row < roi->height; row++) {
    for (int col = 0; col < roi->width; col++) {
      float* outp = out + (size_t)row * roi->width + col;
      if (row < 2 * radius || col < 2 * radius || row >= roi->height - 2 * radius || col >= roi->width - 2 * radius)
      {
        const float* inp = in + (size_t)row * roi->width + col;
        *outp = *inp;
      }
      else
      {
        if (((filters == 0x16161616 || filters == 0x94949494) && ((row & 1) == 0) && ((col & 1) == 0)) // row and col even, with pattern starting by B or R
        ||  ((filters == 0x61616161 || filters == 0x49494949) && (((row + col) & 1) == 1))) // pattern starts with G
        {
          float* in_data = malloc((radius * 2 + 1) * (radius * 2 + 1) * sizeof(float));
          for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
              in_data[(i + radius) * (radius * 2 + 1) + j + radius] = *(in + (size_t)((row + 2 * i) * roi->width + col + 2 * j));
            }
          }

          bool median_not_found = true;
          int median_candidate = 0;
          while (median_not_found)
          {
            int number_of_lower = 0;
            int number_of_strictly_lower = 0;
            for (int i = 0; i < (radius * 2 + 1) * (radius * 2 + 1); i++)
            {
              if (i != median_candidate && in_data[i] <= in_data[median_candidate])
                number_of_lower++;
              if (i != median_candidate && in_data[i] < in_data[median_candidate])
                number_of_strictly_lower++;
            }
            if (number_of_lower == radius * radius * 4)
              median_not_found = false;
            else
              if (number_of_lower > radius * radius * 4 && number_of_strictly_lower <= radius * radius * 4)
                median_not_found = false;
              else
                median_candidate++;
          }
          float mean = 0.0f;
          for (int i = 0; i < (radius * 2 + 1) * (radius * 2 + 1); i++) {
            mean += in_data[i];
          }
          mean = mean / (float)((radius * 2 + 1) * (radius * 2 + 1));

          *outp = in_data[median_candidate];
          free(in_data);
        }
        else
        {
          const float* inp = in + (size_t)row * roi->width + col;
          *outp = *inp;
        }
      }
    }
  }
}
#endif

#define ELEM_SWAP(a,b) { float t=(a);(a)=(b);(b)=t; }
float kth_smallest(float a[], int n, int k)
{
    int i,j,l,m ;
    float x ;

    l=0 ; m=n-1 ;
    while (l<m) {
        x=a[k] ;
        i=l ;
        j=m ;
        do {
            while (a[i]<x) i++ ;
            while (x<a[j]) j-- ;
            if (i<=j) {
                ELEM_SWAP(a[i],a[j]) ;
                i++ ; j-- ;
            }
        } while (i<=j) ;
        if (j<k) l=i ;
        if (k<i) m=j ;
    }
    return a[k] ;
}

#define median(a,n) kth_smallest(a,n,(((n)&1)?((n)/2):(((n)/2)-1)))

#define MEDIAN_W 0.7f

static void median_mean_bayer(const float *const ivoid, uint32_t filters, float* medians, const dt_iop_roi_t *const roi_in)
{
  float* inp = (float*)ivoid;
  int row_offset = 2;
  int col_offset = 2;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int row = row_offset; row < roi_in->height-row_offset; row++) {
    float arrayf[9];
    for (int col = col_offset; col < roi_in->width-col_offset; col++) {
      arrayf[0] = inp[row * roi_in->width + col];
      if (FC(row, col, filters) == 1) {
        // Green
        arrayf[1] = inp[(row + 1) * roi_in->width + col + 1];
        arrayf[2] = inp[(row + 1) * roi_in->width + col - 1];
        arrayf[3] = inp[(row - 1) * roi_in->width + col + 1];
        arrayf[4] = inp[(row - 1) * roi_in->width + col - 1];
        arrayf[5] = inp[(row + 2) * roi_in->width + col];
        arrayf[6] = inp[(row - 2) * roi_in->width + col];
        arrayf[7] = inp[(row) * roi_in->width + col + 2];
        arrayf[8] = inp[(row) * roi_in->width + col - 2];
        medians[row * roi_in->width + col] = MEDIAN_W * median(arrayf, 9) + (1.0f - MEDIAN_W) * inp[row * roi_in->width + col];
      } else {
        // Red or Blue
        arrayf[1] = inp[(row + 2) * roi_in->width + col];
        arrayf[2] = inp[(row - 2) * roi_in->width + col];
        arrayf[3] = inp[(row) * roi_in->width + col + 2];
        arrayf[4] = inp[(row) * roi_in->width + col - 2];
        arrayf[5] = inp[(row + 2) * roi_in->width + col + 2];
        arrayf[6] = inp[(row - 2) * roi_in->width + col - 2];
        arrayf[7] = inp[(row - 2) * roi_in->width + col + 2];
        arrayf[8] = inp[(row + 2) * roi_in->width + col - 2];
        medians[row * roi_in->width + col] = median(arrayf, 9);
      }
    }
  }
}

/* pattern_period = 3 or 6 depending of if we simply want the position of green pixels without differenciating R from B pixels,
  or if we want the complete info */
static void xtrans_pattern_shift(int* row_shift, int* col_shift, const uint8_t(*const xtrans)[6], const dt_iop_roi_t *const roi_in, const int pattern_period) {
  //look for pattern start: find first GRB or GBR pattern
  int row_start = 0;
  int col_start = 0;
  bool start_not_found = true;
  while(start_not_found) {
    if (col_start > roi_in->width-3) {
      col_start = 0;
      row_start++;
    }
    if (row_start >= roi_in->height) {
      printf("Problem: start of xtrans pattern not found!!!");
      row_start = 0;
      col_start = 0;
      start_not_found = false;
    }

    int color1 = FCxtrans(row_start, col_start, roi_in, xtrans);
    int color2 = FCxtrans(row_start, col_start + 1, roi_in, xtrans);
    int color3 = FCxtrans(row_start, col_start + 2, roi_in, xtrans);

    if ((color1 == 1) && ((color2 ^ color3) == 2)) {
      start_not_found = false;
    } else {
      col_start++;
    }
  }
  *row_shift = (pattern_period - (row_start % pattern_period)) % pattern_period;
  *col_shift = (pattern_period - (col_start % pattern_period)) % pattern_period;
}

static void median_mean_xtrans(const float *const ivoid, const uint8_t(*const xtrans)[6], float* medians, const dt_iop_roi_t *const roi_in)
{
  float* inp = (float*)ivoid;

  const int pattern_period = 3; // Here, we don't need to differenciate R and B
  int row_shift = 0;
  int col_shift = 0;

  xtrans_pattern_shift(&row_shift, &col_shift, xtrans, roi_in, pattern_period);

  int row_offset = 6;
  int col_offset = 6;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int row = row_offset; row < roi_in->height-row_offset; row++) {
    for (int col = col_offset; col < roi_in->width-col_offset; col++) {
      int position = ((row + row_shift) % pattern_period) * pattern_period + (col + col_shift) % pattern_period;
      float arrayf[10];
      arrayf[0] = inp[row * roi_in->width + col];
      switch(position) {
        case 0:
          arrayf[1] = inp[(row + 1) * roi_in->width + col + 1];
          arrayf[2] = inp[(row + 1) * roi_in->width + col - 1];
          arrayf[3] = inp[(row - 1) * roi_in->width + col + 1];
          arrayf[4] = inp[(row - 1) * roi_in->width + col - 1];
          arrayf[5] = inp[(row + 3) * roi_in->width + col];
          arrayf[6] = inp[(row - 3) * roi_in->width + col];
          arrayf[7] = inp[(row) * roi_in->width + col + 3];
          arrayf[8] = inp[(row) * roi_in->width + col - 3];
          medians[row * roi_in->width + col] = MEDIAN_W * median(arrayf, 9) + (1.0f - MEDIAN_W) * inp[row * roi_in->width + col];
          break;
        case 1:
          arrayf[1] = inp[(row) * roi_in->width + col - 2];
          arrayf[2] = inp[(row - 2) * roi_in->width + col - 1];
          arrayf[3] = inp[(row + 2) * roi_in->width + col - 1];
          arrayf[4] = inp[(row - 1) * roi_in->width + col + 2];
          arrayf[5] = inp[(row + 1) * roi_in->width + col + 2];
          arrayf[6] = inp[(row + 6) * roi_in->width + col];
          arrayf[7] = inp[(row - 6) * roi_in->width + col];
          arrayf[8] = inp[(row) * roi_in->width + col + 6];
          arrayf[9] = inp[(row) * roi_in->width + col - 6];
          medians[row * roi_in->width + col] = 0.5f * (kth_smallest(arrayf, 10, 4) + kth_smallest(arrayf, 10, 5));
          break;
        case 2:
          arrayf[1] = inp[(row) * roi_in->width + col + 2];
          arrayf[2] = inp[(row - 2) * roi_in->width + col + 1];
          arrayf[3] = inp[(row + 2) * roi_in->width + col + 1];
          arrayf[4] = inp[(row - 1) * roi_in->width + col - 2];
          arrayf[5] = inp[(row + 1) * roi_in->width + col - 2];
          arrayf[6] = inp[(row + 6) * roi_in->width + col];
          arrayf[7] = inp[(row - 6) * roi_in->width + col];
          arrayf[8] = inp[(row) * roi_in->width + col + 6];
          arrayf[9] = inp[(row) * roi_in->width + col - 6];
          medians[row * roi_in->width + col] = 0.5f * (kth_smallest(arrayf, 10, 4) + kth_smallest(arrayf, 10, 5));
          break;
        case 3:
          arrayf[1] = inp[(row - 2) * roi_in->width + col];
          arrayf[2] = inp[(row - 1) * roi_in->width + col - 2];
          arrayf[3] = inp[(row - 1) * roi_in->width + col + 2];
          arrayf[4] = inp[(row + 2) * roi_in->width + col - 1];
          arrayf[5] = inp[(row + 2) * roi_in->width + col + 1];
          arrayf[6] = inp[(row + 6) * roi_in->width + col];
          arrayf[7] = inp[(row - 6) * roi_in->width + col];
          arrayf[8] = inp[(row) * roi_in->width + col + 6];
          arrayf[9] = inp[(row) * roi_in->width + col - 6];
          medians[row * roi_in->width + col] = 0.5f * (kth_smallest(arrayf, 10, 4) + kth_smallest(arrayf, 10, 5));
          break;
        case 4:
          arrayf[1] = inp[(row - 1) * roi_in->width + col - 1];
          arrayf[2] = inp[(row + 1) * roi_in->width + col];
          arrayf[3] = inp[(row) * roi_in->width + col + 1];
          arrayf[4] = inp[(row + 1) * roi_in->width + col + 1];
          arrayf[5] = inp[(row + 3) * roi_in->width + col];
          arrayf[6] = inp[(row - 3) * roi_in->width + col];
          arrayf[7] = inp[(row) * roi_in->width + col + 3];
          arrayf[8] = inp[(row) * roi_in->width + col - 3];
          medians[row * roi_in->width + col] = MEDIAN_W * median(arrayf, 9) + (1.0f - MEDIAN_W) * inp[row * roi_in->width + col];
          break;
        case 5:
          arrayf[1] = inp[(row - 1) * roi_in->width + col + 1];
          arrayf[2] = inp[(row + 1) * roi_in->width + col];
          arrayf[3] = inp[(row) * roi_in->width + col - 1];
          arrayf[4] = inp[(row + 1) * roi_in->width + col - 1];
          arrayf[5] = inp[(row + 3) * roi_in->width + col];
          arrayf[6] = inp[(row - 3) * roi_in->width + col];
          arrayf[7] = inp[(row) * roi_in->width + col + 3];
          arrayf[8] = inp[(row) * roi_in->width + col - 3];
          medians[row * roi_in->width + col] = MEDIAN_W * median(arrayf, 9) + (1.0f - MEDIAN_W) * inp[row * roi_in->width + col];
          break;
        case 6:
          arrayf[1] = inp[(row + 2) * roi_in->width + col];
          arrayf[2] = inp[(row + 1) * roi_in->width + col - 2];
          arrayf[3] = inp[(row + 1) * roi_in->width + col + 2];
          arrayf[4] = inp[(row - 2) * roi_in->width + col - 1];
          arrayf[5] = inp[(row - 2) * roi_in->width + col + 1];
          arrayf[6] = inp[(row + 6) * roi_in->width + col];
          arrayf[7] = inp[(row - 6) * roi_in->width + col];
          arrayf[8] = inp[(row) * roi_in->width + col + 6];
          arrayf[9] = inp[(row) * roi_in->width + col - 6];
          medians[row * roi_in->width + col] = 0.5f * (kth_smallest(arrayf, 10, 4) + kth_smallest(arrayf, 10, 5));
          break;
        case 7:
          arrayf[1] = inp[(row + 1) * roi_in->width + col - 1];
          arrayf[2] = inp[(row - 1) * roi_in->width + col];
          arrayf[3] = inp[(row) * roi_in->width + col + 1];
          arrayf[4] = inp[(row - 1) * roi_in->width + col + 1];
          arrayf[5] = inp[(row + 3) * roi_in->width + col];
          arrayf[6] = inp[(row - 3) * roi_in->width + col];
          arrayf[7] = inp[(row) * roi_in->width + col + 3];
          arrayf[8] = inp[(row) * roi_in->width + col - 3];
          medians[row * roi_in->width + col] = MEDIAN_W * median(arrayf, 9) + (1.0f - MEDIAN_W) * inp[row * roi_in->width + col];
          break;
        case 8:
          arrayf[1] = inp[(row + 1) * roi_in->width + col + 1];
          arrayf[2] = inp[(row - 1) * roi_in->width + col];
          arrayf[3] = inp[(row) * roi_in->width + col - 1];
          arrayf[4] = inp[(row - 1) * roi_in->width + col - 1];
          arrayf[5] = inp[(row + 3) * roi_in->width + col];
          arrayf[6] = inp[(row - 3) * roi_in->width + col];
          arrayf[7] = inp[(row) * roi_in->width + col + 3];
          arrayf[8] = inp[(row) * roi_in->width + col - 3];
          medians[row * roi_in->width + col] = MEDIAN_W * median(arrayf, 9) + (1.0f - MEDIAN_W) * inp[row * roi_in->width + col];
          break;
      }
    }
  }
}

#define NORM 1

typedef union floatint_t
{
  float f;
  uint32_t i;
} floatint_t;

// very fast approximation for 2^-x (returns 0 for x > 126)
static inline float fast_mexp2f(const float x)
{
  const float i1 = (float)0x3f800000u; // 2^0
  const float i2 = (float)0x3f000000u; // 2^-1
  const float k0 = i1 + x * (i2 - i1);
  floatint_t k;
  k.i = k0 >= (float)0x800000u ? k0 : 0;
  return k.f;
}

static void nlm_denoise(const float *const ivoid, float *const ovoid, const dt_iop_roi_t *const roi_in,
                            const dt_iop_roi_t *const roi_out, const float threshold, const uint32_t filters, dt_dev_pixelpipe_iop_t *piece, const uint8_t(*const xtrans)[6])
{
  const int P = ceilf(2.0f * fmin(roi_in->scale, 2.0f) / fmax(piece->iscale, 1.0f));

  int raw_patern_size = 2;
  if (filters == 9u)
    raw_patern_size = 6;

  float *medians = calloc((size_t)sizeof(float), roi_out->width * roi_out->height);
  if (filters != 9u) {
    median_mean_bayer(ivoid, filters, medians, roi_in);
  } else {
    median_mean_xtrans(ivoid, xtrans, medians, roi_in);
  }

  const int K = ceilf(7 * fmin(roi_in->scale, 2.0f) / fmax(piece->iscale, 1.0f)) * raw_patern_size;

  float *Sa = dt_alloc_align(64, (size_t)sizeof(float) * roi_out->width * dt_get_num_threads());
  int *Na = dt_alloc_align(64, (size_t)sizeof(int) * roi_out->width * dt_get_num_threads());
  // we want to sum up weights in col[3], so need to init to 0:
  memset(ovoid, 0x0, (size_t)sizeof(float) * roi_out->width * roi_out->height);
  //float *in = dt_alloc_align(64, (size_t)sizeof(float) * roi_in->width * roi_in->height);
  double *const norms = (double*)calloc(roi_out->width * roi_out->height, sizeof(double));

  float *in = (float *)medians;
  in = (float*)ivoid;

  // for each shift vector
  for(int kj = -K; kj <= K; kj+=raw_patern_size)
  {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) default(none) shared(in, Sa, Na, kj, raw_patern_size)
    #endif
    for(int ki = -K; ki <= 0; ki+=raw_patern_size)
    {
      if ((2*K+1)*ki+kj >= 0)
        continue;

      // TODO: adaptive K tests here!
      // TODO: expf eval for real bilateral experience :)

      if ((kj == 0) && (ki == 0))
        continue;
      int inited_slide = 0;
// don't construct summed area tables but use sliding window! (applies to cpu version res < 1k only, or else
// we will add up errors)
// do this in parallel with a little threading overhead. could parallelize the outer loops with a bit more
// memory

      for(int j = 0; j < roi_out->height; j++)
      {
        if(j + kj < 0 || j + kj >= roi_out->height) continue;
        float *S = Sa + dt_get_thread_num() * roi_out->width;
        int *N = Na + dt_get_thread_num() * roi_out->width;
        const float *ins = in + ((size_t)roi_in->width * (j + kj) + ki);
        float *out = ((float *)ovoid) + (size_t)roi_out->width * j;
        double *norm_j = ((double *)norms) + (size_t)roi_out->width * j;

        const int Pm = MIN(MIN(P, j + kj), j);
        const int PM = MIN(MIN(P, roi_out->height - 1 - j - kj), roi_out->height - 1 - j);
        // first line of every thread
        // TODO: also every once in a while to assert numerical precision!
        if(!inited_slide)
        {
          // sum up a line
          memset(S, 0x0, sizeof(float) * roi_out->width);
          memset(N, 0x0, sizeof(int) * roi_out->width);
          for(int jj = -Pm; jj <= PM; jj++)
          {
            int i = MAX(0, -ki);
            float *s = S + i;
            int *n = N + i;
            const float *inp = in + i + (size_t)roi_in->width * (j + jj);
            const float *inps = in + i + ((size_t)roi_in->width * (j + jj + kj) + ki);
            const int last = roi_out->width + MIN(0, -ki);
            for(; i < last; i++, inp ++, inps ++, s++, n++)
            {
              int color;
              if (filters == 9u)
                color = FCxtrans(j + jj, i, roi_in, xtrans);
              else
                color = FC(j + jj, i, filters);
              if (color == 1) {
                s[0] += fabs(inp[0] - inps[0]);
                n[0]++;
              }
            }
          }
          // only reuse this if we had a full stripe
          if(Pm == P && PM == P) inited_slide = 1;
        }

        // sliding window for this line:
        float *s = S;
        int *n = N;
        float slide = 0.0f;
        float norm = 0.0f;
        // sum up the first -P..P
        for(int i = 0; i < 2 * P + 1; i++)
        {
          slide += s[i];
          norm += n[i];
        }
        for(int i = 0; i < roi_out->width; i++, s++, ins ++, out ++, norm_j++, n++)
        {
          // FIXME: the comment above is actually relevant even for 1000 px width already.
          // XXX    numerical precision will not forgive us:
          if(i - P > 0 && i + P < roi_out->width)
          {
            slide += s[P] - s[-P - 1];
            norm += n[P] - n[-P - 1];
          }
          if(i + ki >= 0 && i + ki < roi_out->width)
          {
            // TODO: could put that outside the loop.
            // DEBUG XXX bring back to computable range:
            double weight = fast_mexp2f(fmaxf(0.0f, slide / norm * .015f * 500000.0f / (1.0f + 100.0f * threshold) - 2.0f));
            out[0] += ins[0] * weight;
            norm_j[0] += weight;
            (out + (size_t)roi_in->width * kj + ki)[0] += (ins - (size_t)roi_in->width * kj - ki)[0] * weight;
            (norm_j + (size_t)roi_in->width * kj + ki)[0] += weight;
          }
        }
        if(inited_slide && j + P + 1 + MAX(0, kj) < roi_out->height)
        {
          // sliding window in j direction:
          int i = MAX(0, -ki);
          s = S + i;
          n = N + i;
          const float *inp = in + i + (size_t)roi_in->width * (j + P + 1);
          const float *inps = in + i + ((size_t)roi_in->width * (j + P + 1 + kj) + ki);
          const float *inm = in + i + (size_t)roi_in->width * (j - P);
          const float *inms = in + i + ((size_t)roi_in->width * (j - P + kj) + ki);
          const int last = roi_out->width + MIN(0, -ki);
          for(; i < last; i++, inp ++, inps ++, inm ++, inms ++, s++)
          {
            int color;
            if (filters == 9u)
              color = FCxtrans(j + P + 1, i, roi_in, xtrans);
            else
              color = FC(j + P + 1, i, filters);
            //attention, enlever inm inms que si eux aussi sont un pixel vert !!!
            if (color == 1) {
              s[0] += fabs(inp[0] - inps[0]);
              n[0]++;
            }
            if (filters == 9u)
              color = FCxtrans(j - P, i, roi_in, xtrans);
            else
              color = FC(j - P, i, filters);
            if (color == 1) {
              s[0] -= fabs(inm[0] - inms[0]);
              n[0]--;
            }
          }
        }
        else
           inited_slide = 0;
      }
    }
  }

  // free shared tmp memory:
  dt_free_align(Sa);
  dt_free_align(Na);
  //dt_free_align(in);

  float *out = (float *)ovoid;
  for(int j = 0; j < roi_out->height; j++)
  {
    for(int i = 0; i < roi_out->width; i++)
    {
      // printf("out: %f\n", norms[j*(roi_out->width)+i]);
      if (norms[j*(roi_out->width)+i] <= 0.00001f) {
        if ((i > raw_patern_size) && (j > raw_patern_size) && (i < roi_out->width - raw_patern_size) && (j < roi_out->height - raw_patern_size)) {
          /* perform quadratic mean */
          float mean = 0.0f;
          float tmp = 0.0f;
          tmp = in[j*(roi_out->width)+i];
          mean += tmp;
          tmp = in[(j + raw_patern_size)*(roi_out->width)+i];
          mean += tmp;
          tmp = in[(j - raw_patern_size)*(roi_out->width)+i];
          mean += tmp;
          tmp = in[j*(roi_out->width)+i+raw_patern_size];
          mean += tmp;
          tmp = in[j*(roi_out->width)+i-raw_patern_size];
          mean += tmp;
          mean = mean / 5.0f;
          out[j*(roi_out->width)+i] = mean;
        } else {
          out[j*(roi_out->width)+i] = in[j*(roi_out->width)+i];
        }
      } else
        out[j*(roi_out->width)+i] = (float)(((double)out[j*(roi_out->width)+i]) / norms[j*(roi_out->width)+i]);
        out[j*(roi_out->width)+i] = (0.6f * out[j*(roi_out->width)+i] + 0.2f * in[j*(roi_out->width)+i] + 0.2f * sqrt(0.2f * in[j*(roi_out->width)+i] * in[j*(roi_out->width)+i] + 0.8f * out[j*(roi_out->width)+i] * out[j*(roi_out->width)+i]));
    }
  }

  free(medians);

#if 0



  if(P < 1)
  {
    // nothing to do from this distance:
    memcpy(ovoid, ivoid, (size_t)sizeof(float) * roi_out->width * roi_out->height);
    return;
  }

  float* medians = (float*)calloc(roi_out->width * roi_out->height, sizeof(float));
  double* sum_values = (double*)calloc(roi_out->width * roi_out->height, sizeof(double));
  double* sum_weights = (double*)calloc(roi_out->width * roi_out->height, sizeof(double));

  float* inp = (float*)ivoid;
  float* outp = (float*)ovoid;

  if (filters != 9u) {
    median_mean_bayer(ivoid, filters, medians, roi_in);
  } else {
    median_mean_xtrans(ivoid, xtrans, medians, roi_in);
  }

  for (int row = 0; row < K+P; row++)
  {
    int row_last = roi_out->height - row - 1;
    for (int col = 0; col < roi_out->width; col++)
    {
      outp[row * roi_out->width + col] = inp[row * roi_out->width + col];
      outp[row_last * roi_out->width + col] = inp[row_last * roi_out->width + col];
    }
  }
  // particular cases: top left and top right columns
  for (int col = 0; col < K+P; col++)
  {
    int col_last = roi_out->width - col - 1;
    for (int row = 0; row < roi_out->height; row++)
    {
      outp[row * roi_out->width + col] = inp[row * roi_out->width + col];
      outp[row * roi_out->width + col_last] = inp[row * roi_out->width + col_last];
    }
  }


  const int G_pattern_period_xtrans = 3; // Here, we don't need to differenciate R and B
  int row_shift = 0;
  int col_shift = 0;

  if (filters == 9u)
    xtrans_pattern_shift(&row_shift, &col_shift, xtrans, roi_in, G_pattern_period_xtrans);

  // search for patches at a distance <= K
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int row_offset = -K; row_offset <= K; row_offset+=raw_patern_size)
  {
    for (int col_offset = -K; col_offset <= 0; col_offset+=raw_patern_size)
    {
      if ((2*K+1)*col_offset+row_offset >= 0)
      continue;

      for (int row = K+P; row < roi_out->height-K-P; row++)
      {
        for (int col = K+P; col < roi_out->width-K-P; col++)
        {
          int color;
          if (filters == 9u)
          color = FCxtrans(row, col, roi_in, xtrans);
          else
          color = FC(row, col, filters);

          if (color != 1 /*1 means green*/)
          continue;


          // compute weight
          float weight = 0.0f;

          float diff = 0.0f;
          float normalize = 0.0f;
          for (int cmp_row_offset = -P; cmp_row_offset <= P; cmp_row_offset++) {
            for (int cmp_col_offset = -P; cmp_col_offset <= P; cmp_col_offset++) {
              int color_tmp;
              if (filters == 9u)
              color_tmp = FCxtrans(row + cmp_row_offset, col + cmp_col_offset, roi_in, xtrans);
              else
              color_tmp = FC(row + cmp_row_offset, col + cmp_col_offset, filters);
              if (color_tmp == 1) /* green*/
              {
                normalize += 1.0f;
                diff = medians[(row + row_offset + cmp_row_offset) * roi_out->width + col + col_offset + cmp_col_offset]
                -medians[(row + cmp_row_offset) * roi_out->width + col + cmp_col_offset];
                #if NORM == 1
                weight += fabs(diff);
                #else
                weight += diff * diff;
                #endif
              } else {
                normalize += 0.3f;
                diff = medians[(row + row_offset + cmp_row_offset) * roi_out->width + col + col_offset + cmp_col_offset]
                -medians[(row + cmp_row_offset) * roi_out->width + col + cmp_col_offset];
                #if NORM == 1
                weight += 0.3f * fabs(diff);
                #else
                weight += 0.3f * diff * diff;
                #endif
              }
            }
          }
          weight = weight / normalize; /* normalize as there may be zones where there is more green pixels than in others */

          if ((row_offset != 0) || (col_offset != 0))
          {
            #if NORM == 1
            weight = exp(-weight /(0.00005f + 0.05f * threshold));
            #else
            weight = exp(-weight /(0.00005f + 0.001f * threshold));
            #endif

            // update new_value and sum_weights
            sum_values[row * roi_out->width + col] += weight * medians[((row + row_offset) * roi_out->width + (col + col_offset))];
            sum_weights[row * roi_out->width + col] += weight;

            int position;
            if (filters != 9u) {
              sum_values[(row - 1) * roi_out->width + col] += weight * medians[((row + row_offset - 1) * roi_out->width + (col + col_offset))];
              sum_weights[(row - 1) * roi_out->width + col] += weight;
              sum_values[(row + 1) * roi_out->width + col] += weight * medians[((row + row_offset + 1) * roi_out->width + (col + col_offset))];
              sum_weights[(row + 1) * roi_out->width + col] += weight;
              sum_values[(row) * roi_out->width + col - 1] += weight * medians[((row + row_offset) * roi_out->width + (col + col_offset - 1))];
              sum_weights[(row) * roi_out->width + col - 1] += weight;
              sum_values[(row) * roi_out->width + col + 1] += weight * medians[((row + row_offset) * roi_out->width + (col + col_offset + 1))];
              sum_weights[(row) * roi_out->width + col + 1] += weight;
            } else {
              position = ((row + row_shift) % G_pattern_period_xtrans) * G_pattern_period_xtrans + (col + col_shift) % G_pattern_period_xtrans;
              switch (position) {
                case 0:
                sum_values[(row - 1) * roi_out->width + col] += weight * medians[((row + row_offset - 1) * roi_out->width + (col + col_offset))];
                sum_weights[(row - 1) * roi_out->width + col] += weight;
                sum_values[(row + 1) * roi_out->width + col] += weight * medians[((row + row_offset + 1) * roi_out->width + (col + col_offset))];
                sum_weights[(row + 1) * roi_out->width + col] += weight;
                sum_values[(row) * roi_out->width + col - 1] += weight * medians[((row + row_offset) * roi_out->width + (col + col_offset - 1))];
                sum_weights[(row) * roi_out->width + col - 1] += weight;
                sum_values[(row) * roi_out->width + col + 1] += weight * medians[((row + row_offset) * roi_out->width + (col + col_offset + 1))];
                sum_weights[(row) * roi_out->width + col + 1] += weight;
                break;
                case 4:
                sum_values[(row - 1) * roi_out->width + col] += weight * medians[((row + row_offset - 1) * roi_out->width + (col + col_offset))];
                sum_weights[(row - 1) * roi_out->width + col] += weight;
                sum_values[(row) * roi_out->width + col - 1] += weight * medians[((row + row_offset) * roi_out->width + (col + col_offset - 1))];
                sum_weights[(row) * roi_out->width + col - 1] += weight;
                break;
                case 5:
                sum_values[(row - 1) * roi_out->width + col] += weight * medians[((row + row_offset - 1) * roi_out->width + (col + col_offset))];
                sum_weights[(row - 1) * roi_out->width + col] += weight;
                sum_values[(row) * roi_out->width + col + 1] += weight * medians[((row + row_offset) * roi_out->width + (col + col_offset + 1))];
                sum_weights[(row) * roi_out->width + col + 1] += weight;
                break;
                case 7:
                sum_values[(row + 1) * roi_out->width + col] += weight * medians[((row + row_offset + 1) * roi_out->width + (col + col_offset))];
                sum_weights[(row + 1) * roi_out->width + col] += weight;
                sum_values[(row) * roi_out->width + col - 1] += weight * medians[((row + row_offset) * roi_out->width + (col + col_offset - 1))];
                sum_weights[(row) * roi_out->width + col - 1] += weight;
                break;
                case 8:
                sum_values[(row + 1) * roi_out->width + col] += weight * medians[((row + row_offset + 1) * roi_out->width + (col + col_offset))];
                sum_weights[(row + 1) * roi_out->width + col] += weight;
                sum_values[(row) * roi_out->width + col + 1] += weight * medians[((row + row_offset) * roi_out->width + (col + col_offset + 1))];
                sum_weights[(row) * roi_out->width + col + 1] += weight;
                break;
                default:
                printf("Problem !!!\n");
              }

              // update values of the other pixel
              sum_values[(row + row_offset) * roi_out->width + col + col_offset] += weight * medians[row * roi_out->width + col];
              sum_weights[(row + row_offset) * roi_out->width + col + col_offset] += weight;

              if (filters != 9u) {
                sum_values[(row + row_offset - 1) * roi_out->width + col + col_offset] += weight * medians[((row - 1) * roi_out->width + col)];
                sum_weights[(row + row_offset - 1) * roi_out->width + col + col_offset] += weight;
                sum_values[(row + row_offset + 1) * roi_out->width + col + col_offset] += weight * medians[((row + 1) * roi_out->width + col)];
                sum_weights[(row + row_offset + 1) * roi_out->width + col + col_offset] += weight;
                sum_values[(row + row_offset) * roi_out->width + col + col_offset - 1] += weight * medians[((row) * roi_out->width + (col + col_offset - 1))];
                sum_weights[(row + row_offset) * roi_out->width + col + col_offset - 1] += weight;
                sum_values[(row + row_offset) * roi_out->width + col + col_offset + 1] += weight * medians[((row) * roi_out->width + (col + col_offset + 1))];
                sum_weights[(row + row_offset) * roi_out->width + col + col_offset + 1] += weight;
              } else {
                position = ((row + row_shift) % G_pattern_period_xtrans) * G_pattern_period_xtrans + (col + col_shift) % G_pattern_period_xtrans;
                switch (position) {
                  case 0:
                  sum_values[(row + row_offset - 1) * roi_out->width + col + col_offset] += weight * medians[((row - 1) * roi_out->width + col)];
                  sum_weights[(row + row_offset - 1) * roi_out->width + col + col_offset] += weight;
                  sum_values[(row + row_offset + 1) * roi_out->width + col + col_offset] += weight * medians[((row + 1) * roi_out->width + col)];
                  sum_weights[(row + row_offset + 1) * roi_out->width + col + col_offset] += weight;
                  sum_values[(row + row_offset) * roi_out->width + col + col_offset - 1] += weight * medians[((row) * roi_out->width + (col + col_offset - 1))];
                  sum_weights[(row + row_offset) * roi_out->width + col + col_offset - 1] += weight;
                  sum_values[(row + row_offset) * roi_out->width + col + col_offset + 1] += weight * medians[((row) * roi_out->width + (col + col_offset + 1))];
                  sum_weights[(row + row_offset) * roi_out->width + col + col_offset + 1] += weight;
                  break;
                  case 4:
                  sum_values[(row + row_offset - 1) * roi_out->width + col + col_offset] += weight * medians[((row - 1) * roi_out->width + col)];
                  sum_weights[(row + row_offset - 1) * roi_out->width + col + col_offset] += weight;
                  sum_values[(row + row_offset) * roi_out->width + col + col_offset - 1] += weight * medians[((row) * roi_out->width + (col + col_offset - 1))];
                  sum_weights[(row + row_offset) * roi_out->width + col + col_offset - 1] += weight;
                  break;
                  case 5:
                  sum_values[(row + row_offset - 1) * roi_out->width + col + col_offset] += weight * medians[((row - 1) * roi_out->width + col)];
                  sum_weights[(row + row_offset - 1) * roi_out->width + col + col_offset] += weight;
                  sum_values[(row + row_offset) * roi_out->width + col + col_offset + 1] += weight * medians[((row) * roi_out->width + (col + col_offset + 1))];
                  sum_weights[(row + row_offset) * roi_out->width + col + col_offset + 1] += weight;
                  break;
                  case 7:
                  sum_values[(row + row_offset + 1) * roi_out->width + col + col_offset] += weight * medians[((row + 1) * roi_out->width + col)];
                  sum_weights[(row + row_offset + 1) * roi_out->width + col + col_offset] += weight;
                  sum_values[(row + row_offset) * roi_out->width + col + col_offset - 1] += weight * medians[((row) * roi_out->width + (col + col_offset - 1))];
                  sum_weights[(row + row_offset) * roi_out->width + col + col_offset - 1] += weight;
                  break;
                  case 8:
                  sum_values[(row + row_offset + 1) * roi_out->width + col + col_offset] += weight * medians[((row + 1) * roi_out->width + col)];
                  sum_weights[(row + row_offset + 1) * roi_out->width + col + col_offset] += weight;
                  sum_values[(row + row_offset) * roi_out->width + col + col_offset + 1] += weight * medians[((row) * roi_out->width + (col + col_offset + 1))];
                  sum_weights[(row + row_offset) * roi_out->width + col + col_offset + 1] += weight;
                  break;
                  default:
                  printf("Problem !!!\n");
                }
              }
            }
          }
        }
      }
      // if (max_weight < 0.001f) {
      //   max_weight = 0.001f;
      // }
      // if (sum_weights < 0.00001f) {
      //   new_value = medians[(row * roi_out->width + col)];
      // } else {
      //   new_value = new_value / sum_weights;
      // }

      //      float diff_io = inp[index] - new_value;
      //      float abs_diff_io = sqrt(fabs(diff_io))/10.0f;
      //      if (diff_io < 0.0f) {
      //        outp[index] = new_value - abs_diff_io;
      //      } else {
      //        outp[index] = new_value + abs_diff_io;
      //      }
      //outp[index] = new_value;
    }
  }
  for (int row = K+P; row < roi_out->height-K-P; row++)
  {
    for (int col = K+P; col < roi_out->width-K-P; col++)
    {
      if (sum_weights[row * roi_out->width + col] < 0.00001f) {
        outp[row * roi_out->width + col] = medians[row * roi_out->width + col];
      } else {
        outp[row * roi_out->width + col] = sum_values[row * roi_out->width + col] / sum_weights[row * roi_out->width + col];
      }
    }
  }
  free(medians);
  free(sum_values);
  free(sum_weights);
  #endif
}

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

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_rawdenoise_data_t *d = (dt_iop_rawdenoise_data_t *)piece->data;

//  const int width = roi_in->width;
//  const int height = roi_in->height;
  const uint32_t filters = piece->pipe->dsc.filters;

  if(!(d->threshold > 0.0f))
  {
    wavelet_denoise(ivoid, ovoid, roi_in, d->threshold, filters);
    //memcpy(ovoid, ivoid, (size_t)sizeof(float)*width*height);
  }
  else
  {
    if (d->threshold == 1.0f && filters == 9u) {
      const uint8_t(*const xtrans)[6] = (const uint8_t(*const)[6])piece->pipe->dsc.xtrans;
      wavelet_denoise_xtrans(ivoid, ovoid, roi_in, d->threshold, xtrans);
    } else {
      const uint8_t(*const xtrans)[6] = (const uint8_t(*const)[6])piece->pipe->dsc.xtrans;
      nlm_denoise(ivoid, ovoid, roi_in, roi_out, d->threshold, filters, piece, xtrans);
    }
  }
}

void reload_defaults(dt_iop_module_t *module)
{
  // init defaults:
  dt_iop_rawdenoise_params_t tmp = (dt_iop_rawdenoise_params_t){ .threshold = 0.01 };

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

  /* threshold */
  g->threshold = dt_bauhaus_slider_new_with_range(self, 0.0, 1.0f, 0.001, p->threshold, 3);
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
  free(self->gui_data);
  self->gui_data = NULL;
}
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
