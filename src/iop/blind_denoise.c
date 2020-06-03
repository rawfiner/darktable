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
#include "common/fast_guided_filter.h"

#include <gtk/gtk.h>
#include <stdlib.h>
#define NB_SCALES 8

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
  float factor[NB_SCALES];
} dt_iop_blind_denoise_params_t;

typedef struct dt_iop_blind_denoise_self_and_index_t
{
  dt_iop_module_t *self;
  uint32_t n;
} dt_iop_blind_denoise_self_and_index_t;

typedef struct dt_iop_blind_denoise_dir_t
{
  // offset in width. Can be -1, 0, or 1
  int8_t w;
  // offset in height. Can be -1, 0, or 1
  int8_t h;
} dt_iop_blind_denoise_dir_t;

typedef struct dt_iop_blind_denoise_gui_data_t
{
  // whatever you need to make your gui happy.
  // stored in self->gui_data
  dt_iop_blind_denoise_self_and_index_t s[NB_SCALES];
  GtkWidget *scale, *factor[NB_SCALES]; // this is needed by gui_update
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

// compute noise variance and signal variance
static void var(const float *const in, float* var_noise, float* var_signal, const unsigned width, const unsigned height, float* out, float wb[3])
{
  // radius for first average
  const int radius1 = 2;
  // radius for second average
  float* in_with_2_pixels_average = dt_alloc_align(64, 4 * sizeof(float) * width * height);
  float* averagetmp = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* average1 = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* diff1 = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* diff2 = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* correlation = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* rgbvar = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* rgbvar2 = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* rgbcov = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* vartmp = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* vartmp2 = dt_alloc_align(64, 3 * sizeof(float) * width * height);
  float* covtmp = dt_alloc_align(64, 3 * sizeof(float) * width * height);

  // (1) compute local average
  // (2) decide which channel has the lowest amount of noise
  // (3) compute the averages of the pixels bellow the average (BLA) and of the
  // pixels higher than average (HTA)
  // (4) find the coefs a and b such as Cbest = a * C1 + b * C2 + cst
  // these coefs are found by forcing BLA and HLA equality of Cbest-avgCbest and
  // of a * (C1-avgC1) + b * (C2-avgC2)
  // (5) compute covariance between all channels
  // (6) compute variance of all channels
  // (7) find the signal variance of Cbest as cov(Cbest, a * C1 + b * C2)
  // which is equal to a * cov(Cbest,C1) + b * cov(Cbest,C2)
  // (8) find a and b coefficients for C1 and C2 to find also their signal variance
  // (9) find noise variance as measured variance - signal variance

  // compute averaged image
#ifdef _OPENM
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(radius1, in_with_2_pixels_average, in, wb) \
  schedule(static) collapse(2)
#endif
  for(int i = 0; i < height; i++)
  {
    for(int j = 0; j < width; j++)
    {
      // find closest and second closest pixels (in terms of values)
      // in pixels that are at a distance of 1 pixel to the considered
      // pixel, and average the value of the found pixel with the one
      // of the considered pixel.
      float min_lower_diff = 1000000.0f;
      const float* min_lower = NULL;
      // second min is the second best pixel.
      // we use it instead of min to avoid some "noise" overfitting
      float min_upper_diff = -1000000.0f;
      const float* min_upper = NULL;
      float current[3];
      float current_mean[3];
      //TODO check if this is useful or useless for timings
      for(int c = 0; c < 3; c++)
      {
        current[c] = in[(i * width + j) * 4 + c];
        current_mean[c] = average1[(i * width + j) * 3 + c];
      }
      for(int ii = MAX(i-2, 0); ii <= MIN(i+2, height-1); ii++)
      {
        for(int jj = MAX(j-2, 0); jj <= MIN(j+2, width-1); jj++)
        {
          if((fabs(i - ii) <= 1) && (fabs(j - jj) <= 1)) continue;
          float diff_all_channels = 0.0f;
          for(int c = 0; c < 3; c++)
          {
            float diff = current[c] - in[(ii * width + jj) * 4 + c];
            diff += current_mean[c] - average1[(ii * width + jj) * 3 + c];
            diff /= wb[c];
            diff_all_channels += diff;
          }
          if(diff_all_channels < 0.0f)
          {
            if(diff_all_channels > min_upper_diff)
            {
              min_upper_diff = diff_all_channels;
              min_upper = &(in[(ii * width + jj) * 4]);
            }
          }
          else
          {
            if(diff_all_channels < min_lower_diff)
            {
              min_lower_diff = diff_all_channels;
              min_lower = &(in[(ii * width + jj) * 4]);
            }
          }
        }
      }
      assert(min_upper != NULL);
      assert(min_lower != NULL);
      for(int c = 0; c < 3; c++)
      {
        int count = 1;
        in_with_2_pixels_average[(i * width + j) * 4 + c] = in[(i * width + j) * 4 + c];
        if(min_upper != NULL)
        {
          in_with_2_pixels_average[(i * width + j) * 4 + c] += min_upper[c];
          count++;
        }
        if(min_lower != NULL)
        {
          in_with_2_pixels_average[(i * width + j) * 4 + c] += min_lower[c];
          count++;
        }
        // problem: this will lead to inaccuracies next, when computing variance from this
        in_with_2_pixels_average[(i * width + j) * 4 + c] /= (float)count;
      }
    }
  }

  // compute local average
#ifdef _OPENM
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(radius1, averagetmp, in) \
  schedule(static) collapse(2)
#endif
  for(int i = 0; i < height; i++)
  {
    for(int j = 0; j < width; j++)
    {
      for(int c = 0; c < 3; c++) averagetmp[(i * width + j) * 3 + c] = 0.0f;
      int start = MAX(j-radius1, 0);
      int end = MIN(j+radius1, width-1);
      float norm = end - start + 1.0f;
      for(int jj = start; jj <= end; jj++)
      {
        for(int c = 0; c < 3; c++)
          averagetmp[(i * width + j) * 3 + c] += in[(i * width + jj) * 4 + c] / norm;
      }
    }
  }
#ifdef _OPENM
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(radius1, averagetmp, average1) \
  schedule(static) collapse(2)
#endif
  for(int i = 0; i < height; i++)
  {
    for(int j = 0; j < width; j++)
    {
      for(int c = 0; c < 3; c++) average1[(i * width + j) * 3 + c] = 0.0f;
      int start = MAX(i-radius1, 0);
      int end = MIN(i+radius1, height-1);
      float norm = end - start + 1.0f;
      for(int ii = start; ii <= end; ii++)
      {
        for(int c = 0; c < 3; c++)
          average1[(i * width + j) * 3 + c] += averagetmp[(ii * width + j) * 3 + c] / norm;
      }
    }
  }
  dt_free_align(averagetmp);

  // compute pixel differences
#ifdef _OPENM
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(average1, in, in_with_2_pixels_average, diff1, diff2) \
  schedule(static)
#endif
  for(int i = 0; i < height * width; i++)
  {
    for(int c = 0; c < 3; c++)
    {
      diff1[i * 3 + c] = in[i * 4 + c] - average1[i * 3 + c];
      diff2[i * 3 + c] = in_with_2_pixels_average[i * 4 + c] - average1[i * 3 + c];
    }
  }
  //dt_free_align(average1);

  // start by computing variance and covariance from diff1
#ifdef _OPENM
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(radius1, vartmp, vartmp2, covtmp, diff1, diff2) \
  schedule(static) collapse(2)
#endif
  for(int i = 0; i < height; i++)
  {
    for(int j = 0; j < width; j++)
    {
      for(int c = 0; c < 3; c++)
      {
        vartmp[(i * width + j) * 3 + c] = 0.0f;
        vartmp2[(i * width + j) * 3 + c] = 0.0f;
        covtmp[(i * width + j) * 3 + c] = 0.0f;
      }
      int start = MAX(j-radius1, 0);
      int end = MIN(j+radius1, width-1);
      for(int jj = start; jj <= end; jj++)
      {
        for(int c = 0; c < 3; c++)
        {
          float curr = diff1[(i * width + jj) * 3 + c];
          float curr2 = diff2[(i * width + jj) * 3 + c];
          float next = diff1[(i * width + jj) * 3 + ((c + 1) % 3)];
          vartmp[(i * width + j) * 3 + c] += curr * curr;
          vartmp2[(i * width + j) * 3 + c] += curr2 * curr2;
          covtmp[(i * width + j) * 3 + c] += curr * next;
        }
      }
    }
  }
  dt_free_align(diff1);
  dt_free_align(diff2);

#ifdef _OPENM
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(radius1, vartmp, vartmp2, covtmp, rgbvar, rgbvar2, rgbcov) \
  schedule(static) collapse(2)
#endif
  for(int i = 0; i < height; i++)
  {
    for(int j = 0; j < width; j++)
    {
      for(int c = 0; c < 3; c++)
      {
        rgbvar[(i * width + j) * 3 + c] = 0.0f;
        rgbvar2[(i * width + j) * 3 + c] = 0.0f;
        rgbcov[(i * width + j) * 3 + c] = 0.0f;
      }
      int start = MAX(i-radius1, 0);
      int end = MIN(i+radius1, height-1);
      int startj = MAX(j-radius1, 0);
      int endj = MIN(j+radius1, width-1);
      float norm = (end - start + 1.0f) * (endj - startj + 1.0f) - 1.0f;
      for(int ii = start; ii <= end; ii++)
      {
        for(int c = 0; c < 3; c++)
        {
          rgbvar[(i * width + j) * 3 + c] += vartmp[(ii * width + j) * 3 + c] / norm;
          rgbvar2[(i * width + j) * 3 + c] += vartmp2[(ii * width + j) * 3 + c] / norm;
          rgbcov[(i * width + j) * 3 + c] += covtmp[(ii * width + j) * 3 + c] / norm;
        }
      }
    }
  }
  dt_free_align(vartmp);
  dt_free_align(covtmp);


#ifdef _OPENM
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(radius1, rgb_var, rgb_var2, average1, in, radius1, wb, var_signal, var_noise) \
  schedule(static) collapse(2)
#endif
  for(int i = 0; i < height; i++)
  {
    for(int j = 0; j < width; j++)
    {
      for(int c = 0; c < 3; c++)
      {
        var_noise[(i * width + j) * 4 + c] = MAX(3.0f * (rgbvar[(i * width + j) * 3 + c] - rgbvar2[(i * width + j) * 3 + c]), 0.0f);
        var_signal[(i * width + j) * 4 + c] = rgbvar[(i * width + j) * 3 + c] - var_noise[(i * width + j) * 4 + c];
      }
      if(((i % 7) == 0) && ((j % 17) == 0))
      {
        //printf("%d, %d, %d\n", bestc, nb_above_average, nb_below_average);
        //printf("%d, %d, %f, %f\n", nb_above_average, nb_below_average, a[bestc], b[bestc]);
        //printf("%.6f; %.10f; %.6f; %.10f; %.6f; %.10f\n", average1[((i * width) + j) * 3 + 0], var_noise[((i * width) + j) * 4 + 0], average1[((i * width) + j) * 3 + 1], var_noise[((i * width) + j) * 4 + 1], average1[((i * width) + j) * 3 + 2], var_noise[((i * width) + j) * 4 + 2]);
      }
    }
  }

#ifdef _OPENM
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(out, var_noise, average1) \
  schedule(static)
#endif
  for(int i = 0; i < height * width; i++)
  {
    for(int c = 0; c < 3; c++)
    {
      out[i * 4 + c] = //in[((i * width) + j) * 4 + c];
                        var_noise[i * 4 + c] / MAX(average1[i * 3 + c], 1e-16);

      //out[i * 4 + c] = average1[i * 3 + c];
    }
  }
  dt_free_align(correlation);
  dt_free_align(rgbvar);
  dt_free_align(rgbcov);
  dt_free_align(average1);
  dt_free_align(in_with_2_pixels_average);
}

#if 0
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
          float value = details[(s * width + j) * 4 + c] / wb[c];
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
        variance[(i * width + j) * 4 + c] = 1.5f * (alpha - beta) * wb[c] * wb[c];
      }
    }
  }
  free(tmp);
}
#endif

static void median_direction(dt_iop_blind_denoise_dir_t* direction, unsigned width, unsigned height)
{
  dt_iop_blind_denoise_dir_t* tmpdir = (dt_iop_blind_denoise_dir_t*)malloc(sizeof(dt_iop_blind_denoise_dir_t) * width * height);
  memcpy(tmpdir, direction, sizeof(dt_iop_blind_denoise_dir_t) * width * height);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(tmpdir, direction, width, height) \
    schedule(static)
#endif
  for(int j = 1; j < height-1; j++)
  {
    for(int i = 1; i < width-1; i++)
    {
      float tmpf;
      int8_t tmp[3];
      tmp[0] = tmpdir[j*width+i].w;
      tmp[1] = tmpdir[(j-1)*width+i].w;
      tmp[2] = tmpdir[(j+1)*width+i].w;
      if(tmp[2] < tmp[1])
      {
        tmpf = tmp[2];
        tmp[2] = tmp[1];
        tmp[1] = tmpf;
      }
      if(tmp[2] < tmp[0])
      {
        tmpf = tmp[2];
        tmp[2] = tmp[0];
        tmp[0] = tmpf;
      }
      if(tmp[1] < tmp[0])
      {
        tmpf = tmp[1];
        tmp[1] = tmp[0];
        tmp[0] = tmpf;
      }
      direction[j*width+i].w = tmp[1];
      tmp[0] = tmpdir[j*width+i].h;
      tmp[1] = tmpdir[j*width+i-1].h;
      tmp[2] = tmpdir[j*width+i+1].h;
      if(tmp[2] < tmp[1])
      {
        tmpf = tmp[2];
        tmp[2] = tmp[1];
        tmp[1] = tmpf;
      }
      if(tmp[2] < tmp[0])
      {
        tmpf = tmp[2];
        tmp[2] = tmp[0];
        tmp[0] = tmpf;
      }
      if(tmp[1] < tmp[0])
      {
        tmpf = tmp[1];
        tmp[1] = tmp[0];
        tmp[0] = tmpf;
      }
      direction[j*width+i].h = tmp[1];
    }
  }
  free(tmpdir);
}

// computes the size of the image after 4 2x2 downsampling that are stick
// together mirrored.
static inline void size_mirrored(const size_t w, const size_t h, size_t* mirrored_w, size_t* mirrored_h)
{
  // to determine the total width of the line, we have to consider
  // the coordinate of the last pixel on the line of the original image.
  // if image has width w, its last pixel is at (w-1)
  // but we consider 2x2 blocks of pixels using the coordinate
  // of the top left pixel, thus last such pixel is at (w-2).
  // Also, we do 2x2 blocks either starting at index 0 or at index 1.
  // if starting at index 1, the relative coordinate become (w-3)
  // From these coordinate (w-2) and (w-3), we get the maximum
  // coordinate we will gate for a downsampled line: (w-2)/2 and (w-3)/2
  // The width obtained starting with 0 is (w-2)/2+1 and starting with 1
  // is (w-3)/2+1.
  // As we combine the images obtained starting from 0 and from 1 using
  // symmetry, total width is (w-2)/2+(w-3)/2+2
  *mirrored_w = (w - 2) / 2 + (w - 3) / 2 + 2;
  *mirrored_h = (h - 2) / 2 + (h - 3) / 2 + 2;
}

static inline float* source_to_dest(const size_t width_out, const size_t height_out, const size_t x, const size_t y, float *const restrict out)
{
  const size_t ch = 4;
  const size_t x_odd = x & 1;
  const size_t y_odd = y & 1;
  size_t x_out = x / 2;
  size_t y_out = y / 2;
  if(x_odd)
  {
    x_out = width_out - 1 - x / 2;
  }
  if(y_odd)
  {
    y_out = height_out - 1 - y / 2;
  }
  return out + ch * (y_out * width_out + x_out);
}

static inline void average_2x2(const float* const in, size_t x, size_t y, const size_t width_in, const size_t height_in, const size_t width_out, const size_t height_out, float* const out)
{
  const size_t ch = 4;
  size_t x_next = x + 1;
  size_t y_next = y + 1;
  x_next = (x_next < width_in) ? x_next : width_in - 1;
  y_next = (y_next < height_in) ? y_next : height_in - 1;

  // Nearest pixels in input array (nodes in grid)
  size_t Y = y * width_in;
  size_t Y_next =  y_next * width_in;
  const float* Q_NW = (float *)in + (Y + x) * ch;
  const float* Q_NE = (float *)in + (Y + x_next) * ch;
  const float* Q_SE = (float *)in + (Y_next + x_next) * ch;
  const float* Q_SW = (float *)in + (Y_next + x) * ch;

  float* pixel_out = source_to_dest(width_out, height_out, x, y, out);

#pragma unroll
  for(size_t c = 0; c < ch; c++)
  {
    pixel_out[c] = 0.25f * (Q_SW[c] + Q_SE[c] + Q_NW[c] + Q_NE[c]);
  }
}

// calculer directement en une passe le downscaled, et I - downscaled_upscaled
// compute downscaled image 4 times, and stick them with symmetry
// input:  | /|
//         |/ |
// output "normal" downscaling : |/|
// output: |/\|
//         |\/|
__DT_CLONE_TARGETS__
static inline void downscale_bilinear_mirrored(const float *const restrict in, const size_t width_in, const size_t height_in, float *const restrict out)
{
  size_t width_out;
  size_t height_out;
  size_mirrored(width_in, height_in, &width_out, &height_out);
#ifdef _OPENMP
#pragma omp parallel for simd collapse(2) default(none) \
  schedule(simd:static) aligned(in, out:64) \
  dt_omp_firstprivate(in, out, width_in, height_in, width_out, height_out)
#endif
  for(size_t i = 0; i < height_in-1; i++)
  {
    for(size_t j = 0; j < width_in-1; j++)
    {
      average_2x2(in, j, i, width_in, height_in, width_out, height_out, out);
    }
  }
}

__DT_CLONE_TARGETS__
static inline void upscale_bilinear_mirrored(float *const restrict mirrored, const size_t width_upscaled, const size_t height_upscaled, float *const restrict upscaled)
{
  const size_t ch = 4;
  size_t width_mirrored;
  size_t height_mirrored;
  size_mirrored(width_upscaled, height_upscaled, &width_mirrored, &height_mirrored);
#ifdef _OPENMP
#pragma omp parallel for simd collapse(2) default(none) \
  schedule(simd:static) aligned(upscaled, mirrored:64) \
  dt_omp_firstprivate(upscaled, mirrored, width_mirrored, height_mirrored, width_upscaled, height_upscaled, ch)
#endif
  for(size_t i = 0; i < height_upscaled; i++)
  {
    for(size_t j = 0; j < width_upscaled; j++)
    {
      size_t i_prev = MAX((int32_t)i - 1, 0);
      size_t j_prev = MAX((int32_t)j - 1, 0);
      // gather the 4 pixels that used (i,j) point
      float* pixel0 = source_to_dest(width_mirrored, height_mirrored, j, i, mirrored);
      float* pixel1 = source_to_dest(width_mirrored, height_mirrored, j_prev, i, mirrored);
      float* pixel2 = source_to_dest(width_mirrored, height_mirrored, j, i_prev, mirrored);
      float* pixel3 = source_to_dest(width_mirrored, height_mirrored, j_prev, i_prev, mirrored);
      for(size_t c = 0; c < 3; c++)
      {
        upscaled[(i * width_upscaled + j) * ch + c] = 0.25f * (pixel0[c] + pixel1[c] + pixel2[c] + pixel3[c]);
      }
    }
  }
}

#if 0
__DT_CLONE_TARGETS__
static inline void downscale_bilinear_2x2(const float *const restrict in, const size_t width_in, const size_t height_in, float *const restrict out)
{
  const size_t ch = 4;
  const size_t width_out = (width_in + 1) / 2;
  const size_t height_out = (height_in + 1) / 2;
  // Fast vectorized bilinear interpolation on ch channels
#ifdef _OPENMP
#pragma omp parallel for simd collapse(2) default(none) \
  schedule(simd:static) aligned(in, out:64) \
  dt_omp_firstprivate(in, out, width_out, height_out, width_in, height_in, ch)
#endif
  for(size_t i = 0; i < height_out; i++)
  {
    for(size_t j = 0; j < width_out; j++)
    {
      // Relative coordinates of the pixel in output space
      const float x_out = ((float)j + 0.5f) /(float)width_out;
      const float y_out = ((float)i + 0.5f) /(float)height_out;

      // Corresponding absolute coordinates of the pixel in input space
      const float x_in = x_out * (float)width_in - 0.5f;
      const float y_in = y_out * (float)height_in - 0.5f;

      // Nearest neighbours coordinates in input space
      size_t x_prev = MAX((size_t)floorf(x_in), 0);
      size_t x_next = x_prev + 1;
      size_t y_prev = MAX((size_t)floorf(y_in), 0);
      size_t y_next = y_prev + 1;

      x_prev = (x_prev < width_in) ? x_prev : width_in - 1;
      x_next = (x_next < width_in) ? x_next : width_in - 1;
      y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
      y_next = (y_next < height_in) ? y_next : height_in - 1;

      // Nearest pixels in input array (nodes in grid)
      const size_t Y_prev = y_prev * width_in;
      const size_t Y_next =  y_next * width_in;
      const float *const Q_NW = (float *)in + (Y_prev + x_prev) * ch;
      const float *const Q_NE = (float *)in + (Y_prev + x_next) * ch;
      const float *const Q_SE = (float *)in + (Y_next + x_next) * ch;
      const float *const Q_SW = (float *)in + (Y_next + x_prev) * ch;

      // Spatial differences between nodes
      const float Dy_next = (float)y_next - y_in;
      const float Dy_prev = 1.f - Dy_next; // because next - prev = 1
      const float Dx_next = (float)x_next - x_in;
      const float Dx_prev = 1.f - Dx_next; // because next - prev = 1

      // Interpolate over ch layers
      float *const pixel_out = (float *)out + (i * width_out + j) * ch;

#pragma unroll
      for(size_t c = 0; c < ch; c++)
      {
        pixel_out[c] = Dy_prev * (Q_SW[c] * Dx_next + Q_SE[c] * Dx_prev) +
                       Dy_next * (Q_NW[c] * Dx_next + Q_NE[c] * Dx_prev);
      }
    }
  }
}
#endif

#define SWAP(x,y) if (diff[y] < diff[x]) { float tmp = diff[x]; diff[x] = diff[y]; diff[y] = tmp; dt_iop_blind_denoise_dir_t tmpdir = dir[x]; dir[x] = dir[y]; dir[y] = tmpdir; }

// for each pixel, direction[0] is the best direction, direction[1] the second best, etc
static void get_details_and_direction(const float* in, float* mean, float* details, unsigned width, unsigned height, unsigned widthmean, unsigned heightmean, dt_iop_blind_denoise_dir_t* direction, float* wb, float* coefs)
{
  float* upscaled_mean = malloc(sizeof(float) * 4 * width * height);
  interpolate_bilinear(mean, widthmean, heightmean, upscaled_mean, width, height, 4);

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(in, upscaled_mean, details, direction, width, height, widthmean, heightmean, wb, coefs) \
    schedule(static)
#endif
  for(int j = 0; j < height; j++)
  {
    for(int i = 0; i < width; i++)
    {
      float diff[9] = {0.0f};
      dt_iop_blind_denoise_dir_t dir[9];
      //dt_iop_blind_denoise_dir_t* dir = &(direction[(j * width + i) * 4]);
      for(int jj = -1; jj <= 1; jj++)
      {
        for(int ii = -1; ii <= 1; ii++)
        {
          int kj = jj;
          int ki = ii;
          unsigned index = (kj + 1) * 3 + ki + 1;
          if(j + kj < 0)
          {
            kj = 0;
          }
          if(i + ki < 0)
          {
            ki = 0;
          }
          if(j + kj > height - 1)
          {
            kj = 0;
          }
          if(i + ki > width - 1)
          {
            ki = 0;
          }
          dir[index].w = ki;
          dir[index].h = kj;
          int indexj = j + kj;
          int indexi = i + ki;
          for(unsigned c = 0; c < 3; c++)
          {
            diff[index] += fabs(upscaled_mean[(indexj * width + indexi) * 4 + c] - in[(j * width + i) * 4 + c]) / wb[c];
          }
        }
      }
      // for(unsigned c = 0; c < 3; c++)
      // {
        // diff[0] += fabs(mean[(j0 * widthmean + i0) * 4 + c] - in[(j * width + i) * 4 + c]) / wb[c];
        // diff[1] += fabs(mean[(j1 * widthmean + i0) * 4 + c] - in[(j * width + i) * 4 + c]) / wb[c];
        // diff[2] += fabs(mean[(j0 * widthmean + i1) * 4 + c] - in[(j * width + i) * 4 + c]) / wb[c];
        // diff[3] += fabs(mean[(j1 * widthmean + i1) * 4 + c] - in[(j * width + i) * 4 + c]) / wb[c];
      //   diff[0] += MAX(mean[(j0 * widthmean + i0) * 4 + c], 0.0f) / wb[c];
      //   diff[1] += MAX(mean[(j1 * widthmean + i0) * 4 + c], 0.0f) / wb[c];
      //   diff[2] += MAX(mean[(j0 * widthmean + i1) * 4 + c], 0.0f) / wb[c];
      //   diff[3] += MAX(mean[(j1 * widthmean + i1) * 4 + c], 0.0f) / wb[c];
      //   avg += in[(j * width + i) * 4 + c] / wb[c];
      // }
      // for(int k = 0; k < 4; k++)
      // {
      //   diff[k] /= MAX(avg, 0.00001f);
      //   if(diff[k] < 1.0f)
      //   {
      //     diff[k] = MAX(diff[k], 0.00001f);
      //     diff[k] = 1.0f / diff[k];
      //   }
      // }

      // sort diff and dir jointly
      SWAP(0, 1);
      SWAP(3, 4);
      SWAP(6, 7);
      SWAP(1, 2);
      SWAP(4, 5);
      SWAP(7, 8);
      SWAP(0, 1);
      SWAP(3, 4);
      SWAP(6, 7);
      SWAP(0, 3);
      SWAP(3, 6);
      SWAP(0, 3);
      SWAP(1, 4);
      SWAP(4, 7);
      SWAP(1, 4);
      SWAP(2, 5);
      SWAP(5, 8);
      SWAP(2, 5);
      SWAP(1, 3);
      SWAP(5, 7);
      SWAP(2, 6);
      SWAP(4, 6);
      SWAP(2, 4);
      SWAP(2, 3);
      SWAP(5, 6);

      const int last = 1;
      if(i <= last) dir[0].w = 1;
      if(j <= last) dir[0].h = 1;
      if(i >= width-1-last) dir[0].w = -1;
      if(j >= height-1-last) dir[0].h = -1;
      // uncomment to compare approach with an approach which has a "normal" upsampling
      // dir[0].w = 0;
      // dir[0].h = 0;
      direction[j * width + i] = dir[0];
      for(unsigned c = 0; c < 3; c++)
      {
        float value = in[(j * width + i) * 4 + c];
        value -= upscaled_mean[((j + dir[0].h) * width + i + dir[0].w) * 4 + c];
        details[(j * width + i) * 4 + c] = value;
      }
    }
  }
  free(upscaled_mean);
}

#undef SWAP

#if 0
#define SWAP(x,y) if (tmpa[y] < tmpa[x]) { float tmp = tmpa[x]; tmpa[x] = tmpa[y]; tmpa[y] = tmp; unsigned tmpindex = index[x]; index[x] = index[y]; index[y] = tmpindex;}

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

      float tmpa[4] = {0.0f};
      unsigned index[4];
      index[0] = (j*width+i)*4;
      index[1] = (j*width+MIN(i+1,width-1))*4;
      index[2] = (MIN(j+1,height-1)*width+i)*4;
      index[3] = (MIN(j+1,height-1)*width+MIN(i+1,width-1))*4;
      for(unsigned c = 0; c < 3; c++)
      {
        tmpa[0] += in[(j*width+i)*4+c];
        tmpa[1] += in[(j*width+MIN(i+1,width-1))*4+c];
        tmpa[2] += in[(MIN(j+1,height-1)*width+i)*4+c];
        tmpa[3] += in[(MIN(j+1,height-1)*width+MIN(i+1,width-1))*4+c];
      }
      SWAP(0, 1);
      SWAP(2, 3);
      SWAP(0, 2);
      SWAP(1, 3);
      SWAP(1, 2);
      float mean = (tmpa[2] + tmpa[1]) / 2.0f;
      float dist3 = fabs(mean - tmpa[3]);
      float dist0 = fabs(mean - tmpa[0]);
      for(unsigned c = 0; c < 3; c++)
      {
        tmpa[0] = in[index[0] + c];
        tmpa[1] = in[index[1] + c];
        tmpa[2] = in[index[2] + c];
        tmpa[3] = in[index[3] + c];
        if(dist3 < dist0) mean = (tmpa[3] + tmpa[2] + tmpa[1] + 0.1f * tmpa[0]) / 3.1f;
        else mean = (tmpa[0] + tmpa[2] + tmpa[1] + 0.1f * tmpa[3]) / 3.1f;
        //FIXME would be nice to have a better (sparser?) downscaling
        //mean = (tmpa[0] + tmpa[2] + tmpa[1] + tmpa[3]) / 4.0f;
        out[(jout * widthout + iout) * 4 + c] = mean;
      }
    }
  }
}
#endif

static int sign(float a)
{
  return (a >= 0.0f) - (a < 0.0f);
}

static gboolean invert_matrix(const float in[9], float out[9])
{
  // use same notation as https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
  const float biga = in[4] * in[8] - in[5] * in[7];
  const float bigb = -in[3] * in[8] + in[5] * in[6];
  const float bigc = in[3] * in[7] - in[4] * in[6];
  const float bigd = -in[1] * in[8] + in[2] * in[7];
  const float bige = in[0] * in[8] - in[2] * in[6];
  const float bigf = -in[0] * in[7] + in[1] * in[6];
  const float bigg = in[1] * in[5] - in[2] * in[4];
  const float bigh = -in[0] * in[5] + in[2] * in[3];
  const float bigi = in[0] * in[4] - in[1] * in[3];

  const float det = in[0] * biga + in[1] * bigb + in[2] * bigc;
  if(det == 0.0f)
  {
    return FALSE;
  }

  out[0] = 1.0f / det * biga;
  out[1] = 1.0f / det * bigd;
  out[2] = 1.0f / det * bigg;
  out[3] = 1.0f / det * bigb;
  out[4] = 1.0f / det * bige;
  out[5] = 1.0f / det * bigh;
  out[6] = 1.0f / det * bigc;
  out[7] = 1.0f / det * bigf;
  out[8] = 1.0f / det * bigi;
  return TRUE;
}

// create the white balance adaptative conversion matrices
// supposes toY0U0V0 already contains the "normal" conversion matrix
static void set_up_conversion_matrices(float toY0U0V0[9], float toRGB[9], float wb[3], float* ref)
{
  // for an explanation of the spirit of the choice of the coefficients of the
  // Y0U0V0 conversion matrix, see part 12.3.3 page 190 of
  // "From Theory to Practice, a Tour of Image Denoising"
  // https://hal.archives-ouvertes.fr/tel-01114299
  // we adapt a bit the coefficients, in a way that follows the same spirit.

  float mean[3];
  for(int c = 0; c < 3; c++)
    mean[c] = MAX(ref[c], 0.0f);

  float sum_invwb = 1.0f/wb[0] + 1.0f/wb[1] + 1.0f/wb[2];
  // we change the coefs to Y0, but keeping the goal of making SNR higher:
  // these were all equal to 1/3 to get the Y0 the least noisy possible, assuming
  // that all channels have equal noise variance.
  // as white balance influences noise variance, we do a weighted mean depending
  // on white balance. Note that it is equivalent to keeping the 1/3 coefficients
  // if we divide by the white balance coefficients beforehand.
  // we then normalize the line so that variance becomes equal to 1:
  // var(Y0) = 1/9 * (var(R) + var(G) + var(B)) = 1/3
  // var(sqrt(3)Y0) = 1
  sum_invwb *= sqrt(3);
  toY0U0V0[0] = sum_invwb / wb[0];
  toY0U0V0[1] = sum_invwb / wb[1];
  toY0U0V0[2] = sum_invwb / wb[2];
  // we also normalize the other line in a way that should give a variance of 1
  // if var(B/wb[B]) == 1, then var(B) = wb[B]^2
  // note that we don't change the coefs of U0 and V0 depending on white balance,
  // apart of the normalization: these coefficients do differences of RGB channels
  // to try to reduce or cancel the signal. If we change these depending on white
  // balance, we will not reduce/cancel the signal anymore.
  // const float stddevU0 = sqrt(0.5f * 0.5f * wb[0] * wb[0] + 0.5f * 0.5f * wb[2] * wb[2]);
  // const float stddevV0 = sqrt(0.25f * 0.25f * wb[0] * wb[0] + 0.5f * 0.5f * wb[1] * wb[1] + 0.25f * 0.25f * wb[2] * wb[2]);
  // toY0U0V0[3] /= stddevU0;
  // toY0U0V0[4] /= stddevU0;
  // toY0U0V0[5] /= stddevU0;
  // toY0U0V0[6] /= stddevV0;
  // toY0U0V0[7] /= stddevV0;
  // toY0U0V0[8] /= stddevV0;
  float RB_ratio = mean[0] / (mean[2] + 0.001f);
  toY0U0V0[3] = 1.0f / (1.0f + RB_ratio);
  toY0U0V0[4] = 0.0f;
  toY0U0V0[5] = -RB_ratio / (1.0f + RB_ratio);
  float GR_ratio = mean[1] / (mean[0] + 0.001f);
  float GB_ratio = mean[1] / (mean[2] + 0.001f);
  toY0U0V0[6] = -0.5f * GR_ratio / (1.0f + 0.5f * GR_ratio + 0.5f * GB_ratio);
  toY0U0V0[7] = 1.0f / (1.0f + 0.5f * GR_ratio + 0.5f * GB_ratio);
  toY0U0V0[8] = -0.5f * GB_ratio / (1.0f + 0.5f * GR_ratio + 0.5f * GB_ratio);
  // uncomment for 'classic' Y0U0V0 transform
  // toY0U0V0[3] = 1.0f;
  // toY0U0V0[4] = 0.0f;
  // toY0U0V0[5] = -1.0f;
  // toY0U0V0[6] = -0.5f;
  // toY0U0V0[7] = 1.0f;
  // toY0U0V0[8] = -0.5f;

  // for(int c = 0; c < 9; c++) toY0U0V0[c] = 0.0f;
  //   toY0U0V0[0] = 1.0f;
  //   toY0U0V0[4] = 1.0f;
  //   toY0U0V0[8] = 1.0f;

  const gboolean is_invertible = invert_matrix(toY0U0V0, toRGB);
  if(!is_invertible)
  {
    // use standard form if whitebalance adapted matrix is not invertible
    float stddevY0 = sqrt(1.0f / 9.0f * (wb[0] * wb[0] + wb[1] * wb[1] + wb[2] * wb[2]));
    toY0U0V0[0] = 1.0f / (3.0f * stddevY0);
    toY0U0V0[1] = 1.0f / (3.0f * stddevY0);
    toY0U0V0[2] = 1.0f / (3.0f * stddevY0);
    invert_matrix(toY0U0V0, toRGB);
  }
}

static inline void matrix_mul(float* matrix, float* in, float* out)
{
  for(int c = 0; c < 3; c++)
  {
    out[c] = 0.0f;
    for(int k = 0; k < 3; k++)
    {
      out[c] += matrix[3 * c + k] * in[k];
    }
  }
}

#undef SWAP
#define SWAP(x,y) {if(tmp[x] > tmp[y]) {float t = tmp[x]; tmp[x] = tmp[y]; tmp[y] = t;}}
static void minmax_thresholding(float* details, unsigned width, unsigned height)
{
  float* in = malloc(sizeof(float) * 4 * width * height);
  memcpy(in, details, sizeof(float) * 4 * width * height);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(details, in, width, height) \
    schedule(static)
#endif
  for(unsigned j = 1; j < height-1; j++)
  {
    for(unsigned i = 1; i < width-1; i++)
    {
      for(int c = 0; c < 3; c++)
      {
        float tmp[9];
        for(int jj = -1; jj <= 1; jj++)
        {
          for(int ii = -1; ii <= 1; ii++)
          {
            tmp[(jj+1)*3+ii+1] = in[((j+jj)*width+(i+ii))*4+c];
          }
        }
        // sorting
        SWAP(0, 1);
        SWAP(3, 4);
        SWAP(6, 7);
        SWAP(1, 2);
        SWAP(4, 5);
        SWAP(7, 8);
        SWAP(0, 1);
        SWAP(3, 4);
        SWAP(6, 7);
        SWAP(0, 3);
        SWAP(3, 6);
        SWAP(0, 3);
        SWAP(1, 4);
        SWAP(4, 7);
        SWAP(1, 4);
        SWAP(2, 5);
        SWAP(5, 8);
        SWAP(2, 5);
        SWAP(1, 3);
        SWAP(5, 7);
        SWAP(2, 6);
        SWAP(4, 6);
        SWAP(2, 4);
        SWAP(2, 3);
        SWAP(5, 6);
        for(int k = 0; k < 8; k++)
        {
          if(tmp[k] > tmp[k+1])
            printf("problem\n");
        }
        float curr = in[(j*width+i)*4+c];
        if(curr == tmp[0])
        {
          details[(j*width+i)*4+c] = tmp[1];
        }
        if(curr == tmp[8])
        {
          details[(j*width+i)*4+c] = tmp[7];
        }
      }
    }
  }
  free(in);
}
#undef SWAP

static void thresholding_and_recompose(float* mean, unsigned widthmean, unsigned heightmean, float* details, float* out, unsigned width, unsigned height, dt_iop_blind_denoise_dir_t* direction, float threshold, float* wb, float* var, const float weight[3])
{
  float* upscaled_mean = malloc(sizeof(float) * 4 * width * height);
  interpolate_bilinear(mean, widthmean, heightmean, upscaled_mean, width, height, 4);

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(upscaled_mean, out, details, direction, width, height, threshold, wb, var, weight) \
    schedule(static)
#endif
  for(unsigned j = 0; j < height; j++)
  {
    for(unsigned i = 0; i < width; i++)
    {
      float toY0U0V0[9] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f,
                           0.5f,      0.0f,      -0.5f,
                           0.25f,     -0.5f,     0.25f};
      float toRGB[9] = {0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f};
      //TODO take var into account to setup the conversion matrices
      float* mean_ptr = &(upscaled_mean[((j + direction[j * width + i].h) * width + i + direction[j * width + i].w) * 4]);
      set_up_conversion_matrices(toY0U0V0, toRGB, wb, mean_ptr);
      float tmp[3];

      matrix_mul(toY0U0V0, &(details[(j * width + i) * 4]), tmp);
      // mean = MAX(mean, 0.0f) + 0.05f;
      // //const float mean_mean = sqrt(MAX(0.01f, direction[(j * width + i) * 4][0] + direction[(j * width + i) * 4][1] + direction[(j * width + i) * 4][2]));
      for(unsigned c = 0; c < 3; c++)
      {
        float thrs = threshold * weight[c];// * mean_mean;
        tmp[c] = /*fabs(mean) * */sign(tmp[c]) * MAX(/*MIN(*/fabs(tmp[c] /*/ mean*/)/*, powf(1.0f / thrs, 4.0f))*/ - thrs, 0.0f);
      }
      matrix_mul(toRGB, tmp, &(details[(j * width + i) * 4]));
      for(unsigned c = 0; c < 3; c++)
      {
        out[(j * width + i) * 4 + c] = MAX(MAX(mean_ptr[c], 0.0f) + details[(j * width + i) * 4 + c], 0.0f);
      }
    }
  }
  free(upscaled_mean);
}

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
  dt_iop_blind_denoise_dir_t* direction[NB_SCALES];
  //float threshold[NB_SCALES] = {0.7, 1.0, 0.9, 0.6, 0.4, 0.2, 0.1, 0.05};
  float threshold[NB_SCALES] = {5.0, 4.5, 3.0, 1.7, 1.0, 0.6, 0.3, 0.1};
//  float threshold[NB_SCALES] = {0.20, 0.20, 0.20, 0.15, 0.10, 0.05, 0.01, 0.007};
//  float threshold[NB_SCALES] = {4.0f, 1.8f, 0.70, 0.3, 0.1, 0.05, 0.01, 0.007};
  unsigned width[NB_SCALES];
  unsigned height[NB_SCALES];
  float coefs[NB_SCALES][4];
  float wb[3];
  for(int i = 0; i < 3; i++) wb[i] = piece->pipe->dsc.temperature.coeffs[i];

  // init
  means[0] = in;
  width[0] = roi_out->width;
  height[0] = roi_out->height;
  details[0] = (float*)malloc(sizeof(float) * 4 * width[0] * height[0]);
  vars[0] = (float*)calloc(sizeof(float), 4 * width[0] * height[0]);
  direction[0] = (dt_iop_blind_denoise_dir_t*)malloc(sizeof(dt_iop_blind_denoise_dir_t) * 4 * width[0] * height[0]);
  for(int i = 1; i < NB_SCALES; i++)
  {
    width[i] = (width[i-1] + 1) / 2;
    height[i] = (height[i-1] + 1) / 2;
    means[i] = (float*)malloc(sizeof(float) * 4 * width[i] * height[i]);
    vars[i] = (float*)calloc(sizeof(float), 4 * width[i] * height[i]);
    details[i] = (float*)malloc(sizeof(float) * 4 * width[i] * height[i]);
    direction[i] = (dt_iop_blind_denoise_dir_t*)malloc(sizeof(dt_iop_blind_denoise_dir_t) * 4 * width[i] * height[i]);
  }

  size_t total_width_out;
  size_t total_height_out;
  size_mirrored(width[0], height[0], &total_width_out, &total_height_out);
  float* dwn_mirrored = dt_alloc_align(64, sizeof(float) * 4 * total_width_out * total_height_out);
  memset(out, 0, width[0] * height[0] * 4 * sizeof(float));
  downscale_bilinear_mirrored(in, width[0], height[0], dwn_mirrored);
  for(size_t i = 0; i < MIN(total_height_out, height[0]); i++)
  {
    for(size_t j = 0; j < MIN(total_width_out, width[0]); j++)
    {
      for(size_t c = 0; c < 4; c++)
      {
        out[((i * width[0]) + j) * 4 + c] = dwn_mirrored[((i * total_width_out) + j) * 4 + c];
      }
    }
  }
  upscale_bilinear_mirrored(dwn_mirrored, width[0], height[0], out);
  return;
  float* var_noise = (float*)malloc(sizeof(float) * 4 * width[0] * height[0]);
  float* var_signal = (float*)malloc(sizeof(float) * 4 * width[0] * height[0]);
  var(in, var_noise, var_signal, width[0], height[0], ovoid, wb);


  for(int k = 0; k < NB_SCALES; k++)
  {
    threshold[k] = threshold[k] * 0.01f * d->factor[k];
    float sum_coefs = 0.0f;
    for(int c = 0; c < 4; c++)
    {
      coefs[k][c] = expf(-c / (k * k / 40.0f + 0.5f));
      sum_coefs += coefs[k][c];
    }
    for(int c = 0; c < 4; c++)
    {
      coefs[k][c] /= sum_coefs;
    }
  }

  for(int i = 0; i < NB_SCALES-1; i++)
  {
    // decompose(means[i], means[i+1], width[i], height[i]);
    interpolate_bilinear(means[i], width[i], height[i], means[i+1], width[i+1], height[i+1], 4);
  }
  for(int i = NB_SCALES-1; i > 0; i--)
  {
    get_details_and_direction(means[i-1], means[i], details[i-1], width[i-1], height[i-1], width[i], height[i], direction[i-1], wb, coefs[i-1]);
    if(i < 4)
    {
      median_direction(direction[i-1], width[i-1], height[i-1]);
    }
  }
  for(int i = NB_SCALES-1; i > 1; i--)
  {
    // const float weight[3] = {1.0f, powf(1.0f - i / (float)NB_SCALES, 4.0f), powf(1.0f - i / (float)NB_SCALES, 4.0f)};
    //const float weight[3] = {1.0f, d->checker_scale / 2.0f, d->checker_scale / 2.0f};
    const float weight[3] = {1.0f, 1.0f, 1.0f};
    //get_details_and_direction(means[i-1], means[i], details[i-1], width[i-1], height[i-1], width[i], height[i], direction[i-1], wb, coefs[i-1]);
    thresholding_and_recompose(means[i], width[i], height[i], details[i-1], means[i-1], width[i-1], height[i-1], direction[i-1], threshold[i-1], wb, vars[i-1], weight);
    //if(i < 5) minmax_thresholding(means[i-1], width[i-1], height[i-1]);
  }
  //const float weight[3] = {1.0f, d->checker_scale / 2.0f * 1.0f - 0 / (float)NB_SCALES, d->checker_scale / 2.0f * 1.0f - 0 / (float)NB_SCALES};
  const float weight[3] = {1.0f, 1.0f, 1.0f};
  //get_details_and_direction(means[0], means[1], details[0], width[0], height[0], width[1], height[1], direction[0], wb, coefs[0]);
  thresholding_and_recompose(means[1], width[1], height[1], details[0], out, width[0], height[0], direction[0], threshold[0], wb, vars[0], weight);
  minmax_thresholding(out, width[0], height[0]);

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
  dt_iop_blind_denoise_params_t tmp;
  tmp.checker_scale = 10;
  for(int i = 0; i < NB_SCALES; i++)
    tmp.factor[i] = 0.1;

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

static void factor_callback(GtkWidget *w, dt_iop_blind_denoise_self_and_index_t *s)
{
  if(darktable.gui->reset) return;
  dt_iop_blind_denoise_params_t *p = (dt_iop_blind_denoise_params_t *)s->self->params;
  p->factor[s->n] = dt_bauhaus_slider_get(w);
  dt_dev_add_history_item(darktable.develop, s->self, TRUE);
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_blind_denoise_gui_data_t *g = (dt_iop_blind_denoise_gui_data_t *)self->gui_data;
  dt_iop_blind_denoise_params_t *p = (dt_iop_blind_denoise_params_t *)self->params;
  dt_bauhaus_slider_set(g->scale, p->checker_scale);
  for(int i = 0; i < NB_SCALES; i++)
    dt_bauhaus_slider_set(g->factor[i], p->factor[i]);
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

  for(int i = 0; i < NB_SCALES; i++)
  {
    g->s[i].self = self;
    g->s[i].n = i;
    g->factor[i] = dt_bauhaus_slider_new_with_range(self, 0.0, 100.0, 0.1, 0.5, 2);
    dt_bauhaus_widget_set_label(g->factor[i], NULL, _("factor"));
    gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->factor[i]), TRUE, TRUE, 0);
    g_signal_connect(G_OBJECT(g->factor[i]), "value-changed", G_CALLBACK(factor_callback), &(g->s[i]));
  }
}

void gui_cleanup(dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
