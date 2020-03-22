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

#define SWAP(x,y) if (diff[y] < diff[x]) { float tmp = diff[x]; diff[x] = diff[y]; diff[y] = tmp; float* tmpdir = dir[x]; dir[x] = dir[y]; dir[y] = tmpdir; }

// for each pixel, direction[0] is the best direction, direction[1] the second best, etc
static void get_details_and_direction(const float* in, float* mean, float* details, unsigned width, unsigned height, float** direction, float* wb, float* coefs)
{
  const unsigned widthmean = (width + 1) / 2;
  const unsigned heightmean = (height + 1) / 2;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(in, mean, details, direction, width, height, widthmean, heightmean, wb, coefs) \
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
      float avg = 0.0f;
      for(unsigned c = 0; c < 3; c++)
      {
        diff[0] += /*fabs(*/mean[(j0 * widthmean + i0) * 4 + c]/* - in[(j * width + i) * 4 + c])*/ / wb[c];
        diff[1] += /*fabs(*/mean[(j1 * widthmean + i0) * 4 + c]/* - in[(j * width + i) * 4 + c])*/ / wb[c];
        diff[2] += /*fabs(*/mean[(j0 * widthmean + i1) * 4 + c]/* - in[(j * width + i) * 4 + c])*/ / wb[c];
        diff[3] += /*fabs(*/mean[(j1 * widthmean + i1) * 4 + c]/* - in[(j * width + i) * 4 + c])*/ / wb[c];
        avg += in[(j * width + i) * 4 + c] / wb[c];
      }
      for(int k = 0; k < 4; k++)
      {
        diff[k] /= MAX(avg, 0.00001f);
        if(diff[k] < 1.0f)
        {
          diff[k] = MAX(diff[k], 0.00001f);
          diff[k] = 1.0f / diff[k];
        }
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
        float value = in[(j * width + i) * 4 + c];
        value -= coefs[0] * dir[0][c];
        value -= coefs[1] * dir[1][c];
        value -= coefs[2] * dir[2][c];
        value -= coefs[3] * dir[3][c];
        details[(j * width + i) * 4 + c] = value;
      }
    }
  }
}

#undef SWAP
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
        mean = (tmpa[0] + tmpa[2] + tmpa[1] + tmpa[3]) / 4.0f;
        out[(jout * widthout + iout) * 4 + c] = mean;
      }
    }
  }
}

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

static void thresholding(float* details, unsigned width, unsigned height, float** direction, float threshold, float* wb, float* var, const float weight[3])
{
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(details, direction, width, height, threshold, wb, var, weight) \
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
      set_up_conversion_matrices(toY0U0V0, toRGB, wb, direction[(j * width + i) * 4]);
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
    }
  }
}

// recompose image from 2 layers
// width and height are the dimensions of out
static void recompose(float* in, float* out, float* details, unsigned width, unsigned height, float** direction, float* coefs)
{
  // const unsigned widthin = (width + 1) / 2;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    dt_omp_firstprivate(in, out, details, direction, width, height, coefs) \
    schedule(static)
#endif
  for(unsigned j = 0; j < height; j++)
  {
    for(unsigned i = 0; i < width; i++)
    {
      for(unsigned c = 0; c < 3; c++)
      {
        float value = coefs[0] * direction[(j * width + i) * 4][c];
        value += coefs[1] * direction[(j * width + i) * 4 + 1][c];
        value += coefs[2] * direction[(j * width + i) * 4 + 2][c];
        value += coefs[3] * direction[(j * width + i) * 4 + 3][c];
        value += details[(j * width + i) * 4 + c];
        out[(j * width + i) * 4 + c] = value;
      }
    }
  }
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
  float** direction[NB_SCALES];
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

  // threshold[0] *= powf(0.03f * d->checker_scale, 4.0f * 2.4f);
  // threshold[1] *= powf(0.03f * d->checker_scale, 4.0f * 2.2f);
  // threshold[2] *= powf(0.03f * d->checker_scale, 4.0f * 2.0f);
  // threshold[3] *= powf(0.03f * d->checker_scale, 4.0f * 1.8f);
  // threshold[4] *= powf(0.03f * d->checker_scale, 4.0f * 1.6f);
  // threshold[5] *= powf(0.03f * d->checker_scale, 4.0f * 1.4f);
  // threshold[6] *= powf(0.03f * d->checker_scale, 4.0f * 1.2f);
  // threshold[7] *= powf(0.03f * d->checker_scale, 4.0f * 1.0f);
  for(int k = 0; k < NB_SCALES; k++)
  {
    threshold[k] = threshold[k] * 0.1f * d->factor[k];
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
    decompose(means[i], means[i+1], width[i], height[i]);
  }
  for(int i = NB_SCALES-1; i > 0; i--)
  {
    get_details_and_direction(means[i-1], means[i], details[i-1], width[i-1], height[i-1], direction[i-1], wb, coefs[i-1]);
  }
  for(int i = NB_SCALES-1; i > 1; i--)
  {
    // const float weight[3] = {1.0f, powf(1.0f - i / (float)NB_SCALES, 4.0f), powf(1.0f - i / (float)NB_SCALES, 4.0f)};
    const float weight[3] = {1.0f, d->checker_scale / 2.0f, d->checker_scale / 2.0f};
    //get_details_and_direction(means[i-1], means[i], details[i-1], width[i-1], height[i-1], direction[i-1], wb, coefs[i-1]);
    //var(details[i-1], vars[i-1], width[i-1], height[i-1], NB_SCALES - i, wb, d->checker_scale);
    thresholding(details[i-1], width[i-1], height[i-1], direction[i-1], threshold[i-1], wb, vars[i-1], weight /*powf(2.0f, NB_SCALES - i) / 10.0f*/);
    //if(i <= 2) memset(details[i-1], 0, sizeof(float) * width[i-1] * height[i-1] * 4);
    recompose(means[i], means[i-1], details[i-1], width[i-1], height[i-1], direction[i-1], coefs[i-1]);
  }
  const float weight[3] = {1.0f, d->checker_scale / 2.0f * 1.0f - 0 / (float)NB_SCALES, d->checker_scale / 2.0f * 1.0f - 0 / (float)NB_SCALES};
  //get_details_and_direction(means[0], means[1], details[0], width[0], height[0], direction[0], wb, coefs[0]);
  //memset(details[0], 0, sizeof(float) * width[0] * height[0] * 4);
  //var(details[0], vars[0], width[0], height[0], NB_SCALES, wb, d->checker_scale);
  thresholding(details[0], width[0], height[0], direction[0], threshold[0], wb, vars[0], weight /*powf(2.0f, NB_SCALES) / 10.0f*/);
  recompose(means[1], out, details[0], width[0], height[0], direction[0], coefs[0]);

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
