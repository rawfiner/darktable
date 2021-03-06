/*
    This file is part of darktable,
    Copyright (C) 2010-2020 darktable developers.

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
#include "common/gaussian.h"
#include "common/fast_guided_filter.h"
#include "common/nlmeans_core.h"

#include <gtk/gtk.h>
#include <stdlib.h>

DT_MODULE_INTROSPECTION(1, dt_iop_cacorrectrgb_params_t)

typedef enum dt_iop_cacorrectrgb_guide_channel_t
{
  DT_CACORRECT_RGB_R = 0,    // $DESCRIPTION: "red"
  DT_CACORRECT_RGB_G = 1,    // $DESCRIPTION: "green"
  DT_CACORRECT_RGB_B = 2     // $DESCRIPTION: "blue"
} dt_iop_cacorrectrgb_guide_channel_t;

typedef enum dt_iop_cacorrectrgb_mode_t
{
  DT_CACORRECT_MODE_STANDARD = 0,  // $DESCRIPTION: "standard"
  DT_CACORRECT_MODE_DARKEN = 1,    // $DESCRIPTION: "darken only"
  DT_CACORRECT_MODE_BRIGHTEN = 2   // $DESCRIPTION: "brighten only"
} dt_iop_cacorrectrgb_mode_t;


typedef struct dt_iop_cacorrectrgb_params_t
{
  dt_iop_cacorrectrgb_guide_channel_t guide_channel; // $DEFAULT: DT_CACORRECT_RGB_G $DESCRIPTION: "guide"
  float radius; // $MIN: 1 $MAX: 1000 $DEFAULT: 5 $DESCRIPTION: "radius"
  float force; // $MIN: 1 $MAX: 1000 $DEFAULT: 5 $DESCRIPTION: "force"
  dt_iop_cacorrectrgb_mode_t mode; // $DEFAULT: DT_CACORRECT_MODE_STANDARD $DESCRIPTION: "correction mode"
} dt_iop_cacorrectrgb_params_t;

typedef struct dt_iop_cacorrectrgb_gui_data_t
{
  GtkWidget *guide_channel, *radius, *force, *mode;
} dt_iop_cacorrectrgb_gui_data_t;

// this returns a translatable name
const char *name()
{
  // make sure you put all your translatable strings into _() !
  return _("chromatic aberrations rgb");
}

// some additional flags (self explanatory i think):
int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
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

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, p1, self->params_size);
}

static void normalize_manifolds(float *const restrict blurred_in, float *const restrict blurred_manifold_lower, float *const restrict blurred_manifold_higher, const size_t width, const size_t height, const dt_iop_cacorrectrgb_guide_channel_t guide)
{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(blurred_in, blurred_manifold_lower, blurred_manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(blurred_in, blurred_manifold_lower, blurred_manifold_higher:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    // normalize
    const float weighth = fmaxf(blurred_manifold_higher[k * 4 + 3], 1E-6);
    const float weightl = fmaxf(blurred_manifold_lower[k * 4 + 3], 1E-6);
    for(size_t c = 0; c < 3; c++)
    {
      blurred_manifold_higher[k * 4 + c] /= weighth;
      blurred_manifold_lower[k * 4 + c] /= weightl;
    }
    // replace by average if weight is too small
    if(weighth < 0.05f)
    {
      for(size_t c = 0; c < 3; c++)
      {
        blurred_manifold_higher[k * 4 + c] = blurred_in[k * 4 + c];
      }
    }
    if(weightl < 0.05f)
    {
      for(size_t c = 0; c < 3; c++)
      {
        blurred_manifold_lower[k * 4 + c] = blurred_in[k * 4 + c];
      }
    }
  }
}

static void get_manifolds(const float* const restrict in, const size_t width, const size_t height,
                          const size_t ch, const float sigma,
                          const dt_iop_cacorrectrgb_guide_channel_t guide,
                          float* const restrict manifolds)
{
  float *const restrict blurred_in = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict manifold_higher = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict manifold_lower = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_manifold_higher = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict blurred_manifold_lower = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));

  float max[4] = {INFINITY, INFINITY, INFINITY, 1.0f};
  float min[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  dt_gaussian_t *g = dt_gaussian_init(width, height, 4, max, min, sigma, 0);
  if(!g) return;
  dt_gaussian_blur_4c(g, in, blurred_in);

  // construct the manifolds
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, blurred_in, manifold_lower, manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(in, blurred_in, manifold_lower, manifold_higher:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float pixelg = in[k * 4 + guide];
    const float avg = blurred_in[k * 4 + guide];
    const float weighth = pixelg >= avg;
    const float weightl = pixelg <= avg;
    for(size_t c = 0; c < 3; c++)
    {
      const float pixel = fmaxf(in[k * 4 + c], 1E-6);
      manifold_higher[k * 4 + c] = pixel * weighth;
      manifold_lower[k * 4 + c] = pixel * weightl;
    }
    manifold_higher[k * 4 + 3] = weighth;
    manifold_lower[k * 4 + 3] = weightl;
  }

  dt_gaussian_blur_4c(g, manifold_higher, blurred_manifold_higher);
  dt_gaussian_blur_4c(g, manifold_lower, blurred_manifold_lower);

  normalize_manifolds(blurred_in, blurred_manifold_lower, blurred_manifold_higher, width, height, guide);

  // note that manifolds were constructed based on the value and average
  // of the guide channel ONLY.
  // this implies that the "higher" manifold in the channel c may be
  // actually lower than the "lower" manifold of that channel.
  // This happens in the following example:
  // guide:  1_____
  //               |_____0
  // guided:        _____1
  //         0_____|
  // here the higher manifold of guide is equal to 1, its lower manifold is
  // equal to 0. The higher manifold of the guided channel is equal to 0
  // as it is the average of the values where the guide is higher than its
  // average, and the lower manifold of the guided channel is equal to 1.

  // refine the manifolds
  // improve result especially on very degraded images
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, blurred_in, manifold_lower, manifold_higher, blurred_manifold_lower, blurred_manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(in, blurred_in, manifold_lower, manifold_higher, blurred_manifold_lower, blurred_manifold_higher:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    // in order to refine the manifolds, we will compute weights
    // for which all channel will have a contribution.
    // this will allow to avoid taking too much into account pixels
    // that have wrong values due to the chromatic aberration
    //
    // for example, here:
    // guide:  1_____
    //               |_____0
    // guided: 1______
    //                |____0
    //               ^ this pixel makes the estimated lower manifold erroneous
    // here, the higher and lower manifolds values computed are:
    // _______|_higher_|________lower_________|
    // guide  |    1   |   0                  |
    // guided |    1   |(1 + 4 * 0) / 5 = 0.2 |
    //
    // the lower manifold of the guided is 0.2 if we consider only the guide
    // to determine weighth and weightl.
    //
    // at this step of the algorithm, we know if the higher manifold of guided
    // is actually higher or lower than the lower manifold.
    //
    // we can refine the manifolds by computing weights that promote pixels that
    // stretch the interval between the manifolds.
    // i.e., in our case, we give higher weights to the pixels that are equal to
    // 0 than to the pixel that is equal to 1 for the computation of the lower
    // manifold
    const float pixelg = in[k * 4 + guide];
    const float avgg = blurred_in[k * 4 + guide];
    if(pixelg > avgg)
    {
      // high manifold
      float weighth = 1.0f;
      for(size_t c = 0; c < 3; c++)
      {
        // if pixel value is close to the low manifold, give it a smaller weight
        // than if it is close to the high manifold
        const float pixel = logf(fmaxf(in[k * 4 + c], 1E-6));
        const float highc = logf(fmaxf(blurred_manifold_higher[k * 4 + c], 1E-6));
        const float lowc = logf(fmaxf(blurred_manifold_lower[k * 4 + c], 1E-6));
        float dist_manifolds = lowc - highc;
        float sign = (dist_manifolds >= 0) - (dist_manifolds < 0);
        dist_manifolds = fmaxf(fabsf(dist_manifolds), 1.0f);
        float dist = fminf(fmaxf(1.0f - sign * (pixel - highc) / dist_manifolds, 0.0f), 1.0f);
        dist *= dist;
        dist *= dist;
        weighth *= dist;
      }
      for(size_t c = 0; c < 3; c++)
      {
        const float pixel = fmaxf(in[k * 4 + c], 1E-6);
        manifold_higher[k * 4 + c] = pixel * weighth;
      }
      manifold_higher[k * 4 + 3] = weighth;
    }
    else
    {
      float weightl = 1.0f;
      for(size_t c = 0; c < 3; c++)
      {
        // if pixel value is close to the high manifold, give it a smaller weight
        // than if it is close to the low manifold
        const float pixel = logf(fmaxf(in[k * 4 + c], 1E-6));
        const float highc = logf(fmaxf(blurred_manifold_higher[k * 4 + c], 1E-6));
        const float lowc = logf(fmaxf(blurred_manifold_lower[k * 4 + c], 1E-6));
        float dist_manifolds = highc - lowc;
        float sign = (dist_manifolds >= 0) - (dist_manifolds < 0);
        dist_manifolds = fmaxf(fabsf(dist_manifolds), 1.0f);
        float dist = fminf(fmaxf(1.0f - sign * (pixel - lowc) / dist_manifolds, 0.0f), 1.0f);
        dist *= dist;
        dist *= dist;
        weightl *= dist;
      }
      for(size_t c = 0; c < 3; c++)
      {
        const float pixel = fmaxf(in[k * 4 + c], 1E-6);
        manifold_lower[k * 4 + c] = pixel * weightl;
      }
      manifold_lower[k * 4 + 3] = weightl;
    }
  }

  dt_gaussian_blur_4c(g, manifold_higher, blurred_manifold_higher);
  dt_gaussian_blur_4c(g, manifold_lower, blurred_manifold_lower);
  normalize_manifolds(blurred_in, blurred_manifold_lower, blurred_manifold_higher, width, height, guide);

  dt_gaussian_free(g);
  dt_free_align(manifold_lower);
  dt_free_align(manifold_higher);


#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(manifolds, blurred_manifold_lower, blurred_manifold_higher, width, height, guide) \
  schedule(simd:static) aligned(manifolds, blurred_manifold_lower, blurred_manifold_higher:64)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    for(size_t c = 0; c < 3; c++)
    {
      manifolds[k * 6 + c] = blurred_manifold_higher[k * 4 + c];
      manifolds[k * 6 + 3 + c] = blurred_manifold_lower[k * 4 + c];
    }
  }
  dt_free_align(blurred_in);
  dt_free_align(blurred_manifold_lower);
  dt_free_align(blurred_manifold_higher);
}

static void apply_correction(const float* const restrict in,
                          const float* const restrict manifolds,
                          const size_t width, const size_t height,
                          const size_t ch, const float sigma,
                          const dt_iop_cacorrectrgb_guide_channel_t guide,
                          const dt_iop_cacorrectrgb_mode_t mode,
                          float* const restrict out)

{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, width, height, guide, manifolds, out, sigma, mode) \
  schedule(simd:static) aligned(in, manifolds, out)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    const float high_guide = fmaxf(manifolds[k * 6 + guide], 1E-6);
    const float low_guide = fmaxf(manifolds[k * 6 + 3 + guide], 1E-6);
    const float log_high = logf(high_guide);
    const float log_low = logf(low_guide);
    const float pixelg = in[k * 4 + guide];
    const float log_pixg = logf(fminf(fmaxf(pixelg, low_guide), high_guide));
    float dist = fabsf(log_high - log_pixg) / fmaxf(fabsf(log_high - log_low), 1E-6);
    dist = fminf(dist, 1.0f);

    for(size_t kc = 1; kc <= 2; kc++)
    {
      const size_t c = (guide + kc) % 3;
      const float pixelc = in[k * 4 + c];

      const float diff_high_manifolds = manifolds[k * 6 + c] - high_guide;
      const float diff_low_manifolds = manifolds[k * 6 + 3 + c] - low_guide;
      const float diff = (diff_low_manifolds * dist) + (diff_high_manifolds * (1.0f - dist));
      const float estimate_d = pixelg + diff;

      const float ratio_high_manifolds = manifolds[k * 6 + c] / high_guide;
      const float ratio_low_manifolds = manifolds[k * 6 + 3 + c] / low_guide;
      const float ratio = powf(ratio_low_manifolds, dist) * powf(ratio_high_manifolds, (1.0f - dist));
      const float estimate_r = pixelg * ratio;

      float dist_dr = (pixelc - estimate_d) / (estimate_r - estimate_d);
      dist_dr = fminf(dist_dr, 1.0f);
      dist_dr = fmaxf(dist_dr, 0.0f);
      const float outp = estimate_d * (1.0f - dist_dr) + estimate_r * dist_dr;

      switch(mode)
      {
        case DT_CACORRECT_MODE_STANDARD:
          out[k * 4 + c] = outp;
          break;
        case DT_CACORRECT_MODE_DARKEN:
          out[k * 4 + c] = fminf(outp, in[k * 4 + c]);
          break;
        case DT_CACORRECT_MODE_BRIGHTEN:
          out[k * 4 + c] = fmaxf(outp, in[k * 4 + c]);
          break;
      }
    }

    out[k * 4 + guide] = pixelg;
    out[k * 4 + 3] = in[k * 4 + 3];
  }
}

void rbf(const float* const restrict in, float* const restrict out, const size_t width, const size_t height, const int guide, const float* spatial_force, const float range_force)
{
  // alloc temporary buffers for top-bottom and bottom-top results
  float *const restrict tb = dt_alloc_sse_ps(dt_round_size_sse(width * height * 4));
  float *const restrict bt = dt_alloc_sse_ps(dt_round_size_sse(width * height * 4));
  // first pass: top to bottom
  // copy-paste first row
  for(size_t j = 0; j < width; j++)
  {
    for(size_t c = 0; c < 3; c++)
    {
      tb[j * 4 + c] = in[j * 4 + c];
    }
    tb[j * 4 + 3] = 1.0f;
  }
  // iterate over the rows
  for(size_t i = 1; i < height; i++)
  {
    // handle j == 0
    for(size_t c = 0; c < 3; c++)
    {
      tb[i * width * 4 + c] = in[i * width * 4 + c];
    }
    tb[i * width * 4 + 3] = 1.0f;
    for(size_t j = 1; j < width-1; j++)
    {
      // compute distance between pixel and it 3 top pixels
      float dist_top = in[(i * width + j) * 4 + guide] - in[((i-1) * width + j) * 4 + guide];
      dist_top *= dist_top;
      float distm_top = in[(i * width + j) * 4 + guide] - tb[((i-1) * width + j) * 4 + guide];
      distm_top *= distm_top;
      dist_top = fmaxf(dist_top, distm_top);
      dist_top = exp2f(-dist_top * 10000.0f / range_force);

      float dist_top_left = in[(i * width + j) * 4 + guide] - in[((i-1) * width + j-1) * 4 + guide];
      dist_top_left *= dist_top_left;
      float distm_top_left = in[(i * width + j) * 4 + guide] - tb[((i-1) * width + j-1) * 4 + guide];
      distm_top_left *= distm_top_left;
      dist_top_left = fmaxf(dist_top_left, distm_top_left);
      dist_top_left = exp2f(-dist_top_left * 10000.0f / range_force);

      float dist_top_right = in[(i * width + j) * 4 + guide] - in[((i-1) * width + j+1) * 4 + guide];
      dist_top_right *= dist_top_right;
      float distm_top_right = in[(i * width + j) * 4 + guide] - tb[((i-1) * width + j+1) * 4 + guide];
      distm_top_right *= distm_top_right;
      dist_top_right = fmaxf(dist_top_right, distm_top_right);
      dist_top_right = exp2f(-dist_top_right * 10000.0f / range_force);

      float total_weight = dist_top + dist_top_left + dist_top_right + spatial_force[i * width + j];
      tb[(i * width + j) * 4 + 3] = total_weight;
      for(size_t c = 0; c < 3; c++)
      {
        tb[(i * width + j) * 4 + c] = dist_top * tb[((i-1) * width + j) * 4 + c] + dist_top_left * tb[((i-1) * width + j-1) * 4 + c] + dist_top_right * tb[((i-1) * width + j+1) * 4 + c] + in[(i * width + j) * 4 + c] * spatial_force[i * width + j];
        tb[(i * width + j) * 4 + c] /= total_weight;
      }
      tb[(i * width + j) * 4 + 3] = total_weight;
    }
    // handle j == width
    for(size_t c = 0; c < 3; c++)
    {
      tb[(i * width + width-1) * 4 + c] = in[(i * width + width-1) * 4 + c];
    }
    tb[(i * width + width-1) * 4 + 3] = 1.0f;
  }

  // second pass: bottom to top
  // copy-paste last row
  for(size_t j = 0; j < width; j++)
  {
    for(size_t c = 0; c < 3; c++)
    {
      bt[((height-1) * width + j) * 4 + c] = in[((height-1) * width + j) * 4 + c];
    }
    bt[((height-1) * width + j) * 4 + 3] = 1.0f;
  }
  // iterate over the rows
  for(int64_t i = height-2; i >= 0; i--)
  {
    // handle j == 0
    for(size_t c = 0; c < 3; c++)
    {
      bt[i * width * 4 + c] = in[i * width * 4 + c];
    }
    bt[i * width * 4 + 3] = 1.0f;
    for(size_t j = 1; j < width-1; j++)
    {
      // compute distance between pixel and it 3 bottom pixels
      // compute distance between pixel and it 3 top pixels
      float dist_bottom = in[(i * width + j) * 4 + guide] - in[((i+1) * width + j) * 4 + guide];
      dist_bottom *= dist_bottom;
      float distm_bottom = in[(i * width + j) * 4 + guide] - bt[((i+1) * width + j) * 4 + guide];
      distm_bottom *= distm_bottom;
      dist_bottom = fmaxf(dist_bottom, distm_bottom);
      dist_bottom = exp2f(-dist_bottom * 10000.0f / range_force);

      float dist_bottom_left = in[(i * width + j) * 4 + guide] - in[((i+1) * width + j-1) * 4 + guide];
      dist_bottom_left *= dist_bottom_left;
      float distm_bottom_left = in[(i * width + j) * 4 + guide] - bt[((i+1) * width + j-1) * 4 + guide];
      distm_bottom_left *= distm_bottom_left;
      dist_bottom_left = fmaxf(dist_bottom_left, distm_bottom_left);
      dist_bottom_left = exp2f(-dist_bottom_left * 10000.0f / range_force);

      float dist_bottom_right = in[(i * width + j) * 4 + guide] - in[((i+1) * width + j+1) * 4 + guide];
      dist_bottom_right *= dist_bottom_right;
      float distm_bottom_right = in[(i * width + j) * 4 + guide] - bt[((i+1) * width + j+1) * 4 + guide];
      distm_bottom_right *= distm_bottom_right;
      dist_bottom_right = fmaxf(dist_bottom_right, distm_bottom_right);
      dist_bottom_right = exp2f(-dist_bottom_right * 10000.0f / range_force);

      float total_weight = dist_bottom + dist_bottom_left + dist_bottom_right + spatial_force[i * width + j];
      bt[(i * width + j) * 4 + 3] = total_weight;
      for(size_t c = 0; c < 3; c++)
      {
        bt[(i * width + j) * 4 + c] = dist_bottom * bt[((i+1) * width + j) * 4 + c] + dist_bottom_left * bt[((i+1) * width + j-1) * 4 + c] + dist_bottom_right * bt[((i+1) * width + j+1) * 4 + c] + in[(i * width + j) * 4 + c] * spatial_force[i * width + j];
        bt[(i * width + j) * 4 + c] /= total_weight;
      }
      bt[(i * width + j) * 4 + 3] = total_weight;
    }
    // handle j == width
    for(size_t c = 0; c < 3; c++)
    {
      bt[(i * width + width-1) * 4 + c] = in[(i * width + width-1) * 4 + c];
    }
    bt[(i * width + width-1) * 4 + 3] = 1.0f;
  }

  // fusion of the 2 passes using the total weights
  for(size_t i = 0; i < height; i++)
  {
    for(size_t j = 0; j < width; j++)
    {
      const float weight_tb = tb[(i * width + j) * 4 + 3];
      const float weight_bt = bt[(i * width + j) * 4 + 3];
      const float total_weight = weight_tb + weight_bt;
      for(size_t c = 0; c < 3; c++)
      {
        const float value_tb = tb[(i * width + j) * 4 + c] * weight_tb;
        const float value_bt = bt[(i * width + j) * 4 + c] * weight_bt;
        out[(i * width + j) * 4 + c] = (value_tb + value_bt) / total_weight;
      }
    }
  }

  dt_free_align(tb);
  dt_free_align(bt);

  //TODO copy alpha channel
}


void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_cacorrectrgb_params_t *d = (dt_iop_cacorrectrgb_params_t *)piece->data;
  const float scale = piece->iscale / roi_in->scale;
  const int ch = piece->colors;
  const size_t width = roi_out->width;
  const size_t height = roi_out->height;
  const float* in = (float*)ivoid;
  float *const restrict transformed_in1 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict transformed_in2 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict laplacian_log = dt_alloc_sse_ps(dt_round_size_sse(width * height));
  float *const restrict spatial_weight = dt_alloc_sse_ps(dt_round_size_sse(width * height));
  float *const restrict corr1 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  float *const restrict corr2 = dt_alloc_sse_ps(dt_round_size_sse(width * height * ch));
  memcpy(corr2, corr1, sizeof(float)); //TODO remove. Only here to silent a warning.
  float* out = (float*)ovoid;
  const float sigma = MAX(d->radius / scale, 1.0f);

  if(ch != 4)
  {
    memcpy(out, in, width * height * ch * sizeof(float));
    return;
  }

  const dt_iop_cacorrectrgb_guide_channel_t guide = d->guide_channel;
  const dt_iop_cacorrectrgb_mode_t mode = d->mode;
  const float force = d->force;

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, transformed_in1, transformed_in2, width, height, guide, force) \
  schedule(simd:static) aligned(in, transformed_in1, transformed_in2)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    float guide_in = fmaxf(in[k * 4 + guide], 1E-4);
    transformed_in1[k * 4 + guide] = log2f(guide_in); // gamma correction
    transformed_in2[k * 4 + guide] = log2f(guide_in); // gamma correction
    for(size_t ci = 1; ci <= 2; ci++)
    {
      const size_t c = (guide + ci) % 3;
      float in_c = fmaxf(in[k * 4 + c], 1E-4);
      if(in_c > guide_in)
      {
        transformed_in1[k * 4 + c] = (2.0f - guide_in / in_c);
        transformed_in2[k * 4 + c] = (2.0f - guide_in / in_c);
      }
      else
      {
        transformed_in1[k * 4 + c] = (in_c / guide_in);
        transformed_in2[k * 4 + c] = (in_c / guide_in);
      }
    }
    // transformed_in1[k * 4 + ((guide + 1) % 3)] /= force; // useless with rbf
    // transformed_in2[k * 4 + ((guide + 2) % 3)] /= force; // useless with rbf
  }

  memset(laplacian_log, 0, sizeof(float) * width * height);
  // compute max diff at each pixel
  for(size_t i = 1; i < height-1; i++)
  {
    for(size_t j = 1; j < width-1; j++)
    {
      float max_diff = 0.0f;
      float center = transformed_in1[(i * width + j) * 4 + guide];
      for(size_t ii = i-1; ii <= i+1; ii++)
      {
        for(size_t jj = j-1; jj <= j+1; jj++)
        {
          float diff = fabsf(center - transformed_in1[(ii * width + jj) * 4 + guide]);
          if(diff > max_diff)
            max_diff = diff;
        }
      }
      laplacian_log[i * width + j] = max_diff;
    }
  }

  // smooth this difference with a gaussian blur
  float max = INFINITY;
  float min = 0.0f;
  dt_gaussian_t *g = dt_gaussian_init(width, height, 1, &max, &min, sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, laplacian_log, spatial_weight);
  dt_gaussian_free(g);

  // convert from log difference to ratio
  for(size_t k = 0; k < width * height; k++)
  {
    spatial_weight[k] = fmaxf(exp2f(-spatial_weight[k] * (2.0f * sigma + 1.0f) / 10.0f) / sigma, 0.0001f / sigma);
  }

  // use the smoothed difference in the rbf as spatial_force
  rbf(transformed_in1, corr1, width, height, guide, spatial_weight, exp2f(force-10.0f));

  // const float w = 1.f;
  // const float norm[4] = {w, w, w, 1.0f };
  // const dt_nlmeans_param_t params = { .scattering = (float)(d->radius) / 100.0f,
  //                                     .scale = scale,
  //                                     .luma = 1.0,    //no blending
  //                                     .chroma = 1.0,
  //                                     .center_weight = 1.0f,
  //                                     .sharpness = 4000.0f / d->force,
  //                                     .patch_radius = 0,
  //                                     .search_radius = 6,
  //                                     .decimate = 0,
  //                                     .norm = norm };
  // nlmeans_denoise(transformed_in1, corr1, roi_in, roi_out, &params);
  // nlmeans_denoise(transformed_in2, corr2, roi_in, roi_out, &params);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in, spatial_weight, corr1, corr2, out, width, height, guide, force) \
  schedule(simd:static) aligned(in, corr1, corr2, out)
#endif
  for(size_t k = 0; k < width * height; k++)
  {
    float guide_out = in[k * 4 + guide];
    out[k * 4 + guide] = guide_out + 0.0f * spatial_weight[k];//guide_out;
    size_t c1 = (guide + 1) % 3;
    size_t c2 = (guide + 2) % 3;
    float tin_c1 = corr1[k * 4 + c1];// * force;
    float tin_c2 = corr1[k * 4 + c2];// * force;//corr2
    if(tin_c1 > 1.0f)
    {
      out[k * 4 + c1] = guide_out / (2.0f - tin_c1);
    }
    else
    {
      out[k * 4 + c1] = guide_out * tin_c1;
    }
    if(tin_c2 > 1.0f)
    {
      out[k * 4 + c2] = guide_out / (2.0f - tin_c2);
    }
    else
    {
      out[k * 4 + c2] = guide_out * tin_c2;
    }
  }

  //FIXME free!
  dt_free_align(laplacian_log);
  dt_free_align(spatial_weight);
  return;

  const float downsize = 3.0f;
  const size_t ds_width = width / downsize;
  const size_t ds_height = height / downsize;
  float *const restrict ds_in = dt_alloc_sse_ps(dt_round_size_sse(ds_width * ds_height * ch));
  // we use only one variable for both higher and lower manifolds in order
  // to save time by doing only one bilinear interpolation instead of 2.
  float *const restrict ds_manifolds = dt_alloc_sse_ps(dt_round_size_sse(ds_width * ds_height * 6));
  // Downsample the image for speed-up
  interpolate_bilinear(in, width, height, ds_in, ds_width, ds_height, 4);
  get_manifolds(ds_in, ds_width, ds_height, ch, sigma / downsize, guide, ds_manifolds);
  dt_free_align(ds_in);
  float *const restrict manifolds = dt_alloc_sse_ps(dt_round_size_sse(width * height * 6));
  // upscale manifolds
  interpolate_bilinear(ds_manifolds, ds_width, ds_height, manifolds, width, height, 6);
  dt_free_align(ds_manifolds);
  apply_correction(in, manifolds, width, height, ch, sigma, guide, mode, out);
  dt_free_align(manifolds);
}

/** gui setup and update, these are needed. */
void gui_update(dt_iop_module_t *self)
{
  dt_iop_cacorrectrgb_gui_data_t *g = (dt_iop_cacorrectrgb_gui_data_t *)self->gui_data;
  dt_iop_cacorrectrgb_params_t *p = (dt_iop_cacorrectrgb_params_t *)self->params;

  dt_bauhaus_combobox_set_from_value(g->guide_channel, p->guide_channel);
  dt_bauhaus_slider_set_soft(g->radius, p->radius);
  dt_bauhaus_slider_set_soft(g->force, p->force);
  dt_bauhaus_combobox_set_from_value(g->mode, p->mode);
}

/** optional: if this exists, it will be called to init new defaults if a new image is
 * loaded from film strip mode. */
void reload_defaults(dt_iop_module_t *module)
{
  dt_iop_cacorrectrgb_params_t *d = (dt_iop_cacorrectrgb_params_t *)module->default_params;

  d->guide_channel = DT_CACORRECT_RGB_G;
  d->radius = 5.0f;
  d->force = 5.0f;
  d->mode = DT_CACORRECT_MODE_STANDARD;

  dt_iop_cacorrectrgb_gui_data_t *g = (dt_iop_cacorrectrgb_gui_data_t *)module->gui_data;
  if(g)
  {
    dt_bauhaus_combobox_set_default(g->guide_channel, d->guide_channel);
    dt_bauhaus_slider_set_default(g->radius, d->radius);
    dt_bauhaus_slider_set_soft_range(g->radius, 1.0, 20.0);
    dt_bauhaus_slider_set_default(g->force, d->force);
    dt_bauhaus_slider_set_soft_range(g->force, 1.0, 20.0);
    dt_bauhaus_combobox_set_default(g->mode, d->mode);
  }
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_cacorrectrgb_gui_data_t *g = IOP_GUI_ALLOC(cacorrectrgb);
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);
  g->guide_channel = dt_bauhaus_combobox_from_params(self, "guide_channel");
  gtk_widget_set_tooltip_text(g->guide_channel, _("channel used as a reference to\n"
                                           "correct the other channels.\n"
                                           "use sharpest channel if some\n"
                                           "channels are blurry.\n"
                                           "try changing guide channel if you\n"
                                           "have artefacts."));
  g->radius = dt_bauhaus_slider_from_params(self, "radius");
  gtk_widget_set_tooltip_text(g->radius, _("increase for stronger correction\n"));

  g->force = dt_bauhaus_slider_from_params(self, "force");
  gtk_widget_set_tooltip_text(g->force, _("increase for stronger correction\n"));

  gtk_box_pack_start(GTK_BOX(self->widget), dt_ui_label_new(_("advanced parameters:")), TRUE, TRUE, 0);
  g->mode = dt_bauhaus_combobox_from_params(self, "mode");
  gtk_widget_set_tooltip_text(g->mode, _("correction mode to use.\n"
                                         "can help with multiple\n"
                                         "instances for very damaged\n"
                                         "images.\n"
                                         "darken only is particularly\n"
                                         "efficient to correct blue\n"
                                         "chromatic aberration."));
}
