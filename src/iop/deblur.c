/*
    This file is part of darktable,
    Copyright (C) 2010-2021 darktable developers.

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

#include <gtk/gtk.h>
#include <stdlib.h>


DT_MODULE_INTROSPECTION(1, dt_iop_deblur_params_t)


/**
 * ===========================
 * DESCRIPTION OF THE APPROACH
 * ===========================
 *
 * --------------------------------------------------
 ** (1) 2D blurs as a composition of several 1D blurs
 * --------------------------------------------------
 * We only consider here blurs that can be decomposed as a convolution of several
 * 1D blurs.
 * In particular, the module focuses on a blur applied consecutively and with same
 * radius along the horizontal, vertical, and the 2 diagonal axis.
 * (The approach is more generic though, and could work for any blur that can
 * be expressed with several 1D blurs, whatever their directions and radius)
 *
 * ---------------------
 ** (2) sum and integral
 * ---------------------
 * A sum is very similar to an integral and vice versa, in the discrete and
 * continuous domain.
 *
 * ------------------------------------
 ** (3) average and normalized integral
 * ------------------------------------
 * An average is a sum normalized by the total number of elements.
 * Similarly, we can normalize an integral, by dividing it by the integration width.
 *
 * --------------------------------------------
 ** (4) 1D function approximation/interpolation
 * --------------------------------------------
 * Functions can be approximated, or interpolated in case of discrete functions,
 * in various ways.
 * Typical ways are to use cosinus and sinus with Fourier transform, or to use
 * polynomials (either with Lagrange polynomial interpolation for discrete data,
 * polynomial approximation e.g. with Taylor's theorem).
 *
 * ---------------------------------------
 ** (5) function approximation with arctan
 * ---------------------------------------
 * We propose to use a local function approximation with a sum of arctan.
 * arctan has the nice property of having a very stable behavior when x gets
 * farther from 0, contrary to cosinus sinus and most polynomials that are not
 * stable at all.
 *
 * --------------------------------------
 ** (6) pixel perception (in sharp image)
 * --------------------------------------
 * We consider that the value recorded by the camera for one pixel corresponds
 * to the normalized integral of the light between -0.5 and 0.5 (assuming 1 is
 * the pixel's width).
 *
 * ---------------------------------------
 ** (7) blur impact
 * ---------------------------------------
 * In case of blur, the value recorded by the camera for one pixel corresponds
 * in our model to the normalized integral of the light between -blur_radius
 * and +blur_radius.
 *
 * -----------------------
 ** (8) integral of arctan
 * -----------------------
 * We want to use a sum of arctan to represent the light signal.
 * arctan primitive is x arctan(x)-1/2 log(x²+1)
 * When blurred (equivalently, under normalized integration), an arctan becomes:
 * (((x-blur_radius)arctan(x-blur_radius)-1/2 log(((x-blur_radius)²+1))
 *    - ((x+blur_radius)arctan(x+blur_radius)-1/2 log(((x+blur_radius)²+1))) / (2*blur_radius)
 *
 * --------------------------------------------
 ** (9) using arctan+x/(x²+1) instead of arctan
 * --------------------------------------------
 * To make computations easier and faster, we will actually represent the
 * light signal as a sum of arctan+x/(x²+1) instead of arctan.
 * This makes the primitive simpler: x arctan(x).
 * When blurred (equivalently, under normalized integration), an arctan+x/(x²+1)
 * becomes:
 * ((x-blur_radius)arctan(x-blur_radius)-((x+blur_radius)arctan(x+blur_radius))) / (2*blur_radius)
 *
 * This does not change the behavior of the function, arctan+x/(x²+1) has very
 * similar behavior to arctan.
 *
 * -----------------------
 ** (10) naming convention
 * -----------------------
 * The function:
 * x, blur_radius -> ((x-blur_radius)arctan(x-blur_radius)
 *                    -((x+blur_radius)arctan(x+blur_radius))) / (2*blur_radius)
 * will be called f in the following.
 *
 * --------------------------------------
 ** (11) from blurry image to sharp image
 * --------------------------------------
 * We now have all the background of the approach to explain it.
 * The approach consist in doing a local interpolation of a row of the image
 * with a weighted sum of f.
 * For a blur of radius r, we consider the 4r+1 pixels around a center pixel:
 * 2r pixels to the left, 2r pixels to the right.
 * Then, we consider a sum of 4r f functions, centered in the 4r x positions that
 * exist between pixels (this allows to create a dirac ).
 * We also consider a constant in this sum as well.
 * The sum looks like this:
 *     a_1 f(x-2r+0.5,r)
 *   + a_2 f(x-2r+1+.5,r)
 *   + a_3 f(x-2r+2+.5,r)
 *   + a_4 f(x-2r+3+.5,r)
 *   + ...
 *   + a_4r f(x+2r-0.5,r)
 *   + c
 *
 * We thus have 4r+1 variables (a_1 to a_4r, plus c), and 4r+1 pixel values.
 * We can use gaussian elimination to solve the system.
 * Once we known all the 4r+1 variables, we get the sharp estimate of the image
 * by computing the same sum, but with a blur radius no larger than a pixel (see
 * r is replaced by 0.5 in the formula):
 *     a_1 f(x-2r+0.5,0.5)
 *   + a_2 f(x-2r+1+.5,0.5)
 *   + a_3 f(x-2r+2+.5,0.5)
 *   + a_4 f(x-2r+3+.5,0.5)
 *   + ...
 *   + a_4r f(x+2r-0.5,0.5)
 *   + c
 *
 * Once the blur is reverted along one axis, do the same along the other axis.
 *
 * Note: the matrix of the left part of the system of equation can be inverted once for all.
 * Then, for each pixel we get the result with a matrix-vector product between the
 * inverted matrix and the vector of 4r+1 pixel values.
 *
 * ----------------------------
 ** (12) improving the approach
 * ----------------------------
 * The approach may be improved by the following ideas:
 * - use a scaling factor inside the arctan, for instance using arctan(2x) instead of arctan(x)
 * - perform a gaussian blur of small radius to make the blur applied on the image due to lens
 *   blur followed by the gaussian blur closer to the blur we are able to handle. This may
 *   improve the result because it may not be possible to decompose lens blurs as 1D blurs.
 * - in noisy context, or in the context of an image with sharp and blurry areas, it may
 *   be interesting weight the result depending on the sparsity of the a_1...a_4r
 *   coefficients, as we expect them to be quite sparse if the blur radius is correct.
 *
 * ========================
 * EXAMPLE OF THE ALGORITHM
 * ========================
 * ---------------------------------------
 ** (1) blur parameters
 * ---------------------------------------
 *
 * ---------------------------------------
 ** (2) image interpolation
 * ---------------------------------------
 *
 * ---------------------------------------
 ** (3) sharp image recovery
 * ---------------------------------------
 *
 **/


typedef struct dt_iop_deblur_params_t
{
  int radius; // $MIN: 1 $MAX: 30 $DEFAULT: 1 $DESCRIPTION: "blur radius"
} dt_iop_deblur_params_t;

typedef struct dt_iop_deblur_gui_data_t
{
  GtkWidget *radius;
} dt_iop_deblur_gui_data_t;

const char *name()
{
  return _("deblur");
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

const char *description(struct dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("deblur an image"),
                                      _("corrective"),
                                      _("linear, raw, scene-referred"),
                                      _("linear, raw"),
                                      _("linear, raw, scene-referred"));
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

int legacy_params(dt_iop_module_t *self, const void *const old_params, const int old_version,
                  void *new_params, const int new_version)
{
  return 1;
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, p1, self->params_size);
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  //dt_iop_deblur_params_t *d = (dt_iop_deblur_params_t *)piece->data;
  const size_t ch = piece->colors;
  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  memcpy(ovoid, ivoid, ch * width * height * sizeof(float));
  //TODO essai avec rayon 1 :
  // flouter horizontalement, puis déflouter en faisant à chaque pixel la multiplication matricielle pour trouver les coefs


}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_deblur_gui_data_t *g = (dt_iop_deblur_gui_data_t *)self->gui_data;
  dt_iop_deblur_params_t *p = (dt_iop_deblur_params_t *)self->params;

  dt_bauhaus_slider_set(g->radius, p->radius);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_deblur_gui_data_t *g = IOP_GUI_ALLOC(deblur);
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  g->radius = dt_bauhaus_slider_from_params(self, "radius");
}

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
