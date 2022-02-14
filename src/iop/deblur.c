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
#include "iop/gaussian_elimination.h"

#include <gtk/gtk.h>
#include <stdlib.h>
#include <math.h>


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
 * For a blur of radius r, we consider the 2m+1 pixels around a center pixel:
 * m pixels to the left, m pixels to the right. (The choice of the value of
 * m will be discussed in part (12)).
 * Then, we consider a sum of 2m f functions, centered in the 2m x positions that
 * exist between pixels (this allows to create a dirac ).
 * We also consider a constant in this sum as well.
 * The sum looks like this:
 *     a_1 f(x-m+0.5,r)
 *   + a_2 f(x-m+1+.5,r)
 *   + a_3 f(x-m+2+.5,r)
 *   + a_4 f(x-m+3+.5,r)
 *   + ...
 *   + a_2m f(x+m-0.5,r)
 *   + c
 *
 * We thus have 2m+1 variables (a_1 to a_2m, plus c), and 2m+1 pixel values.
 * We can use gaussian elimination to solve the system.
 * Once we known all the 2m+1 variables, we get the sharp estimate of the image
 * by computing the same sum, but with a blur radius no larger than a pixel (see
 * r is replaced by 0.5 in the formula):
 *     a_1 f(x-m+0.5,0.5)
 *   + a_2 f(x-m+1+.5,0.5)
 *   + a_3 f(x-m+2+.5,0.5)
 *   + a_4 f(x-m+3+.5,0.5)
 *   + ...
 *   + a_4r f(x+m-0.5,0.5)
 *   + c
 *
 * Once the blur is reverted along one axis, do the same along the other axis.
 *
 * Note: the matrix of the left part of the system of equation can be inverted once for all.
 * Then, for each pixel we get the result with a matrix-vector product between the
 * inverted matrix and the vector of 2m+1 pixel values.
 *
 * -----------------------------------------------
 ** (12) adding a scaling factor inside the arctan
 * -----------------------------------------------
 * A scaling factor can be used inside the arctan: for instance we can use
 * arctan(2x) instead of arctan(x).
 * See in the example of the algorithm below.
 *
 * ------------------------------
 ** (13) choosing the value of m
 * ------------------------------
 * The use of an interpolation based on arctan allows to keep the problem
 * local: at some points, adding more f functions centered far away from the
 * pixel will be equivalent to change the constant value, as the f functions
 * have a derivative that is almost null for points far away from their center.
 * The question is: how far away can we consider that the f functions are well
 * approximated by a constant function at the considered pixel position.
 *
 * We experimentally looked at the value of the f function for various blur
 * at x >= r.
 * The scaling factor heavily influence the margin we need to take
 * Here are the m values we need to take to have a derivative <= 0.001:
 * scaling_factor |   r | m-r
 * 1              |   1 | 7
 * 1              | 100 | 8.69
 * 10             |   1 | 0.861
 * 10             | 100 | 0.869
 * 100            |   1 | 0.0869
 * 100            | 100 | 0.0869
 * 1000           |   1 | 0.0087
 * 1000           | 100 | 0.0087
 *
 * The width of the inflexion part of the curve is very stable
 * forall radius (m-r variation is very small when changing r).
 *
 * We want m-r as small as possible in order to make the algorithm faster.
 * Taking m=r+1 is a sufficient margin (for our threshold of 0.001)
 * as soon as the scaling_factor becomes higher than 8.6.
 *
 * -------------------------
 ** (15) complexity analysis
 * -------------------------
 * The algorithm complexity is decomposed as (r begin the blur radius and n the
 * number of pixels in the image):
 * - matrix invertion: O(r^3)
 * - for all pixels, matrix-vector product: O(r²n)
 * - for all pixels, evaluation of the sum of 2m f functions: O(rn)
 *
 * The matrix-vector products dominates the overall complexity (n being way
 * larger than r in our case).
 *
 * --------------------------
 ** (15) from O(r²n) to O(rn)
 * --------------------------
 * The algorithm can be sped-up by solving a larger system of linear equations
 * that gives solutions simultaneously for several pixels.
 * If we consider 2 side-by-side pixels, and solving the system of the 2m+2
 * surrounding pixels gives solutions for both of these pixels.
 * The optimal number of pixels to recover at once is 2m: complexity is (2m+c)²/c
 * Having c multiple of m is optimal, and removes the squaring. Then, if we consider
 * the simplified fraction (2+d)²/d and we derivate it for d, we can see that d=2
 * gives the minimum value. So c = dm = 2m is optimal.
 * This gives a system of 2m+2m equations.
 * The complexity of the algorithm becomes in this case:
 * - matrix invertion: complexity remains O(r^3) but constant is multiplied by 8.
 * - for each set of 2m pixels, we do one matrix-vector product: O((4m)²n/(2m)) = O(4mn) = O(rn)
 * - for all pixels, we evaluate of the sum of 4m f functions (complexity remains O(rn)
 *   but constant is multiplied by 2). Note that we don't really need to evaluate
 *   4m functions, but 2m+1 is enough: all the f functions centered far away from the
 *   considered pixel can be considered as constant, and we can evaluate them
 *   by doing the sum of their coefficients multiplied by PI/2 (arctan(inf))
 *   or -PI/2 (arctan(-inf)) depending if the f functions were on the left or
 *   the right of the considered pixel.
 *
 * ----------------------------
 ** (16) improving the approach
 * ----------------------------
 * The approach may be improved by the following ideas:
 * - perform a gaussian blur of small radius to make the blur applied on the image due to lens
 *   blur followed by the gaussian blur closer to the blur we are able to handle. This may
 *   improve the result because it may not be possible to decompose lens blurs as 1D blurs.
 * - in noisy context, or in the context of an image with sharp and blurry areas, it may
 *   be interesting weight the result depending on the sparsity of the a_1...a_2m
 *   coefficients, as we expect them to be quite sparse if the blur radius is correct.
 *
 * ========================
 * EXAMPLE OF THE ALGORITHM
 * ========================
 *
 * Our example will be made for this (sharp) line of pixel:
 *         _ _ _
 *
 *
 * _ _ _ _       _ _
 * 0 0 0 0 1 1 1 0 0
 *
 * We will consider a simple blur that blurs 3 pixels together.
 * The radius will be 1.5.
 * The line becomes:
 *           _
 *         _   _
 *       _       _
 *   _ _
 * ? 0 0 ⅓ ⅔ 1 ⅔ ⅓ ?
 *         ^
 * We want to recover the original value of the central pixel.
 * We will need 4r+1=7 pixels to invert the blur.
 *
 * Our sum of f functions is:
 * s(x) = a1 f(x+2.5, 1.5) + a2 f(x+1.5, 1.5) + a3 f(x+0.5, 1.5)
 *        + a4 f(x-0.5, 1.5) + a5 f(x-1.5, 1.5) + a6 f(x-2.5, 1.5) + c
 *
 * Note how each f function is centered in between 2 pixels.
 *
 * Let's find the coefficients.
 * We have a set of 7 equations:
 * s(-3) = 0
 * s(-2) = 0
 * s(-1) = 1/3
 * s(0)  = 2/3
 * s(1)  = 1
 * s(2)  = 2/3
 * s(3)  = 1/3
 *
 * Rewriten in matrix vector notation:
 *
 *      ⎡a1⎤   ⎡0⎤
 *      |a2|   |0|
 *      |a3|   |⅓|
 *      |a4|   |⅔|
 *      |a5|   |1|
 *      |a6|   |⅔|
 *  M x ⎣ c⎦ = ⎣⅓⎦
 *
 * each line of M is defined using one of the equation, the top line being:
 * (the r in f(y,r) is omitted here to save space)
 *  | f(-3+2.5), f(-3+1.5), f(-3+0.5), f(-3-0.5), f(-3-1.5), f(-3-2.5), 1/3 | <- corresponds to s(-3) = 0
 *
 * M =
 *  ⎡ f(-0.5), f(-1.5), f(-2.5), f(-3.5), f(-4.5), f(-5.5), 1/3 ⎤
 *  |  f(0.5), f(-0.5), f(-1.5), f(-2.5), f(-3.5), f(-4.5), 1/3 |
 *  |  f(1.5),  f(0.5), f(-0.5), f(-1.5), f(-2.5), f(-3.5), 1/3 |
 *  |  f(2.5),  f(1.5),  f(0.5), f(-0.5), f(-1.5), f(-2.5), 1/3 |
 *  |  f(3.5),  f(2.5),  f(1.5),  f(0.5), f(-0.5), f(-1.5), 1/3 |
 *  |  f(4.5),  f(3.5),  f(2.5),  f(1.5),  f(0.5), f(-0.5), 1/3 |
 *  ⎣  f(5.5),  f(4.5),  f(3.5),  f(2.5),  f(1.5),  f(0.5), 1/3 ⎦
 *
 * Let's compute all the f(x).
 * M =
 *  ⎡  0.476299757397,    1.2490457724,   1.50595749709,   1.55090213305,   1.56224952636,   1.56634141689, 0.3333333333 ⎤
 *  | -0.476299757397,  0.476299757397,    1.2490457724,   1.50595749709,   1.55090213305,   1.56224952636, 0.3333333333 |
 *  |   -1.2490457724, -0.476299757397,  0.476299757397,    1.2490457724,   1.50595749709,   1.55090213305, 0.3333333333 |
 *  |  -1.50595749709,   -1.2490457724, -0.476299757397,  0.476299757397,    1.2490457724,   1.50595749709, 0.3333333333 |
 *  |  -1.55090213305,  -1.50595749709,   -1.2490457724, -0.476299757397,  0.476299757397,    1.2490457724, 0.3333333333 |
 *  |  -1.56224952636,  -1.55090213305,  -1.50595749709,   -1.2490457724, -0.476299757397,  0.476299757397, 0.3333333333 |
 *  ⎣  -1.56634141689,  -1.56224952636,  -1.55090213305,  -1.50595749709,   -1.2490457724, -0.476299757397, 0.3333333333 ⎦
 *
 * Now, let's invert M.
 * M⁻¹ =
 * [[ -9.66909668  26.16063717 -25.73193133   2.63325878  22.73593626  -27.14947387  11.02066968]
 * [ 16.49154048 -41.72731856  39.10432859  -4.3091701  -33.14364462   39.71306842 -16.1288042 ]
 * [ -9.24039084  23.10894135 -21.03092342   3.69552722  13.0262261  -16.16651247   6.60713206]
 * [ -6.60713206  16.16651247 -13.0262261   -3.69552722  21.03092342  -23.10894135   9.24039084]
 * [ 16.1288042  -39.71306842  33.14364462   4.3091701  -39.10432859   41.72731856 -16.49154048]
 * [-11.02066968  27.14947387 -22.73593626  -2.63325878  25.73193133  -26.16063717   9.66909668]
 * [  3.70279557  -2.09507846  -2.63792858   5.06042293  -2.63792858   -2.09507846   3.70279557]]
 *
 * Note: M construction and inversion can be done once, it does not vary with considered pixel.
 *
 * Now, we can compute the a1...a6,c coefficients.
 * ⎡a1⎤         ⎡0⎤
 * |a2|         |0|
 * |a3|         |⅓|
 * |a4|         |⅔|
 * |a5|         |1|
 * |a6|         |⅔|
 * ⎣ c⎦ = M⁻¹ x ⎣⅓⎦
 *
 * ⎡a1⎤   ⎡  9.59296953⎤
 * |a2|   |-18.22074268|
 * |a3|   | 12.51142956|
 * |a4|   |  3.811045  |
 * |a5|   |-12.85539939|
 * |a6|   |  8.63605949|
 * ⎣ c⎦ = ⎣  0.28049264⎦
 *
 * Now, we can estimate our pixel value as:
 * p(x) = a1 f(x+2.5, 0.5) + a2 f(x+1.5, 0.5) + a3 f(x+0.5, 0.5)
 *        + a4 f(x-0.5, 0.5) + a5 f(x-1.5, 0.5) + a6 f(x-2.5, 0.5) + c
 * evaluated in 0.
 *
 * p(0) = 1.1256
 *
 * Even though the value is not perfectly reconstructed, it is much closer
 * to the sharp value than before.
 *
 * Using a large scaling factor inside the arctan of f improves the estimate a lot
 *  f: x, blur_radius -> ((x-blur_radius)arctan((x-blur_radius)*scaling_factor)
 *                    -((x+blur_radius)arctan((x+blur_radius)*scaling_factor))) / (2*blur_radius)
 *
 * A scaling factor of 8 leads to the following result:
 *
 * ⎡a1⎤   ⎡ 0.01498434⎤
 * |a2|   |-0.00133442|
 * |a3|   |-0.34631693|
 * |a4|   | 0.03102008|
 * |a5|   |-0.03168486|
 * |a6|   | 0.3476704 |
 * ⎣ c⎦ = ⎣-0.00695036⎦   (note that the vector is much more sparse)
 *
 * p(0) = 1.0137
 *
 * A scaling factor of 100 leads to the following result:
 *
 * ⎡a1⎤   ⎡ 1.02624874e-03⎤
 * |a2|   |-6.59300098e-06|
 * |a3|   |-3.20352518e-01|   < a3 and a6 contain almost all the information for this example
 * |a4|   | 2.05897921e-03|
 * |a5|   |-2.06227428e-03|
 * |a6|   | 3.20359118e-01|   < a3 and a6 contain almost all the information for this example
 * ⎣ c⎦ = ⎣-5.32232281e-04⎦   (note that the vector is even more sparse)
 *
 * p(0) = 1.0011
 *
 **/


typedef struct dt_iop_deblur_params_t
{
  float radius; // $MIN: 1.0 $MAX: 30.0 $DEFAULT: 1.0 $DESCRIPTION: "blur radius"
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

static inline float f(float x, float r)
{
  // big scaling_factor seems to give better results
  const float scaling_factor = 100.0f;
  return ((x - r) * atanf(scaling_factor * (x - r))
          - (x + r) * atanf(scaling_factor * (x + r))) / (2.0f * r);
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  //dt_iop_deblur_params_t *d = (dt_iop_deblur_params_t *)piece->data;
  const size_t ch = piece->colors;
  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  double test[9] = {0.1, 0.5, 0.7, 0.8, 0.1, 0.5, 0.7, 0.7, 0.9};
  double* inv = gauss_invert((double*)test, 3);
  if(inv != NULL)
    printf("%lf  %lf  %lf\n%lf  %lf  %lf\n%lf  %lf  %lf\n", inv[0], inv[1], inv[2], inv[3], inv[4], inv[5], inv[6], inv[7], inv[8]);
  memcpy(ovoid, ivoid, ch * width * height * sizeof(float));
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
