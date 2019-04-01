#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

// call this to find a and b parameters.
// to begin, only find the a parameter.
// should be called with the pfm of the original image and the one
// of the denoised image as commandline parameters.

static float*
read_pfm(const char *filename, int *wd, int*ht)
{
  FILE *f = fopen(filename, "rb");
  if(!f) return 0;
  fscanf(f, "PF\n%d %d\n%*[^\n]", wd, ht);
  fgetc(f); // eat only one newline

  float *p = (float *)malloc(sizeof(float)*3*(*wd)*(*ht));
  fread(p, sizeof(float)*3, (*wd)*(*ht), f);
  for(int k=0;k<3*(*wd)*(*ht);k++) p[k] = fmaxf(0.0f, p[k]);
  fclose(f);
  return p;
}

#if 0
static void
write_pfm(const char *filename, float *buf, int wd, int ht)
{
  FILE *f = fopen(filename, "wb");
  if(!f) return;
  fprintf(f, "PF\n%d %d\n-1.0\n", wd, ht);
  fwrite(buf, sizeof(float)*3, wd*ht, f);
  fclose(f);
}
#endif


#define MIN(a,b) ((a>b)?b:a)
#define MAX(a,b) ((a>b)?a:b)

static inline float
clamp(float f, float m, float M)
{
  return MAX(MIN(f, M), m);
}

#define NB_BITS_PRECISION 256
int main(int argc, char *arg[])
{
  if(argc < 3)
  {
    fprintf(stderr, "usage: %s input_noisy.pfm input_smooth.pfm\n", arg[0]);
    exit(1);
  }
  int wd, ht;
  float *input = read_pfm(arg[1], &wd, &ht);
  float *inputblurred = read_pfm(arg[2], &wd, &ht);
  double var[3][NB_BITS_PRECISION];
  unsigned nb_elts[3][NB_BITS_PRECISION];
  for(int level = 0; level < NB_BITS_PRECISION; level++)
  {
    for(int c = 0; c < 3; c++)
    {
      var[c][level] = 0.0;
      nb_elts[c][level] = 0;
    }
  }
  if(argc < 10)
  {
    for(int i = 0; i < ht; i++)
    {
      for(int j = 0; j < wd; j++)
      {
        for(int c = 0; c < 3; c++)
        {
          int index = (i * wd + j) * 3 + c;
          float pixel_diff = input[index] - inputblurred[index];
          unsigned level = (unsigned)(inputblurred[index] * NB_BITS_PRECISION);
          if(level < NB_BITS_PRECISION)
          {
            var[c][level] += pixel_diff * pixel_diff;
            nb_elts[c][level]++;
          }
        }
      }
    }
    for(int level = 0; level < NB_BITS_PRECISION; level++)
    {
      for(int c = 0; c < 3; c++)
      {
        if(nb_elts[c][level] > 0) var[c][level] /= nb_elts[c][level];
      }
      fprintf(stdout, "%f %f %f %f %d %d %d\n", level / (float)NB_BITS_PRECISION, var[0][level], var[1][level],
              var[2][level], nb_elts[0][level], nb_elts[1][level], nb_elts[2][level]);
    }
  }
  if(argc >= 10 && !strcmp(arg[3], "-c"))
  {
    const float a[3] = { atof(arg[4]), atof(arg[5]), atof(arg[6]) },
                b[3] = { atof(arg[7]), atof(arg[8]), atof(arg[9]) };

    // perform anscombe transform
    for(int i = 0; i < ht; i++)
    {
      for(int j = 0; j < wd; j++)
      {
        for(int c = 0; c < 3; c++)
        {
          int index = (i * wd + j) * 3 + c;
          input[index] /= a[c];
          float d = fmaxf(0.0f, input[index] + 3.0 / 8.0 + (b[c] / a[c]) * (b[c] / a[c]));
          input[index] = 2.0f * sqrtf(d);

          unsigned level = (unsigned)(inputblurred[index] * NB_BITS_PRECISION);
          inputblurred[index] /= a[c];
          d = fmaxf(0.0f, inputblurred[index] + 3.0 / 8.0 + (b[c] / a[c]) * (b[c] / a[c]));
          inputblurred[index] = 2.0f * sqrtf(d);

          float pixel_diff = input[index] - inputblurred[index];
          var[c][level] += pixel_diff * pixel_diff;
          nb_elts[c][level]++;
        }
      }
    }
    for(int level = 0; level < NB_BITS_PRECISION; level++)
    {
      for(int c = 0; c < 3; c++)
      {
        if(nb_elts[c][level] > 0)
        {
          var[c][level] /= nb_elts[c][level];
        }
      }
      fprintf(stdout, "%f %f %f %f %d %d %d\n", level / (float)NB_BITS_PRECISION, var[0][level], var[1][level],
              var[2][level], nb_elts[0][level], nb_elts[1][level], nb_elts[2][level]);
    }
  }
  free(inputblurred);
  free(input);
  exit(0);
}
