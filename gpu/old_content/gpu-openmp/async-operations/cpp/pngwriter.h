#ifndef PNGWRITER_H_
#define PNGWRITER_H_

#if __cplusplus
  extern "C" {
#endif
int save_png(int *data, const int nx, const int ny, const char *fname);

#if __cplusplus
  }
#endif
#endif
