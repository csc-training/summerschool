#ifndef FIELD_H_
#define FIELD_H_

#define NX 256
#define NY 256

// Field structure definition
typedef struct {
    int nx;
    int ny;
    double dx;
    double dy;
    double dx2;
    double dy2;
    double data[NX + 2][NY + 2];
} field;

#endif
