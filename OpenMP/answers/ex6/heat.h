/* 2D heat equation

   Copyright (C) 2014  CSC - IT Center for Science Ltd.

   Licensed under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Code is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   Copy of the GNU General Public License can be onbtained from
   see <http://www.gnu.org/licenses/>.
*/

// Datatype for temperature field
typedef struct {
    /* nx and ny are the true dimensions of the field. The array data
       contains also ghost layers, so it will have dimensions nx+2 x ny+2 */
    int nx;
    int ny;
    double dx;
    double dy;
    double dx2;
    double dy2;
    double **data;
} field;

// We use here fixed grid spacing
#define DX 0.01
#define DY 0.01

double **malloc_2d(int nx, int ny);

void free_2d(double **array);

void initialize_field_metadata(field * temperature, int nx, int ny);

void initialize(field * temperature1, field * temperature2);

void evolve(field * curr, field * prev, double a, double dt);

void output(field * temperature, int iter);

void read_input(field * temperature1, field * temperature2,
                char *filename);

void copy_field(field * temperature1, field * temperature2);

void swap_fields(field * temperature1, field * temperature2);

void finalize(field * temperature1, field * temperature2);
