#pragma once

#include "io.hpp"

namespace heat {
// These constants are used by the simulation
struct Constants {
    // Diffusion constant
    const double a = 0.0;
    // The inverse of the square of the grid spacing
    const double inv_dx2 = 0.0;
    const double inv_dy2 = 0.0;
    // Time step computed from the diffusion constant and the grid spacing
    const double dt = 0.0;

    Constants(const Input &input)
        : a(input.diffusion_constant),
          inv_dx2(1.0 / (input.grid_spacing_x * input.grid_spacing_x)),
          inv_dy2(1.0 / (input.grid_spacing_y * input.grid_spacing_y)),
          dt(1.0 /
             (inv_dx2 * inv_dy2 * 2.0 * a * (1.0 / inv_dx2 + 1.0 / inv_dy2))) {}
    };
}
