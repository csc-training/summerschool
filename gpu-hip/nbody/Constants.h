#ifndef CONSTANTS_H_
#define CONSTANTS_H_

//#define NULL 0

#define WIDTH	1024 // Image render width
#define HEIGHT	1024 // Image render height
#define NUM_BODIES (1024*32) // Number of small particles
#define PI      3.14159265358979323846 
#define TO_METERS 1.496e11 // Meters in an AU
#define SYSTEM_SIZE 3.5    // Farthest particles in AU
#define SYSTEM_THICKNESS 0.08  //  Thickness in AU
#define INNER_BOUND 0.3    // Closest particles to center in AU
#define SOFTENING (0.015*TO_METERS) // Softens particles interactions at close distances
#define SOLAR_MASS 2.0e30  // in kg
#define BINARY_SEPARATION 0.07 // AU (only applies when binary code uncommented)
#define EXTRA_MASS 1.5 // 0.02 Disk mask as a portion of center star/black hole mass
#define ENABLE_FRICTION 0 // For experimentation only. Will probably cause weird results
#define FRICTION_FACTOR 25.0 // Only applies if friction is enabled
#define MAX_DISTANCE 0.75 //2.0  Barnes-Hut Distance approximation factor
#define G 6.67408e-11 // The gravitational constant
#define RENDER_SCALE 2.5 // "Zoom" of images produced
#define MAX_VEL_COLOR 40000.0  // Both in km/s
#define MIN_VEL_COLOR 14000.0
#define PARTICLE_BRIGHTNESS 0.35//0.03 for 256/512k, 0.4 for 16k
#define PARTICLE_SHARPNESS 1.0 // Probably leave this alone
#define DOT_SIZE 8 // 15  // Range of pixels to render
#define TIME_STEP (3*32*1024) //(1*128*1024) Simulated time between integration steps, in seconds
#define STEP_COUNT 2000 // Will automatically stop running after this many steps
#define RENDER_INTERVAL 1 // How many timesteps to simulate in between each frame rendered
#define DEBUG_INFO true // Print lots of info to the console

struct vec3
{
	float x, y, z;
};

struct body
{
	vec3 position, velocity;
	float mass;
};

struct color
{
	float r, g, b;
};

#endif /* CONSTANTS_H_ */
