#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include "Constants.h"
#include <hip/hip_runtime.h>

void initializeBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass);
void runSimulation(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass, char* image, float* hdImage);
__global__ void interactBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass);
float magnitude(vec3 v);
void renderClear(char* image, float* hdImage);
__global__ void GPUrenderBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass, float* hdImage);
float clamp(float x);
void writeRender(char* data, float* hdImage, int step);

int main()
{
	std::cout << SYSTEM_THICKNESS << "AU thick disk\n";;
	char *image = new char[WIDTH*HEIGHT*3];
	float *hdImage = new float[WIDTH*HEIGHT*3];
	//struct body *bodies = new struct body[NUM_BODIES];
	
	float* xpos = new float[NUM_BODIES];
	float* ypos = new float[NUM_BODIES];
	float* zpos = new float[NUM_BODIES];
	float* xvel = new float[NUM_BODIES];
	float* yvel = new float[NUM_BODIES];
	float* zvel = new float[NUM_BODIES];
	float* mass = new float[NUM_BODIES];
	initializeBodies(xpos,ypos,zpos,xvel,yvel,zvel,mass);
	runSimulation(xpos,ypos,zpos,xvel,yvel,zvel,mass, image, hdImage);
	std::cout << "\nwe made it\n";
	delete[] image;
	return 0;
}

void initializeBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass)
{
	using std::uniform_real_distribution;
	uniform_real_distribution<float> randAngle (0.0, 200.0*PI);
	uniform_real_distribution<float> randRadius (INNER_BOUND, SYSTEM_SIZE);
	uniform_real_distribution<float> randHeight (0.0, SYSTEM_THICKNESS);
	std::default_random_engine gen (0);
	float angle;
	float radius;
	float velocity;

	//STARS
	velocity = 0.67*sqrt((G*SOLAR_MASS)/(4*BINARY_SEPARATION*TO_METERS));
	//STAR 1
	xpos[0] = 0.0;///-BINARY_SEPARATION;
	ypos[0] = 0.0;
	zpos[0] = 0.0;
	xvel[0] = 0.0;
	yvel[0] = 0.0;//velocity;
	zvel[0] = 0.0;
	mass[0] = SOLAR_MASS;

	    ///STARTS AT NUMBER OF STARS///
	float totalExtraMass = 0.0;
	for (int i=1; i<NUM_BODIES; i++)
	{
		angle = randAngle(gen);
		radius = sqrt(SYSTEM_SIZE)*sqrt(randRadius(gen));
		velocity = pow(((G*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
					  	  	  	  	  / (radius*TO_METERS)), 0.5);
		xpos[i] =  radius*cos(angle);
		ypos[i] =  radius*sin(angle);
		zpos[i] =  randHeight(gen)-SYSTEM_THICKNESS/2;
		xvel[i] =  velocity*sin(angle);
		yvel[i] = -velocity*cos(angle);
		zvel[i] =  0.0;
		mass[i] = (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
		totalExtraMass += (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
	}
	std::cout << "\nTotal Disk Mass: " << totalExtraMass;
	std::cout << "\nEach Particle weight: " << (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES
			  << "\n______________________________\n";
}

void runSimulation(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass, char* image, float* hdImage)
{
	float *d_xpos;	float *d_ypos;	float *d_zpos;	float *d_xvel;	float *d_yvel;	float *d_zvel;	float *d_mass;	char *d_image;	float *d_hdImage;
	
	hipMalloc(&d_xpos,NUM_BODIES*sizeof(float));
	hipMalloc(&d_ypos,NUM_BODIES*sizeof(float));
	hipMalloc(&d_zpos,NUM_BODIES*sizeof(float));
	hipMalloc(&d_xvel,NUM_BODIES*sizeof(float));
	hipMalloc(&d_yvel,NUM_BODIES*sizeof(float));
	hipMalloc(&d_zvel,NUM_BODIES*sizeof(float));
	hipMalloc(&d_mass,NUM_BODIES*sizeof(float));
	hipMalloc(&d_image,WIDTH*HEIGHT*3*sizeof(char));
	hipMalloc(&d_hdImage,WIDTH*HEIGHT*3*sizeof(float));
	int nBlocks=(NUM_BODIES+1024-1)/1024;

	//createFirstFrame
	renderClear(image, hdImage);
	hipMemcpy(d_hdImage,hdImage,WIDTH*HEIGHT*3*sizeof(float),hipMemcpyHostToDevice);
	hipLaunchKernelGGL(GPUrenderBodies, dim3(nBlocks+1), dim3(1024), 0, 0, d_xpos,d_ypos,d_zpos,d_xvel,d_yvel,d_zvel,d_mass,d_hdImage);
	hipMemcpy(hdImage,d_hdImage,WIDTH*HEIGHT*3*sizeof(float),hipMemcpyDeviceToHost);
	writeRender(image, hdImage, 1);
	
	for (int step=1; step<STEP_COUNT; step++)
	{
		std::cout << "\nBeginning timestep: " << step;
		printf("\nStartK\n");
		hipMemcpy(d_xpos, xpos, NUM_BODIES*sizeof(float), hipMemcpyHostToDevice);
		hipMemcpy(d_ypos, ypos, NUM_BODIES*sizeof(float), hipMemcpyHostToDevice);
		hipMemcpy(d_zpos, zpos, NUM_BODIES*sizeof(float), hipMemcpyHostToDevice);
		hipMemcpy(d_xvel, xvel, NUM_BODIES*sizeof(float), hipMemcpyHostToDevice);
		hipMemcpy(d_yvel, yvel, NUM_BODIES*sizeof(float), hipMemcpyHostToDevice);
		hipMemcpy(d_zvel, zvel, NUM_BODIES*sizeof(float), hipMemcpyHostToDevice);
		hipMemcpy(d_mass, mass, NUM_BODIES*sizeof(float), hipMemcpyHostToDevice);
		printf("StartK1\n");	
		hipLaunchKernelGGL(interactBodies, dim3(nBlocks), dim3(1024), 0, 0, d_xpos,d_ypos,d_zpos,d_xvel,d_yvel,d_zvel,d_mass);
		hipDeviceSynchronize();
		printf("EndK\n");
		hipMemcpy( xpos,d_xpos, NUM_BODIES*sizeof(float), hipMemcpyDeviceToHost);
		hipMemcpy( ypos, d_ypos,NUM_BODIES*sizeof(float), hipMemcpyDeviceToHost);
		hipMemcpy( zpos,d_zpos, NUM_BODIES*sizeof(float), hipMemcpyDeviceToHost);
		hipMemcpy( xvel,d_xvel, NUM_BODIES*sizeof(float), hipMemcpyDeviceToHost);
		hipMemcpy( yvel,d_yvel, NUM_BODIES*sizeof(float), hipMemcpyDeviceToHost);
		hipMemcpy( zvel,d_zvel, NUM_BODIES*sizeof(float), hipMemcpyDeviceToHost);
		hipMemcpy( mass, d_mass,NUM_BODIES*sizeof(float), hipMemcpyDeviceToHost);
		//printf("EndK2\n");

		if (step%RENDER_INTERVAL==0)
		{
			std::cout << "\nWriting frame " << step;
			if (DEBUG_INFO)	{std::cout << "\nClearing Pixels..." << std::flush;}
			renderClear(image, hdImage);
			if (DEBUG_INFO) {std::cout << "\nRendering Particles..." << std::flush;}
			//renderBodies(pos, vel, hdImage);
			hipMemcpy(d_hdImage,hdImage,WIDTH*HEIGHT*3*sizeof(float),hipMemcpyHostToDevice);
			hipLaunchKernelGGL(GPUrenderBodies, dim3(nBlocks+1), dim3(1024), 0, 0, d_xpos,d_ypos,d_zpos,d_xvel,d_yvel,d_zvel,d_mass,d_hdImage);
			hipDeviceSynchronize();
			hipMemcpy(hdImage,d_hdImage,WIDTH*HEIGHT*3*sizeof(float),hipMemcpyDeviceToHost);
			if (DEBUG_INFO) {std::cout << "\nWriting frame to file..." << std::flush;}
			writeRender(image, hdImage, step);
		}
		if (DEBUG_INFO) {std::cout << "\n-------Done------- timestep: "
			       << step << "\n" << std::flush;}
	}
}
__global__ void interactBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < NUM_BODIES)
	{		
		float Fx=0.0f; float Fy=0.0f; float Fz=0.0f;
		float xposi=xpos[i];
		float yposi=ypos[i];
		float zposi=zpos[i];
		#pragma unroll
		for(int j=0; j < NUM_BODIES; j++)
		{
			if(i!=j)
			{ 
				vec3 posDiff;
				posDiff.x = (xposi-xpos[j])*TO_METERS;
				posDiff.y = (yposi-ypos[j])*TO_METERS;
				posDiff.z = (zposi-zpos[j])*TO_METERS;
				float dist = sqrt(posDiff.x*posDiff.x+posDiff.y*posDiff.y+posDiff.z*posDiff.z);
				float F = TIME_STEP*(G*mass[i]*mass[j]) / ((dist*dist + SOFTENING*SOFTENING) * dist);
				//float Fa = F/mass[i];
				Fx-=F*posDiff.x;
				Fy-=F*posDiff.y;
				Fz-=F*posDiff.z;
			}	
		}
		xvel[i] += Fx/mass[i];
		yvel[i] += Fy/mass[i];
		zvel[i] += Fz/mass[i];
		xpos[i] += TIME_STEP*xvel[i]/TO_METERS;
		ypos[i] += TIME_STEP*yvel[i]/TO_METERS;
		zpos[i] += TIME_STEP*zvel[i]/TO_METERS;
	}
}

float magnitude(vec3 v)
{
	return sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
}

void renderClear(char* image, float* hdImage)
{
	for (int i=0; i<WIDTH*HEIGHT*3; i++)
	{
		image[i] = 0; //char(image[i]/1.2);
		hdImage[i] = 0.0;
	}
}

__global__ void GPUrenderBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass, float* hdImage)
{
	/// ORTHOGONAL PROJECTION
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	float velocityMax = MAX_VEL_COLOR; //35000
	float velocityMin = sqrt(0.8*(G*(SOLAR_MASS+EXTRA_MASS*SOLAR_MASS))/
				(SYSTEM_SIZE*TO_METERS)); //MIN_VEL_COLOR;
	if(i<NUM_BODIES)
	{
		float vxsqr=xvel[i]*xvel[i];
		float vysqr=yvel[i]*yvel[i];
		float vzsqr=zvel[i]*zvel[i];
		float vMag = sqrt(vxsqr+vysqr+vzsqr);
		int x = (WIDTH/2.0)*(1.0+xpos[i]/(SYSTEM_SIZE*RENDER_SCALE));
		int y = (HEIGHT/2.0)*(1.0+ypos[i]/(SYSTEM_SIZE*RENDER_SCALE));

		if (x>DOT_SIZE && x<WIDTH-DOT_SIZE && y>DOT_SIZE && y<HEIGHT-DOT_SIZE)
		{
			float vPortion = sqrt((vMag-velocityMin) / velocityMax);
			float xPixel = (WIDTH/2.0)*(1.0+xpos[i]/(SYSTEM_SIZE*RENDER_SCALE));
			float yPixel = (HEIGHT/2.0)*(1.0+ypos[i]/(SYSTEM_SIZE*RENDER_SCALE));
			float xP = floor(xPixel);
			float yP = floor(yPixel);
			color c;
			c.r = max(min(4*(vPortion-0.333),1.0),0.0);
                        c.g = max(min(min(4*vPortion,4.0*(1.0-vPortion)),1.0),0.0);
                        c.b = max(min(4*(0.5-vPortion),1.0),0.0);
			for (int a=-DOT_SIZE/2; a<DOT_SIZE/2; a++)
			{
				for (int b=-DOT_SIZE/2; b<DOT_SIZE/2; b++)
				{
					float cFactor = PARTICLE_BRIGHTNESS /(pow(exp(pow(PARTICLE_SHARPNESS*(xP+a-xPixel),2.0)) + exp(pow(PARTICLE_SHARPNESS*(yP+b-yPixel),2.0)),/*1.25*/0.75)+1.0);
					//colorAt(int(xP+a),int(yP+b),c, cFactor, hdImage);
					int pix = 3*(xP+a+WIDTH*(yP+b));
					hdImage[pix+0] += c.r*cFactor;
					hdImage[pix+1] += c.g*cFactor;
					hdImage[pix+2] += c.b*cFactor;
				}
			}
		}
	}
}

float clamp(float x)
{
	return max(min(x,1.0),0.0);
}

void writeRender(char* data, float* hdImage, int step)
{
	
	for (int i=0; i<WIDTH*HEIGHT*3; i++)
	{
		data[i] = int(255.0*clamp(hdImage[i]));
	}

	int frame = step/RENDER_INTERVAL + 1;//RENDER_INTERVAL;
	std::string name = "images/Step"; 
	int i = 0;
	if (frame == 1000) i++; // Evil hack to avoid extra 0 at 1000
	for (i; i<4-floor(log(frame)/log(10)); i++)
	{
		name.append("0");
	}
	name.append(std::to_string(frame));
	name.append(".ppm");

	std::ofstream file (name, std::ofstream::binary);

	if (file.is_open())
	{
//		size = file.tellg();
		file << "P6\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";
		file.write(data, WIDTH*HEIGHT*3);
		file.close();
	}

}

