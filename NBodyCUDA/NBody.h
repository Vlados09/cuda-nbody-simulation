//Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __NBODY_HEADER__
#define __NBODY_HEADER__

// Simulation constants:
#define G			9.8f				//gravitational constant
#define dt			0.01f				//time step
#define dt_x_G      G * dt              // shortcut calculation
#define SOFTENING	2.0f				//softening parameter to help with numerical instability
#define SOFTENING2  SOFTENING*SOFTENING //softening squared

// Input parameters: 
#define VALUE_SIZE  32					//max input length for a single value
#define NUM_VALUES  5					//number of values required to represent a body
#define MIN_ARGS    4					//min number of arguments required to run
#define LINE_SIZE VALUE_SIZE*NUM_VALUES //max input line size in the file

// Determines if the coordinate is within activity map size
#define between_1(x, y) (x < 1 && x >= 0 && y < 1 && y >= 0)

struct nbody{
	float x, y, vx, vy, m;
};

struct nbody_soa {
	float* x, * y, * vx, * vy, * m;
};

typedef enum MODE { CPU, OPENMP, CUDA } MODE;
typedef struct nbody nbody;
typedef struct nbody_soa nbody_soa;


/** nbodyCUDA/nbodyCPU
	* Performs the full nbody simulation on GPU/CPU either over I itterations or using visualiser
	*/
void nbodyCUDA(void);
void nbodyCPU(void);

/** stepCUDA/stepCPU
	* Perform main simulation step using Newtons formula on a GPU/CPU.
	*/
void stepCUDA(void);
void stepCPU(void);

/** calc_activity
	* Calculates activity map densities going through each square and checking how many bodies are in it.
	*/
void calc_activity(void);

/** parse_args
    * Parses commnad line arguments
	* @param argc  number of arguments
	* @param agrv  a pointer to command line arguments string
	*/
void parse_args(int argc, char *argv[]);

/** calc_activity
	* Reads a csv file with nbodies information.
	* @param f	a pointer to a csv file that should be read
	*/
void read_csv();

/** calc_activity
	* Reads a single line in a csv file transfering information about a single body into appropriate data structure
	* @param line  a pointer to an array of characters representing the line to be read
	* @param b a pointer to the body where to store information from the line
	*/
void read_line(char* line, nbody *b);


/** nbody_AOS2SOA
	* Converts nbody structure from Array of Structures (AOS) to Structure of Arrays (SOA) 
	*/
void nbody_AOS2SOA();

/** swap_arrays/bodies
	* Swapps two arrays/body strcutures
	*/
void swap_arrays(float** arr1, float** arr2);
void swap_bodies(nbody_soa** body1, nbody_soa** body2);

/** calc_activity
	* Fills out the bodies structure with random data
	* x and y = random between [0, 1]
	* vx and vy = 0
	* m = 1/N
	*/
void random_bodies(void);

/** calc_activity
	* Prints activity values in a structured format (used for debuging)
	*/
void print_activity();

/** calc_activity
	* Prints all the information about a given body
	* @param b pointer to a body structure
	*/
void print_body(nbody *b);

/** calc_activity
	* Print help information about the command line inputs and flags
	*/
void print_help();

#endif	//__NBODY_HEADER__