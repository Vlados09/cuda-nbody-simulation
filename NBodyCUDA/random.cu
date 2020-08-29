#include <stdlib.h>
#include "random.h"

void init_random() {
	srand(RAND_SEED);
}

float random_unit_float() {
	return ((float)(rand() % RAND_MAX) / (float)RAND_MAX);
}
