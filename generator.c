#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define NUM_RECORDS 32768

float marks[NUM_RECORDS];

int main(void) {
    
	FILE *f = NULL;
	f = fopen("marks.dat", "wb"); //write flag
	if (f == NULL){
		fprintf(stderr, "Error: Could not create marks.dat file \n");
		exit(1);
	}

	//Create data
	float highest = 0;
	for (int i = 0; i < NUM_RECORDS; i++){
		marks[i] = rand() / (float) RAND_MAX * 100.0f;
		if (marks[i] > highest){
			highest = marks[i];
		}
	}

	//read student data
	if (fwrite(marks, sizeof(float), NUM_RECORDS, f) != NUM_RECORDS){
		fprintf(stderr, "Error: Not all marks written!\n");
		exit(1);
	}
	fclose(f);

	printf("Highest mark is %f\n", highest);
    
	return 0;
}


