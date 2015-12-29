#include "neuralnetwork.h"

int main(int argc, char *argv[]) {
	if(argc < 2){
		printf("No input file detected.");
		return 0;
	}
	datap_t *head = (datap_t *)malloc(sizeof(datap_t));
	FILE *file = fopen(argv[argc-1], "r");
	int inputs, cases;
	size_t length;
	fscanf(file, "%d %d", &inputs, &cases);
	//Get the Data
	length = getData(head, file, inputs);
	printf("%zu", length);
	//Clean the Data
	normalize(head, inputs, cases);
	//Analyze the Data
	double weights[inputs];
	double rate = aggregateNetwork(head, inputs, length, cases, weights);
	printResults(head, weights, cases, inputs);
	printf("Aggregate Classification Sucess Rate: %5.3lf%%", rate);
}

//Michael Krumdick
