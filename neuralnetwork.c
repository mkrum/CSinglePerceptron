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
	length = getData(head, file, inputs);
	normalize(head, inputs, cases);
	aggregateNetwork(head, inputs, length, cases);
}

