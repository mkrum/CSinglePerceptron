#include "neuralnetwork.h"

//This function retrieves the data from the text file, returns the length of the linked list
int getData(datap_t *point, FILE *file, int inputs){
	static size_t length = 1;
	char rawString[500];
	fscanf(file, "%s\n", rawString);
	char * divide;
	divide = strtok(rawString, ",");
	int i;
	for(i = 0; i < inputs; i++){
		char *ptr;
		point->data[i] = strtod(divide, &ptr);
		divide = strtok(NULL, ",");
	}
	strcpy(point->predict, divide);
	if (!feof(file)){
		length++;
		datap_t *new = (datap_t *)malloc(sizeof(datap_t));
		point->next = new;
		return getData(new, file, inputs);
	} else {
		return length;
	}
}
//Finds the Maximum data value at a certain index for normalizing

double findMax(datap_t *point, int i, double max){
	if (point->data[i] > max){
		max = point->data[i];
	}
	if(point->next != NULL){
		 return findMax(point->next, i, max);
	} else {
		return max;
	}
}
//Finds the Minimum data value at a certain index for normalizing

double findMin(datap_t *point, int i, double min){
	if (point->data[i] < min){
		min = point->data[i];
	}
	if(point->next != NULL){
		 return findMin(point->next, i, min);
	} else {
		return min;
	}
}
//goes through and sets the data values to the proper scale
void editPoints(datap_t *point, int i, double max, double min){
	point->data[i] = (point->data[i] - min)/(double)(max - min);
	if(point->next != NULL){
		editPoints(point->next, i, max, min);
	}
}
//converts the predict string into a useable number for testing
void setTargets(datap_t *point, int cases, char words[100][40]){
	int i = 0;
	int j, k;
	while(point->next != NULL){
		k = 0;
		for(j = 0; j < i; j++){
			if(!strcmp(point->predict, words[j]) && strlen(words[j]) > 0){
				point->target = j * (1.0/(cases - 1));
				k++;
			}
		}
		if(k == 0){
			strcpy(words[i], point->predict);
			point->target = i * (1.0/(cases - 1));
			i++;
		}
		point = point->next;
	}
}		
//Puts all the data in a range between one and zero
void normalize(datap_t *head, int inputs, int cases){
	int i;
	double max, min;
	char names[cases][40];
	setTargets(head, cases, names);
	for(i = 0; i < inputs; i++){
		max = findMax(head, i, 0);
		min = findMin(head, i, 100);
		editPoints(head, i, max, min);
	}
}
//randomizes the order of the data
void shuffleData(datap_t *head, size_t length){
	datap_t *deck[length];
	createArray(head, deck, 0);
	shuffleArray(deck, length);
}
//creates a pointer array for better shuffling	
void createArray(datap_t *point, datap_t *deck[], int i){
	deck[i] = point;
	if(point->next != NULL){
		createArray(point->next, deck, i + 1);
	}
}
//Generates a random number at the right scale for Fisher Yates
static int fyrand(int n){
	srand(time(NULL));
	int limit = RAND_MAX - RAND_MAX % n;
	int num;
	do{
		num = rand();
	}while(num >= limit);
	return num % n + 1; //To ensure that the head isnt shuffled
}

//Fisher-Yates shuffle
void shuffleArray(datap_t *deck[], size_t length){
	if(length > 1){
		size_t i;
		for(i = length - 1; i > 1; i--){
			size_t j = fyrand(i + 1);
			swap(deck[i - 1], deck[i], deck[j - 1], deck[j]);
			datap_t *temp = (datap_t *)malloc(sizeof(datap_t));
			temp = deck[j];
			deck[j] = deck[i];
			deck[i] = temp;
		}
	}
}
//Swaps two elements in the linked list
void swap(datap_t * prepoint1, datap_t * point1, datap_t *prepoint2, datap_t *point2){
	prepoint1->next = point2;
	prepoint2->next = point1;
	datap_t *temp = (datap_t *)malloc(sizeof(datap_t));
	temp = point1->next;
	point1->next = point2->next;
	point2->next = temp;
}
//computes the combined value for the data
double compute(datap_t *point, double weight[], int inputs){
	double sum = 0;
	int i;
	for(i = 0; i < inputs; i++){
			sum += weight[i] * point->data[i];
	}
	return sigmoid(sum);
}
//Seperates about 20% of the data to prevent over fitting
datap_t* seperateTestData(datap_t * head, size_t length){
	datap_t *temp1 = (datap_t *)malloc(sizeof(datap_t));
	temp1 = head;
	size_t testLength = ceil(length * .8); //set aside 20 percent of the data
	int i;
	for(i = 0; i < (testLength) - 2; i++){
		temp1 = temp1->next;
	}
	datap_t *temp2 = (datap_t *)malloc(sizeof(datap_t));
	temp2 = temp1->next;
	temp1->next = NULL;
	return temp2; 
}
//Activation function, keeps values between one and zero
double sigmoid(double input){
	return (1.0/(1 + exp(-input)));
}
//Tests the accuracy of the weights
double testWeights(datap_t *point, double weight[], int inputs, int cases){
	double out;
	int i;
	int successes = 0;
	while(point->next != NULL){
			out = compute(point, weight, inputs);
			if(fabs(out - point->target) < (1/(double)cases)){
			successes++;
		}
		point = point->next;
	}
	return successes;
}
//sets the weights to random initial values
void initializeWeights(double weights[40], int inputs){
	int i;
	for(i = 0;i < inputs;i++){
		weights[i] = (double)rand() / RAND_MAX;
	}
}
//Back propagates the error through the network
void backpropagate(datap_t * point, int inputs, double weights[]){
	double step_size = .01;
	double output = compute(point, weights, inputs); 
	int j;
	double delta = output * (1 - output) * (point->target - output);
	int i;
	for(i = 0;i < inputs; i++){
		double deltaweight = step_size * delta * point->data[i];
		weights[i] += deltaweight;
	}
}
//Backprogates throughout the entire list, learning the proper weights
void learn(datap_t *point, double weights[], int inputs){
	while(point->next != NULL){
		backpropagate(point, inputs, weights);
		point = point->next;
	}
}
//A single neural network
double network(datap_t *head,datap_t *testHead, int inputs, int length, int cases, double weights []){
	size_t successes;
	int j;
	for(j = 0; j < 2000; j++){
		learn(head, weights, inputs);
	}
	successes = testWeights(testHead, weights, inputs, cases);
	return successes/(floor(length*.2)) * 100;
}
//Finds the aggregate success value for a bunch of individual networks
double aggregateNetwork(datap_t *head, int inputs, int length, int cases, double weights[]){
	initializeWeights(weights, inputs);
	datap_t *testHead = (datap_t *)malloc(sizeof(datap_t));
	int i;
	double rateSum = 0;
	for(i = 0; i < 200; i++){
		shuffleData(head, length);
		testHead = seperateTestData(head, length);
		rateSum += network(head, testHead, inputs, length, cases, weights);
		reconnectList(head, testHead);
	}
	return rateSum/200;
}
//reconnects the testhead back to the head to return the list to origional size
void reconnectList(datap_t * point1, datap_t * point2){
	while(point1->next != NULL){
		point1 = point1->next;
	}
	point1->next = point2;
}
//Prints out the final results
void printResults(datap_t * point, double weight[], int cases, int inputs){
	double out;
	int i = 0;
	while(point->next != NULL){
		out = compute(point, weight, inputs);
		printf("%-20s : ", point->predict);
		if(fabs(out - point->target) < (1/(double)cases)){
			printf(" Success\n");
		} else {
			printf(" Failure\n");
		}
		point = point->next;
	}
}

//Michael Krumdick
