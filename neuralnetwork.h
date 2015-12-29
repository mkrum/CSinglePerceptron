#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 


typedef struct datapoint{
	double data[40];
	char predict[40];
	double target;
	struct datapoint *next;
} datap_t;

int getData(datap_t *, FILE *, int);
void printData(datap_t *, int);
double findMax(datap_t *, int, double);
double findMin(datap_t *, int, double);
void editPoints(datap_t *, int, double, double);
void normalize(datap_t *, int, int);
void setTargets(datap_t *, int, char [5][40]);
void seperate(datap_t *, int);
void shuffleData(datap_t *, size_t);
void shuffleArray(datap_t *[], size_t);
void createArray(datap_t *, datap_t *[], int);
datap_t* seperateTestData(datap_t *, size_t);
void swap(datap_t * prepoint1, datap_t * point1, datap_t *prepoint2, datap_t *point2);
double sigmoid(double);
double testWeights(datap_t *, double[], int, int);
void initializeWeights(double [], int);
double error(double, double);
double compute(datap_t *, double [], int);
void backpropagate(datap_t *, int, double[]);
void learn(datap_t *, double [], int);
double network(datap_t *, datap_t *, int, int, int, double []);
double aggregateNetwork(datap_t *, int, int, int, double []);
void reconnectList(datap_t *, datap_t *);
void printResults(datap_t *, double [], int, int);

//Michael Krumdick
