#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"

// #define IMAGES_USED_FOR_TRAINING (int)1000
#define IMAGES_USED_FOR_TRAINING (int)10000
#define IMAGES_USED_FOR_TESTING (int)3000

// Model parameters
#define INPUT_NODES (int) 784
#define HIDDEN_NODES (int) 300
#define OUTPUT_NODES (int) 10
#define LEARNING_RATE (float)0.1f


// Determine save path per proc size
char* get_disk_path(int number_of_threads){	
	char str[4];
	sprintf(str, "%d", number_of_threads);
	char *path = malloc(21 * sizeof(char));
	strcpy(path, "testing_net/proc_");
	strcat(path, str);
	strcat(path, "/");
	return path;                
}


// Calculates wall-time in seconds
double wtime( void )
{
    double wtime;
    struct timespec tstruct;
    clock_gettime(CLOCK_MONOTONIC, &tstruct);
    wtime = (double) tstruct.tv_nsec/1.0e+9;
    wtime += (double) tstruct.tv_sec;
    return wtime;
}


// Trains the model of a set of images
void training(int num_of_threads){

	// Get unique path based on num_of_threads
	char* path = get_disk_path(num_of_threads);
	
	// Load training data and create model 
	Img** imgs = csv_to_imgs("./data/mnist_train.csv", IMAGES_USED_FOR_TRAINING);
	NeuralNetwork* net = network_create(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE);

	// Train network and save learned weights to disk
	network_train_batch_imgs(net, imgs, IMAGES_USED_FOR_TRAINING);
	network_save(net, path);
}


// Tests the trained model and reports accuracy
void testing(int num_of_threads){

	char* path = get_disk_path(num_of_threads);

	// Load testing data and learned model parameters
	Img** imgs = csv_to_imgs("data/mnist_test.csv", IMAGES_USED_FOR_TESTING);
	NeuralNetwork* net = network_load(path);

	// Score model accuracy
	double score = network_predict_imgs(net, imgs, 1000);
	printf("Score: %1.5f\n", score);
	
	// Free memory
	imgs_free(imgs, IMAGES_USED_FOR_TESTING);
	network_free(net);
}


// Accepts number of threads to use for parallelization
// Trains and tests neural network on MNIST data 
int main(int argc, char *argv[]) {

	int num_of_threads;
	double elapsed_train, elapsed_test;

	// Check if only one arg is provided
    if (argc != 2) {  
        printf("Usage: %s <integer>\n", argv[0]);  // Print usage if not correct
        return 1;
    }

	// Set number of threads to be used from profiling batch script
	num_of_threads = atoi(argv[1]);
	printf("passed in num_of_threads: %d\n", num_of_threads);

	srand(time(NULL));

#pragma omp single
	omp_set_num_threads(num_of_threads);
	
	// // TRAIN NETWORK - Learned network values saved to disk
	// elapsed_train = wtime();
	// training(num_of_threads);
	// printf("Training Network - Time elapsed = %g seconds.\n", wtime() - elapsed_train);

	// TEST NETWORK - Use learned network values from disk to compute accuracy of model
	elapsed_test = wtime();
	testing(num_of_threads);
	printf("Testing Network - Time elapsed = %g seconds.\n", wtime() - elapsed_test);

#pragma omp single
		printf("omp_get_num_threads: %d\n", omp_get_num_threads());

	return 0;
}