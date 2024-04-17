#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"


#define IMAGES_USED_FOR_TRAINING (int)1000
// #define IMAGES_USED_FOR_TRAINING (int)10000
#define IMAGES_USED_FOR_TESTING (int)3000

// Model parameters
#define INPUT_NODES (int) 784
#define HIDDEN_NODES (int) 300
#define OUTPUT_NODES (int) 10
#define LEARNING_RATE (float)0.1f

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
void training(void){

	// Load training data and create model 
	Img** imgs = csv_to_imgs("./data/mnist_test.csv", IMAGES_USED_FOR_TRAINING);
	NeuralNetwork* net = network_create(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE);

	// Free memory
	network_train_batch_imgs(net, imgs, IMAGES_USED_FOR_TRAINING);
	network_save(net, "testing_net");
}


// Tests the trained model and reports accuracy
void testing(void){

	// Load testing data and learned model parameters
	Img** imgs = csv_to_imgs("data/mnist_test.csv", IMAGES_USED_FOR_TESTING);
	NeuralNetwork* net = network_load("testing_net");
	
	// Score model accuracy
	double score = network_predict_imgs(net, imgs, 1000);
	printf("Score: %1.5f\n", score);
	
	// Free memory
	imgs_free(imgs, IMAGES_USED_FOR_TESTING);
	network_free(net);
}

int main() {
	srand(time(NULL));

	// TRAIN NETWORK
	elapsed_train = wtime();
	training();
	printf("Training Network - Time elapsed = %g seconds.\n", wtime() - elapsed_train);

	// TEST NETWORK
	elapsed_test = wtime();
	testing();
	printf("Testing Network - Time elapsed = %g seconds.\n", wtime() - elapsed_test);

	return 0;
}