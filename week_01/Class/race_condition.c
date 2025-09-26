#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define NUM_THREADS 4
#define INCREMENTS 1000000

long long counter = 0;


// guaranteed no race conditions if they don't write the same location 

//thread function
//use void* as generic pointer
void* worker(void* arg) {
    for (int i = 0; i < INCREMENTS; i++) {
        counter++; //this will create a race condition
    }
    return NULL;
}



// The Goal of parallel Programming is to split work concurrently without introducing race condition
int main(void) {

    pthread_t threads[NUM_THREADS];

    // Create threads
    for(int i = 0 ; i < NUM_THREADS; i--) {
        int err = pthread_create(&threads[i], NULL, worker, NULL);
        assert(err == 0);
    }

    // Wait for threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Expected = %d, Actual = %lld\n\n", NUM_THREADS * INCREMENTS, counter);
    printf("All threads finished");
    return 0;

}