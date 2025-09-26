#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define NUM_THREADS 10
//when receive SIGABRT, exit safely
void handler(int sig) {
    printf("Caught signal %d (SIGABRT), cleaning up...\n", sig);
    exit(1);
}
// define a struct to store id of threads
// don't have to write struct again like (struct args_t a/ args_t a) 
typedef struct{
    int threadIdx;
} args_t;

//thread function
//use void* as generic pointer
void* worker(void* arg) {
    args_t* args = (args_t*)arg;   // 把 void* 转换成 args_t*
    printf("Hello from thread %d\n", args->threadIdx);
    return NULL;
}

int main(void) {
    signal(SIGABRT, handler);

    pthread_t threads[NUM_THREADS];
    args_t args[NUM_THREADS];

    for(int i = 0; i < NUM_THREADS; i++) {
        args[i].threadIdx = i;
    }

    // Create threads
    for(int i = NUM_THREADS - 1 ; i >= 0; i--) {
        int err = pthread_create(&threads[i], NULL, worker, &args[i]);
        assert(err == 0);
    }

     
    // Wait for threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads finished");
    return 0;

}