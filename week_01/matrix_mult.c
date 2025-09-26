#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

// Matrix structure
typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

// Thread parameters
typedef struct {
    Matrix *A;
    Matrix *B;
    Matrix *C;
    int row_start;
    int row_end;
} ThreadArg;

// Allocate matrix
Matrix *alloc_matrix(int rows, int cols) {
    Matrix *M = malloc(sizeof(Matrix));
    M->rows = rows;
    M->cols = cols;
    M->data = calloc(rows * cols, sizeof(double));
    return M;
}

// Free matrix
void free_matrix(Matrix *M) {
    free(M->data);
    free(M);
}

// Single-threaded matrix multiplication
void matmul_single(Matrix *A, Matrix *B, Matrix *C) {
    int n = A->rows, m = A->cols, p = B->cols;
    double *a = A->data, *b = B->data, *c = C->data;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += a[i * m + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }
}

// Multi-threaded worker function
void *matmul_worker(void *arg) {
    ThreadArg *t = (ThreadArg *)arg;
    Matrix *A = t->A, *B = t->B, *C = t->C;
    int n = A->rows, m = A->cols, p = B->cols;
    double *a = A->data, *b = B->data, *c = C->data;

    for (int i = t->row_start; i < t->row_end; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += a[i * m + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }
    return NULL;
}

// Multi-threaded matrix multiplication
void matmul_multi(Matrix *A, Matrix *B, Matrix *C, int num_threads) {
    pthread_t threads[num_threads];
    ThreadArg args[num_threads];

    int rows_per_thread = A->rows / num_threads;
    int extra = A->rows % num_threads;
    int current = 0;

    for (int t = 0; t < num_threads; t++) {
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].row_start = current;
        args[t].row_end = current + rows_per_thread + (t < extra ? 1 : 0);
        current = args[t].row_end;
        pthread_create(&threads[t], NULL, matmul_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

// Get high precision time
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Verify results
int verify_results(Matrix *C1, Matrix *C2) {
    int n = C1->rows, p = C1->cols;
    for (int i = 0; i < n * p; i++) {
        double diff = C1->data[i] - C2->data[i];
        if (diff > 1e-10 || diff < -1e-10) return 0;
    }
    return 1;
}

// Print matrix
void print_matrix(Matrix *M) {
    for (int i = 0; i < M->rows; i++) {
        for (int j = 0; j < M->cols; j++) {
            printf("%6.2f ", M->data[i * M->cols + j]);
        }
        printf("\n");
    }
}

// Small matrix test
void test_small() {
    int cases[][4] = {
        {1,1,1,1}, // A=1x1, B=1x1
        {1,1,1,5}, // A=1x1, B=1x5
        {2,1,1,3}, // A=2x1, B=1x3
        {2,2,2,2}  // A=2x2, B=2x2
    };
    int num_cases = sizeof(cases)/sizeof(cases[0]);

    for (int t = 0; t < num_cases; t++) {
        int a_rows = cases[t][0], a_cols = cases[t][1];
        int b_rows = cases[t][2], b_cols = cases[t][3];

        Matrix *A = alloc_matrix(a_rows, a_cols);
        Matrix *B = alloc_matrix(b_rows, b_cols);
        Matrix *C1 = alloc_matrix(a_rows, b_cols);
        Matrix *C2 = alloc_matrix(a_rows, b_cols);

        // Fill A and B with test data
        for (int i = 0; i < a_rows * a_cols; i++) A->data[i] = i+1;
        for (int i = 0; i < b_rows * b_cols; i++) B->data[i] = i+1;

        matmul_single(A, B, C1);
        matmul_multi(A, B, C2, 2);

        printf("Case %d: %dx%d * %dx%d, Correct=%s\n",
               t+1, a_rows, a_cols, b_rows, b_cols,
               verify_results(C1, C2) ? "✓" : "✗");

        free_matrix(A); free_matrix(B);
        free_matrix(C1); free_matrix(C2);
    }
}

// Performance benchmark
void benchmark(int n, int threads) {
    Matrix *A = alloc_matrix(n, n);
    Matrix *B = alloc_matrix(n, n);
    Matrix *C1 = alloc_matrix(n, n);
    Matrix *C2 = alloc_matrix(n, n);

    for (int i = 0; i < n * n; i++) {
        A->data[i] = (rand() % 100) / 10.0;
        B->data[i] = (rand() % 100) / 10.0;
    }

    double t1 = get_time();
    matmul_single(A, B, C1);
    double single_time = get_time() - t1;

    t1 = get_time();
    matmul_multi(A, B, C2, threads);
    double multi_time = get_time() - t1;

    int correct = verify_results(C1, C2);
    double speedup = single_time / multi_time;

    printf("N=%d, Threads=%d, Single=%.3fs, Multi=%.3fs, Speedup=%.2f, Correct=%s\n",
           n, threads, single_time, multi_time, speedup, correct ? "✓" : "✗");

    free_matrix(A); free_matrix(B); free_matrix(C1); free_matrix(C2);
}

int main() {
    printf("=== Matrix Multiplication with pthreads ===\n");
    printf("CPU cores available: %ld\n\n", sysconf(_SC_NPROCESSORS_ONLN));

    // Small matrix tests
    printf("=== Small Matrix Tests ===\n");
    test_small();
    printf("\n");

    // Large matrix performance tests
    int sizes[] = {1000, 2000};
    int num_sizes = sizeof(sizes)/sizeof(sizes[0]);
    int threads_list[] = {1, 4, 16, 32, 64, 128};
    int num_threads = sizeof(threads_list)/sizeof(threads_list[0]);

    srand(42);

    printf("=== Performance Test ===\n");
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        printf("Matrix Size: %dx%d\n", n, n);
        for (int t = 0; t < num_threads; t++) {
            benchmark(n, threads_list[t]);
        }
        printf("\n");
    }

    return 0;
}