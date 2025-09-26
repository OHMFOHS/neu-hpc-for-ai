# Matrix Multiplication Performance Analysis Results

This project implements and analyzes both single-threaded and multi-threaded matrix multiplication algorithms in C using pthreads. The implementation includes comprehensive testing, performance benchmarking.

## System Specifications

- **CPU Cores**: 10
- **Operating System**: macOS (Darwin 25.0.0)
- **Compiler**: GCC with -O2 optimization
- **Threading Library**: POSIX Threads (pthread)


### Test Matrix Sizes
- 1000×1000 matrices
- 2000×2000 matrices  

### Performance Data:
=== Small Matrix Tests ===
Case 1: 1x1 * 1x1, Correct=✓
Case 2: 1x1 * 1x5, Correct=✓
Case 3: 2x1 * 1x3, Correct=✓
Case 4: 2x2 * 2x2, Correct=✓

=== Performance Test ===
Matrix Size: 1000x1000
N=1000, Threads=1, Single=1.199s, Multi=1.167s, Speedup=1.03, Correct=✓
N=1000, Threads=4, Single=1.167s, Multi=0.308s, Speedup=3.79, Correct=✓
N=1000, Threads=16, Single=1.159s, Multi=0.145s, Speedup=7.97, Correct=✓
N=1000, Threads=32, Single=1.195s, Multi=0.142s, Speedup=8.40, Correct=✓
N=1000, Threads=64, Single=1.163s, Multi=0.142s, Speedup=8.18, Correct=✓
N=1000, Threads=128, Single=1.171s, Multi=0.141s, Speedup=8.29, Correct=✓

Matrix Size: 2000x2000
N=2000, Threads=1, Single=10.084s, Multi=10.476s, Speedup=0.96, Correct=✓
N=2000, Threads=4, Single=10.351s, Multi=2.637s, Speedup=3.93, Correct=✓
N=2000, Threads=16, Single=10.318s, Multi=1.739s, Speedup=5.93, Correct=✓
N=2000, Threads=32, Single=10.490s, Multi=1.444s, Speedup=7.27, Correct=✓
N=2000, Threads=64, Single=10.733s, Multi=1.699s, Speedup=6.32, Correct=✓
N=2000, Threads=128, Single=10.458s, Multi=1.630s, Speedup=6.42, Correct=✓


The results show that with proper optimization techniques, multi-threaded matrix multiplication can achieve significant performance improvements while maintaining correctness and code quality.
