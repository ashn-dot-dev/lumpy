#!/usr/bin/env python3

import time
import subprocess
import gc
import os
import tempfile

# Fibonacci implementations
def python_fib(n):
    if n < 2:
        return n
    return python_fib(n-1) + python_fib(n-2)

# Write Lumpy version to a temporary file
lumpy_code = """
let fib = function(n) {
    if n < 2 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
};

println(fib(20));
"""

# Create C version for comparison
c_code = """
#include <stdio.h>

int fib(int n) {
    if (n < 2) {
        return n;
    }
    return fib(n-1) + fib(n-2);
}

int main() {
    printf("%d\\n", fib(20));
    return 0;
}
"""

def time_function(func, *args):
    gc.collect()
    start = time.time()
    result = func(*args)
    end = time.time()
    return (end - start), result

def run_lumpy_benchmark():
    with tempfile.NamedTemporaryFile(suffix='.lumpy', delete=False) as f:
        f.write(lumpy_code.encode('utf-8'))
        temp_file = f.name
    
    gc.collect()
    start = time.time()
    result = subprocess.run(['python3', 'lumpy.py', temp_file], 
                            capture_output=True, text=True, check=True)
    end = time.time()
    
    os.unlink(temp_file)
    return end - start, int(result.stdout.strip())

def run_c_benchmark():
    with tempfile.NamedTemporaryFile(suffix='.c', delete=False) as f:
        f.write(c_code.encode('utf-8'))
        c_file = f.name
    
    exe_file = c_file + '.out'
    subprocess.run(['gcc', '-O3', c_file, '-o', exe_file], check=True)
    
    gc.collect()
    start = time.time()
    result = subprocess.run([exe_file], capture_output=True, text=True, check=True)
    end = time.time()
    
    os.unlink(c_file)
    os.unlink(exe_file)
    return end - start, int(result.stdout.strip())

def main():
    # Number of test runs
    runs = 5
    n = 20  # Fibonacci number to calculate
    
    print("=== Fibonacci Performance Comparison (n=20) ===")
    
    # Python benchmark
    py_times = []
    for _ in range(runs):
        elapsed, result = time_function(python_fib, n)
        py_times.append(elapsed)
    avg_py_time = sum(py_times) / len(py_times)
    print(f"Python:  {avg_py_time:.6f} seconds")
    
    # Lumpy benchmark
    lumpy_times = []
    for _ in range(runs):
        elapsed, result = run_lumpy_benchmark()
        lumpy_times.append(elapsed)
    avg_lumpy_time = sum(lumpy_times) / len(lumpy_times)
    print(f"Lumpy:   {avg_lumpy_time:.6f} seconds")
    
    # C benchmark
    try:
        c_times = []
        for _ in range(runs):
            elapsed, result = run_c_benchmark()
            c_times.append(elapsed)
        avg_c_time = sum(c_times) / len(c_times)
        print(f"C:       {avg_c_time:.6f} seconds")
        
        # Calculate ratios
        print("\n=== Performance Ratios ===")
        print(f"Lumpy is {avg_lumpy_time/avg_py_time:.2f}x slower than Python")
        print(f"Lumpy is {avg_lumpy_time/avg_c_time:.2f}x slower than C")
        print(f"Python is {avg_py_time/avg_c_time:.2f}x slower than C")
    except Exception as e:
        print(f"C benchmark failed: {e}")
        print("\n=== Performance Ratios ===")
        print(f"Lumpy is {avg_lumpy_time/avg_py_time:.2f}x slower than Python")

if __name__ == "__main__":
    main()