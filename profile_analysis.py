#!/usr/bin/env python3

import cProfile
import pstats
import io
import os
import tempfile
import subprocess

# Define a simple Lumpy program that exercises key parts of the interpreter
lumpy_code = """
# Test function calls and recursion
let fib = function(n) {
    if n < 2 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
};

# Test vector operations
let vector_test = function() {
    let v = [];
    for i in 100 {
        v.push(i);
    }
    
    let sum = 0;
    for x in v {
        sum = sum + x;
    }
    return sum;
};

# Test map operations
let map_test = function() {
    let m = map{};
    for i in 100 {
        m[i] = i * i;
    }
    
    let sum = 0;
    for k, v in m {
        sum = sum + v;
    }
    return sum;
};

# Run the tests
println(fib(15));
println(vector_test());
println(map_test());
"""

def run_profiled_lumpy():
    # Create a temporary file for the Lumpy code
    with tempfile.NamedTemporaryFile(suffix='.lumpy', delete=False) as f:
        f.write(lumpy_code.encode('utf-8'))
        temp_file = f.name
    
    # Set up profiling
    pr = cProfile.Profile()
    pr.enable()
    
    # Run Lumpy
    result = subprocess.run(['python3', 'lumpy.py', temp_file], 
                            capture_output=True, text=True, check=True)
    
    # Stop profiling
    pr.disable()
    
    # Clean up
    os.unlink(temp_file)
    
    # Create a StringIO object to capture the stats output
    s = io.StringIO()
    
    # Get detailed stats sorted by different metrics
    
    # By cumulative time
    print("\n=== Top Functions by Cumulative Time ===")
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    s.truncate(0)
    s.seek(0)
    
    # By total time
    print("\n=== Top Functions by Total Time ===")
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(20)
    print(s.getvalue())
    s.truncate(0)
    s.seek(0)
    
    # By call count
    print("\n=== Top Functions by Call Count ===")
    ps = pstats.Stats(pr, stream=s).sort_stats('calls')
    ps.print_stats(20)
    print(s.getvalue())
    s.truncate(0)
    s.seek(0)
    
    # Analyze specific function families
    print("\n=== Analysis of Key Function Families ===")
    
    # Value operations (copy, __eq__, etc.)
    print("\n-- Value Operations --")
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats('copy', 'cow', '__eq__', '__hash__')
    print(s.getvalue())
    s.truncate(0)
    s.seek(0)
    
    # AST evaluation
    print("\n-- AST Evaluation --")
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats('eval')
    print(s.getvalue())
    s.truncate(0)
    s.seek(0)
    
    # Environment and variable lookup
    print("\n-- Environment Operations --")
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats('get', 'lookup')
    print(s.getvalue())
    
    return result.stdout, pr

if __name__ == "__main__":
    print("Running profiled Lumpy code...")
    output, profile = run_profiled_lumpy()
    print("\nLumpy program output:")
    print(output)
    
    print("\n=== Performance Optimization Suggestions ===")
    print("Based on the profiling results above, we can identify potential optimization areas.")
    print("Run this script to see the actual profile data and make informed optimization decisions.")