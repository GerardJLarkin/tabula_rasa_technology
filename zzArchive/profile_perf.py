## python profiler test
# profile_perf.py

# from concurrent.futures import ThreadPoolExecutor

# def find_divisors(n):
#     return [i for i in range(1, n + 1) if n % i == 0]

# def slow_function():
#     print("Slow thread started")
#     try:
#         return find_divisors(100_000_000)
#     finally:
#         print("Slow thread ended")

# def fast_function():
#     print("Fast thread started")
#     try:
#         return find_divisors(50_000_000)
#     finally:
#         print("Fast thread ended")

# def main():
#     with ThreadPoolExecutor(max_workers=2) as pool:
#         pool.submit(slow_function)
#         pool.submit(fast_function)

#     print("Main thread ended")

# if __name__ == "__main__":
#     main()

from cProfile import Profile
from pstats import SortKey, Stats

def fib(n):
    return n if n < 2 else fib(n - 2) + fib(n - 1)

with Profile() as profile:
    print(f"{fib(35) = }")
    (
        Stats(profile)
        .strip_dirs()
        .sort_stats(SortKey.CALLS)
        .print_stats()
    )