from concurrent.futures import ProcessPoolExecutor
import math
import time

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419
]

# 素数判定
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = math.floor(math.sqrt(n))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

# 逐次実行
def sequential_execution():
    print("Sequential Execution Start")
    start_time = time.time()
    results = [(n, is_prime(n)) for n in PRIMES]
    end_time = time.time()
    for number, prime in results:
        print(f'{number} is prime: {prime}')
    print(f"Sequential Execution Time: {end_time - start_time:.2f} seconds")

# 並列実行
def parallel_execution():
    print("Parallel Execution Start")
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        results = zip(PRIMES, executor.map(is_prime, PRIMES))
    end_time = time.time()
    for number, prime in results:
        print(f'{number} is prime: {prime}')
    print(f"Parallel Execution Time: {end_time - start_time:.2f} seconds")

def main():
    print("Main Start")
    sequential_execution()
    print("_" * 40)
    parallel_execution()
    print("Main End")

if __name__ == '__main__':
    main()

