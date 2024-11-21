from concurrent.futures import ThreadPoolExecutor
import time

def func_1():
    for n in range(3):
        time.sleep(2)
        print(f'func_1: {n}')
    return '結果1'

def func_2():
    for n in range(3):
        time.sleep(1)
        print(f'func_2: {n}')
    return '結果2'

def main():
    print("Start")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_1 = executor.submit(func_1)
        future_2 = executor.submit(func_2)

    result_1 = future_1.result()
    result_2 = future_2.result()

    print(f'result_1: {result_1}')
    print(f'result_2: {result_2}')

    print("End")

if __name__ == '__main__':
    main()
