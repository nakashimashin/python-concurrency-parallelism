import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import time

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://nonexistent-subdomain.python.org/']

# URLの読み込み
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# 逐次実行
def sequential_execution():
    print("Sequential Execution Start")
    start_time = time.time()
    for url in URLS:
        try:
            data = load_url(url, 60)
            print(f'{url} page is {len(data)} bytes')
        except Exception as exc:
            print(f'{url} generated an exception: {exc}')
    end_time = time.time()
    print(f"Sequential Execution Time: {end_time - start_time:.2f} seconds")

# 並行実行
def concurrent_execution():
    print("Concurrent Execution Start")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                print(f'{url} page is {len(data)} bytes')
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')
    end_time = time.time()
    print(f"Concurrent Execution Time: {end_time - start_time:.2f} seconds")

def main():
    print("Main Start")
    sequential_execution()
    print("_" * 40)
    concurrent_execution()
    print("Main End")

if __name__ == '__main__':
    main()