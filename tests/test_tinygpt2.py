import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

URL = "http://localhost:8000/chat"
PAYLOAD = {"message": "Found me 100000 vegan restaurants in Tokyo"}
HEADERS = {"Content-Type": "application/json"}

def one_request(session, idx):
    r = session.post(URL, headers=HEADERS, data=json.dumps(PAYLOAD), timeout=10)
    # Optionally inspect: return r.json()
    return r.status_code

def worker(worker_id, n=20):
    with requests.Session() as s:
        codes = []
        for j in range(n):
            try:
                codes.append(one_request(s, j))
            except Exception as e:
                codes.append(f"ERR:{e}")
        return worker_id, codes

def main():
    start = time.time()
    futures = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i in range(8):
            futures.append(ex.submit(worker, i, 20))
        for f in as_completed(futures):
            wid, codes = f.result()
            print(f"worker {wid} -> {codes.count(200)} OK / {len(codes)} total")
    print(f"Elapsed: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()