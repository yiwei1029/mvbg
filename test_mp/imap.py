from multiprocessing import Pool
from tqdm import tqdm
import time


def worker(i):
    # 执行任务的函数
    for _ in tqdm(range(5),desc=f"worker {i}"):
        i/(i-4)
        time.sleep(1)

def worker1(a,b):
    for _ in tqdm(range(5)):
        sum([a,b])
    time.sleep(2)

if __name__ == '__main__':
    p = Pool(4)
    start_time = time.time()
    args = [(i,j) for i,j in enumerate(range(10))]
    print(args)
    # list(tqdm(iterable=(p.imap(worker, range(10))), total=10,postfix=f'epoch {1}'))
    list(tqdm(p.starmap(worker1,args)))
    print(f"共耗时{int(time.time() - start_time)}s")
    p.close()
    p.join()