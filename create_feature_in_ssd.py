import numpy as np

if __name__ == "__main__":
    mmap_feat3 = np.memmap("/ssd/ogbn-products-feat.npy", dtype=np.float32, mode='w+', shape=(2449029, 100))
    mmap_feat3 = np.random.rand(2449029, 100)
    mmap_feat = np.memmap("/ssd/ogbn-papers100M-feat.npy", dtype=np.float32, mode='w+', shape=(111059956, 100))
    mmap_feat = np.random.rand(111059956, 100)
    mmap_feat2 = np.memmap("/ssd/friendster-feat.npy", dtype=np.float32, mode='w+', shape=(65608366, 128))
    mmap_feat2 = np.random.rand(65608366, 128)
