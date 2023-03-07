import numpy as np
import cupy as cp


cuda_code = r"""
__device__ int Find(int x, int* parents) {
	int parent = parents[x];

	if(parent != x) {
		int next, prev = x;

		while(parent > (next = parents[parent])) {
			parents[prev] = next;
			prev = parent;
			parent = next;
		}
	}

	return parent;
}

__device__ void Union(int x, int y, int* parents) {
	if(x <= y) {
		return;
	}

	int xRoot = Find(x, parents);
	int yRoot = Find(y, parents);
	bool repeat;

	do {
		repeat = false;

		if(xRoot != yRoot) {
			int ret;

			if(xRoot < yRoot) {
				if((ret = atomicCAS(&parents[yRoot], yRoot, xRoot)) != yRoot) {
					yRoot = ret;
					repeat = true;
				}
			} else {
				if((ret = atomicCAS(&parents[xRoot], xRoot, yRoot)) != xRoot) {
					xRoot = ret;
					repeat = true;
				}
			}
		}
	} while(repeat);
}

extern "C"{

__global__ void compute_core_samples(float* X, int N, float eps, int min_samples, int* core_samples) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < N) {
		int n_neighbours = 0;

		for(int i=0; i<N; i++) {
			if(i == idx) {
				continue;
			}

			float dist = sqrt( (X[2*i] - X[2*idx]) * (X[2*i] - X[2*idx]) + (X[2*i+1] - X[2*idx+1]) * (X[2*i+1] - X[2*idx+1]) );
			if(dist < eps) {
				n_neighbours += 1;

				if(n_neighbours >= min_samples) {
					core_samples[idx] = 1;
					break;
				}
			}
		}
	}
}

__global__ void compute_labels(float* X, int N, float eps, int* core_samples, int* labels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < N && core_samples[idx] == 1) {
		for(int i=0; i<N; i++) {
			if(i == idx || core_samples[i] != 1) {
				continue;
			}

			float dist = sqrt( (X[2*i] - X[2*idx]) * (X[2*i] - X[2*idx]) + (X[2*i+1] - X[2*idx+1]) * (X[2*i+1] - X[2*idx+1]) );
			if(dist < eps) {
				Union(idx, i, labels);
			}
		}
	}
}

__global__ void label_noise(float* X, int N, float eps, int* core_samples, int* labels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < N && core_samples[idx] == 0) {
		float min_dist = eps;

		for(int i=0; i<N; i++) {
			if(i == idx || core_samples[i] != 1) {
				continue;
			}

			float dist = sqrt( (X[2*i] - X[2*idx]) * (X[2*i] - X[2*idx]) + (X[2*i+1] - X[2*idx+1]) * (X[2*i+1] - X[2*idx+1]) );
			if(dist < min_dist) {
				labels[idx] = labels[i];
				min_dist = dist;
			}
		}
	}
}

}
"""
module = cp.RawModule(code=cuda_code)
compute_core_samples = module.get_function("compute_core_samples")
compute_labels = module.get_function("compute_labels")
label_noise = module.get_function("label_noise")

def PDSdbscan(X: np.ndarray, eps: float, min_samples: int):
	N = X.shape[0]
	X = cp.asarray(X.reshape(-1), dtype=cp.float32)
	core_samples = cp.zeros([N], dtype=cp.int32)
	labels = cp.arange(N, dtype=cp.int32)

	compute_core_samples((1 + N//1024,), (1024,), (X, cp.int32(N), cp.float32(eps), cp.int32(min_samples), core_samples))
	compute_labels((1 + N//1024,), (1024,), (X, cp.int32(N), cp.float32(eps), core_samples, labels))
	label_noise((1 + N//1024,), (1024,), (X, cp.int32(N), cp.float32(eps), core_samples, labels))

	core_samples = cp.asnumpy(core_samples).astype(bool)
	labels = cp.asnumpy(labels)

	# Relabel to that we have nice numbers (i.e. 0, 1, 2 etc...)
	labels[~core_samples & (labels == np.arange(N))] = -1
	label_values = list(set(labels))
	for i in range(len(label_values)):
		if label_values[i] != i and label_values[i] != -1:
			labels[labels == label_values[i]] = i

	return labels, core_samples

