#ifndef __MEMORY_MONITOR_H__
#define __MEMORY_MONITOR_H__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <map>
class MemoryMonitor
{
public:
	static MemoryMonitor* instance(){
		static MemoryMonitor* monitor = new MemoryMonitor();
		return monitor;
	}
	void* cpuMalloc(int size);
	cudaError_t gpuMalloc(void** devPtr, int size);
	MemoryMonitor(): gpuMemory(0), cpuMemory(0){}
	void printCpuMemory(){
		// sprintf(logStr, "total malloc cpu memory %fMb\n", cpuMemory / 1024 / 1024);
		// LOG(logStr, "Result/log.txt");
		printf("total malloc cpu memory %fMb\n", cpuMemory / 1024 / 1024);
		}
	void printGpuMemory(){
		// sprintf(logStr, "total malloc gpu memory %fMb\n", gpuMemory / 1024 / 1024);
		// LOG(logStr, "Result/log.txt");
		printf("total malloc gpu memory %fMb\n", gpuMemory / 1024 / 1024);
		}
	void freeGpuMemory(void* ptr);
	void freeCpuMemory(void* ptr);
private:
	float cpuMemory;
	float gpuMemory;
	std::map<void*, float>cpuPoint;
	std::map<void*, float>gpuPoint;
};
#endif
