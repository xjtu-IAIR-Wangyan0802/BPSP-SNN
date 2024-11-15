#ifndef __LOCAL_CONNECT_CU_H__
#define __LOCAL_CONNECT_CU_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class LocalConnect: public ConvLayerBase
{
public:
	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight();
	void clearMomentum();
	void calCost();
	
	LocalConnect(std::string name);

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);

	~LocalConnect(){
		delete outputs;
	}

	cuMatrix<float>* getOutputs(){
		return outputs;
	};

	cuMatrix<float>* getCurDelta(){
		return curDelta;
	}

    cuMatrix<bool>* getSpikingOutputs(){
        return NULL;
    }

    cuMatrix<int>* getFireCount(){
        return NULL;
    }

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	virtual void printParameter(){
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w[0]->toCpu();
		sprintf(logStr, "weight:%f, %f;\n", w[0]->get(0,0,0), w[0]->get(0,1,0));
		LOG(logStr, "Result/log.txt");
		b[0]->toCpu();
		sprintf(logStr, "bias  :%f\n", b[0]->get(0,0,0));
		LOG(logStr, "Result/log.txt");
	}
private:
	cuMatrix<float>* inputs;
	cuMatrix<float>* preDelta;
	cuMatrix<float>* outputs;
	cuMatrix<float>* curDelta;
	int kernelSize;
	int batch;
	int NON_LINEARITY;
	float lambda;
	int localKernelSize;
private:
	cuMatrixVector<float> w;
	cuMatrixVector<float> wgrad;
	cuMatrixVector<float> b;
	cuMatrixVector<float> bgrad;
	cuMatrixVector<float> momentum_w;
	cuMatrixVector<float> momentum_b;
	cuMatrixVector<float> wgradTmp;
};

#endif 
