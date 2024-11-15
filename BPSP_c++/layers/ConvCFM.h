#ifndef __CONV_COMBINE_FEATURE_MAP_CU_H__
#define __CONV_COMBINE_FEATURE_MAP_CU_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class ConvCFM: public ConvLayerBase
{
public:
	ConvCFM(std::string name);
	~ConvCFM(){
		delete outputs;
	}


	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight();
	void clearMomentum();
	void calCost();

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);

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

	int getInputDim(){
		return inputDim;
	}

	int getOutputDim(){
		return outputDim;
	}

	virtual void printParameter(){
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w[0]->toCpu();
		sprintf(logStr, "weight:%f, %f, %f;\n", w[0]->get(0,0,0), w[0]->get(0,1,0), w[0]->get(0, 2, 0));
		LOG(logStr, "Result/log.txt");
		b[0]->toCpu();
		b[1]->toCpu();
		sprintf(logStr, "bias  :%f %f\n", b[0]->get(0,0,0), b[1]->get(0, 0, 0));
		LOG(logStr, "Result/log.txt");
	}

private:
	cuMatrix<float>* inputs;
	cuMatrix<float>* preDelta;
	cuMatrix<float>* outputs;
	cuMatrix<float>* curDelta; // size(curDelta) == size(outputs)
	int kernelSize;
	int padding;
	int batch;
	int NON_LINEARITY;
	int cfm;
	float lambda;
private:
	cuMatrixVector<float> w;
	cuMatrixVector<float> wgrad;
	cuMatrixVector<float> wgradTmp;
	cuMatrixVector<float> b;
	cuMatrixVector<float> bgrad;
	cuMatrixVector<float> momentum_w;
	cuMatrixVector<float> momentum_b;
};

#endif 
