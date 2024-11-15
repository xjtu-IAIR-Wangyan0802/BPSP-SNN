#include "Spiking.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../common/util.h"
#include "../readData/readSpeechData.h"
#include <fstream>
#include <assert.h>
#include <math.h>

//#define DEBUG
#define I_IDX 0
#define O_IDX 0

/*
 * Device func for accumulate the spike response累计脉冲响应
*/
__device__ float d_Spiking_accumulate_spikes(
    int inputSize,
    int outputSize,
    float* input_resp,
    bool* output,
    int o_idx,
    float* weights,
    float* weights_lat,
    float* biases,
    int t,
    int dummyFreq, 
    int endTime);

/*
 * Device func for spike gradient for each pair of binary spike response
 */
__device__ float d_Spiking_gradient(
    bool* output,
    bool* input,
    float delta,
    int o_idx,
    int i_idx,
    int outputSize,
    int inputSize,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);

/*
 * Device func for spike gradient for each pair of spike train in times
 */
__device__ float d_Spiking_gradient_spiketime(
    int* output_time,
    int* input_time,
    int n_ospikes,
    int n_ispikes,
    float delta,
    int o_idx,
    int i_idx,
    float lat_factor,
    int outputSize,
    int inputSize,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);

/*
 * Device func for spike gradient for bias in times
 */
__device__ float d_Spiking_bias_gradient_spiketime(
    int* output_time,
    int n_ospikes,
    float delta,
    int o_idx,
    int dummyFreq,
    int outputSize,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);

    
/*
 * dim3 block = dim3(1);
 * dim3 thread= dim3(256);
 */
__global__ void g_getCost_output(
    int*   fireCount,
    float* groundTruth,
    float* cost,
    int*   y,
    int    batch,
    int    cols,
    float UNDESIRED_LEVEL,
    float DESIRED_LEVEL,
    float MARGIN);

/*
 * dim3 block = dim3(1);
 * dim3 thread= dim3(256);
 */
__global__ void g_getDelta_output(
    float* outputDelta,
    int*   fireCount,
    float* groundTruth,
    int    len,
    float  MARGIN);
  
/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(1024, outputSize));
 */
__global__ void g_getMaxCount(
    int* fireCount,
    int* maxCount,
    int cols); 

/*
 * dim3 block = dim3(batch, inputSize);
 * dim3 thread= min(1024, outputSize);
 */
__global__ void g_Spiking_wgrad(
        bool* inputs,
        bool* outputs,
        float* curDelta,
        float* wgradTmp,
        int inputSize,
        int outputSize,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S);

/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, inputSize);
 */
__global__ void g_Spiking_wgrad_sideEffect(
        float* weights,
        int* batchFireCount,
        float* batchAccEffect,
        float  vth,
        int inputSize,
        int outputSize,
        float * batchSideEffect);

/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, inputSize);
 */
__global__ void g_Spiking_wgrad_spiketime(
        float* batchSideEffect,
        float* batchAccEffect,
        float* curDelta,
        float* latFactor,
        float* wgradTmp,
        int inputSize,
        int outputSize);

/*
 * dim3 block = dim3(outputSize);
 * dim3 thread= dim3(batch);
 */
__global__ void g_Spiking_bgrad_spiketime(
        int* outputs_time,
        int* batchFireCount,
        float* curDelta,
        float* bgradTmp,
        int outputSize,
        int endTime,
        int dummyFreq,
        int T_REFRAC,
        float TAU_M,
        float TAU_S);


/*
 * block = dim3(outputSize * inputSize);
 * thread= dim3(batch);
*/
__global__ void g_Spiking_gradAdd(
	float* wgradTmp,
	float* wgrad,
	float* w,
    float* w_sq_sum,
	int batch,
	float lambda,
    float beta,
    float limit,
    int inputSize,
	int wArea);


/*
 * dim3 block = dim3(batch, inputSize);
 * dim3 thread= min(1024, outputSize);
 */
__global__ void g_Spiking_debug_spiketime(
    int* inputs_time,
    int* outputs_time,
    int* batchPreFireCount,
    int* batchFireCount,
    int inputSize,
    int outputSize,
    int endTime);


void Spiking::calCost()
{
    cost->gpuClear();//将成本函数的值清零
    if(predict == NULL){
        printf("Warning::Try to compute the cost when the predict is not properly set!\n ");
        return;
    }
    g_getCost_output<<<dim3(1), dim3(256), sizeof(float) * 256>>>(fireCount->getDev(),
            groundTruth->getDev(),
            cost->getDev(),
            predict,
            batch,
            fireCount->cols,
            UNDESIRED_LEVEL,
            DESIRED_LEVEL,
            MARGIN);
    cudaStreamSynchronize(0);
    getLastCudaError("Spiking:g_getCost_output");
}

void Spiking::feedforward()
{
    if((inputs == NULL))
    {
        printf("Spiking init error\n");
        exit(0);
    }

    // fast input response
    //将输入矩阵从bool类型转换为float类型，并将其存储在inputs_float数组中
    g_cast_bool_2_float<<<dim3(batch, endTime), min(1024, inputSize)>>>(inputs->getDev(), endTime, inputSize, inputs->cols, inputs->channels, batch, inputs_float->getDev());
    //计算输入矩阵和权重矩阵的点积，并将结果存储在inputs_resp_tmp数组中
    matrixMul(w, inputs_float, inputs_resp_tmp); //input_resp_tmp rows:outputSize; cols:endTime*batch
    //将inputs_resp_tmp数组转换为批处理的格式，并将结果存储在inputs_resp数组中
    g_transform_2_batch<<<dim3(batch, outputSize), min(1024, endTime)>>>(inputs_resp_tmp->getDev(), endTime, outputSize, batch, inputs_resp->getDev());

    //将输入矩阵中的脉冲时间转换为格式化的脉冲时间，并将结果存储在inputs_time_format数组中
    g_convert_spiketimes<<<dim3(batch, endTime), min(1024, inputSize)>>>(inputs_time->getDev(), endTime, inputSize, inputs_time->cols, inputs_time->channels, batch, inputs_time_format->getDev());
    //将预先计算的脉冲发放次数转换为格式化的脉冲发放次数，并将结果存储在preFireCount_format数组中。
    g_convert_firecounts<<<dim3(batch), min(1024, inputSize)>>>(preFireCount->getDev(), preFireCount->getArea(), inputSize, preFireCount->cols, preFireCount->channels, batch, preFireCount_format->getDev());

    dim3 thread= dim3(min(1024, outputSize));
    dim3 block = dim3(batch);
    ConfigSpiking * config = (ConfigSpiking*) Config::instance()->getLayerByName(m_name); 
    int dummyFreq = config->getBiasFreq();
    //脉冲网络前馈计算
    g_Spiking_feedforward<<<block, thread>>>(
            inputs_resp->getDev(),
            w->getDev(),
            w_laterial == NULL ? NULL : w_laterial->getDev(),
            b->getDev(),
            outputs->getDev(),
            fireCount->getDev(),
            inputSize,
            outputSize,
            endTime,
            threshold,
            dummyFreq,
            T_REFRAC,//不应期，固定值2
            TAU_M,
            TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("Spiking::g_Spiking_feedforward");

    block = dim3(batch, 1);
    thread = dim3(min(outputSize, 1024));

    // transform the binary response matrix to the spike times
    //将二进制响应矩阵转换为脉冲次数矩阵，直接存储脉冲计数
    g_response_2_spiketime<<<block, thread>>>(
            outputs->getDev(),
            outputs_time->getDev(),
            outputs->getArea(),
            outputSize,
            endTime);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("Spiking:g_response_2_spiketime");

}


void Spiking::backpropagation()
{ 
    if(m_name == std::string("output")){
        // compute the cost function，返回差值的平方,gt设置(如果是正确类别的话，放电数目35，否则5)
        g_getCost_output<<<dim3(1), dim3(256), sizeof(float) * 256>>>(fireCount->getDev(), groundTruth->getDev(), cost->getDev(), predict, batch, fireCount->cols, UNDESIRED_LEVEL, DESIRED_LEVEL, MARGIN);
        cudaStreamSynchronize(0);
        getLastCudaError("Spiking::g_getCost_output");

        // compute the delta (error)计算放电计数和gt之间的绝对差值，大于一个阈值MARGIN-5，返回差值，否则返回0
        //公式15，计算输出层的损失Delta，误差对总的突触后膜电势求导
        g_getDelta_output<<<dim3(1), dim3(256)>>>(curDelta->getDev(), fireCount->getDev(), groundTruth->getDev(), curDelta->getLen(), MARGIN);
        cudaStreamSynchronize(0);
        getLastCudaError("Spiking::g_getDelta_output");

        // apply the sample weights
        //根据样本权重缩放误差。它通过遍历所有当前批次的样本，并使用线程索引threadIdx.x来确定当前线程负责计算哪个神经元的误差。
        g_boostWeight_output<<<dim3(batch), dim3(outputSize)>>>(curDelta->getDev(), sample_weights, curDelta->getLen());
        cudaStreamSynchronize(0);
        getLastCudaError("Spiking::g_boostWeight_output");

        // compute the lateral factors if applicable计算横向因子
        if(lateralFactor != NULL && w_laterial != NULL){
            int threads = min(outputSize, 1024);
            //计算突出累积效应比，effect_ratio_jl * 1/vth * effect_ratio_lj * 1/vth
            g_getLateralFactor_output<<<dim3(batch, outputSize), threads, sizeof(float) * threads>>>(
                outputs_time->getDev(),
                fireCount->getDev(),
                lateralW,
                predict,
                lateralFactor->getDev(),
                threshold,
                outputSize,
                endTime,
                T_REFRAC,
                TAU_M,
                TAU_S);
            cudaStreamSynchronize(0);
            getLastCudaError("Spiking::g_getLateralFactor_output");
        }

        //如果目标神经元不放电的时候，修改目标神经元的输出脉冲序列和输出脉冲放电计数
        g_modifySpikes<<<dim3(batch), dim3(min(outputSize, 1024))>>>(outputs->getDev(), predict, fireCount->getDev(), DESIRED_LEVEL, endTime, outputSize);
        cudaStreamSynchronize(0);
        getLastCudaError("Spiking::g_modifySpikes");

        // retransform the binary matrix to the spike times since the outputs might be changed
        //将二进制响应矩阵转换为脉冲次数矩阵
        g_response_2_spiketime<<<dim3(batch, 1), dim3(min(outputSize, 1024))>>>(
                outputs->getDev(),
                outputs_time->getDev(),
                outputs->getArea(),
                outputSize,
                endTime);
        checkCudaErrors(cudaStreamSynchronize(0));
        getLastCudaError("Spiking:g_response_2_spiketime");
    }
    // pre compute the accumulative synaptic effect, and effect ratio (if applicable)
    // 预先计算累积突触效应和效应比(如果适用)
    dim3 thread = dim3(min(1024, outputSize));
    dim3 block  = dim3(batch, inputSize);
    //cudaFuncCachePreferL1表示使用48KB一级缓存和16KB共享内存
    cudaFuncSetCacheConfig(g_Spiking_synaptic_effect, cudaFuncCachePreferL1);
    //计算effectRatio，w*(e/o)
    //预先计算累积突触效应和效应比，并将结果存储在accEffect和effectRatio数组中。
    g_Spiking_synaptic_effect<<<block, thread>>>(
        inputs_time_format->getDev(),
        outputs_time->getDev(),
        preFireCount_format->getDev(),
        fireCount->getDev(),
        w->getDev(),
        accEffect->getDev(),
        effectRatio == NULL ? NULL : effectRatio->getDev(),
        inputSize,
        outputSize,
        endTime,
        T_REFRAC,
        TAU_M,
        TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_Spiking_synaptic_effect");


   
    // divide the curDelta by vth
    block = dim3(batch, 1);
    thread = dim3(min(1024, outputSize));
    //g_divide_by_threshold：delta[o_idx] /= threshold，通过计算总的突触后膜电势/阈值，求脉冲放电计数，脉冲发放次数矩阵
    g_divide_by_threshold<<<block, thread>>>(curDelta->getDev(), curDelta->getArea(), curDelta->cols, threshold);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_divide_by_threshold");
    
    //计算结果1/v *sum(w*e/o)


    // compute preDelta: curDelta: batch * outputSize; w: outputSize * inputSize
    //计算前一层的误差，如果当前层不是输出层，则调用matrixMul函数计算当前层的误差和权重矩阵的点积，并将结果存储在preDelta_format数组中。
    if(preDelta == NULL){
        
        ConfigSpiking* config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
        assert(config->m_input == "data");
    }
    else{
        if(effectRatio != NULL){
            matrixMul(curDelta, effectRatio, preDelta_format);//
        }
        else{
            matrixMul(curDelta, w, preDelta_format);
        }
        // preDelta_format: (batch, channels * size, 1) -> preDelta: (batch, size, channels)
        block = batch;
        thread = min(512, preDelta->channels * preDelta->cols);
        //将preDelta_format数组转换为前一层的误差矩阵，并将结果存储在preDelta数组中。
        g_preDeltaFormat<<<block, thread>>>(preDelta_format->getDev(), preDelta->getDev(),
            preDelta->rows, preDelta->cols, preDelta->channels);
        cudaStreamSynchronize(0);
        getLastCudaError("g_preDeltaFormat");
    } 
}



/*
 * block = dim3(outputSize, 1);
 * thread= dim3(min(inputSize, 1024));
*/
//计算权重矩阵每一行的平方和,将权重矩阵的每一行视为一个向量，并计算该向量的平方和
//正则化，以防止权重矩阵过度拟合训练数据
__global__ void g_Spiking_calSquareSum(
    float* w,
    float* w_sq_sum,
    int outputSize,
    int inputSize,
    float weight_limit)
{
    extern __shared__ float _sum[];
    int o_id = blockIdx.x;
    int tid = threadIdx.x;

    //共享内存中的值初始化为0
    _sum[tid] = 0;
    __syncthreads();

    //计算每个线程所对应的元素的贡献,每个线程将计算出权重矩阵中一个元素的平方，然后将它们加到共享内存中。
    for(int i = 0; i < inputSize; i += blockDim.x)
    {
        int id = i + tid;
        if(id < inputSize)
        { 
            int wid = id + o_id * inputSize;
            float weight = w[wid];
            _sum[tid] += (weight/weight_limit) * (weight/weight_limit);
        }
    }
    __syncthreads();
    int len = blockDim.x;
    //使用并行的规约算法来计算线程块的总和，并将结果存储在w_sq_sum数组中
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(tid < skip && (tid + skip) < len)
        {
            _sum[tid] += _sum[tid + skip];
        }
        len = skip;
    }
    if(tid == 0)
        w_sq_sum[o_id] = _sum[0] / inputSize;
}

/*
 * block = dim3(outputSize * inputSize);
 * thread= dim3(batch);
*/
//计算权重梯度并更新权重矩阵
__global__ void g_Spiking_gradAdd(
	float* wgradTmp,//每个样本的权重梯度的临时存储器的指针
	float* wgrad,//权重梯度的指针
	float* w,//权重矩阵的指针
    float* w_sq_sum,//存储每行平方和的数组的指针
	int batch,
	float lambda,
    float beta,
    float limit,
    int inputSize,//权重矩阵的列数
	int wArea)//权重矩阵的大小
{
	extern __shared__ float _sum[];

	int wid = blockIdx.x;
	int tid = threadIdx.x;

	_sum[tid] = 0;
	__syncthreads();

    //每个线程对应的样本的权重梯度加到共享内存中
	for(int i = 0; i < batch; i += blockDim.x)
	{
		int b = i + threadIdx.x;//当前线程对应的样本的索引
		if(b < batch)
		{
			_sum[threadIdx.x] += wgradTmp[b * wArea + wid];
		}
	}
	__syncthreads();
	int len = blockDim.x;
    //用于使用并行的规约算法计算线程块的总和
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(tid < skip && (tid + skip) < len)
		{
			_sum[tid] += _sum[tid + skip];
		}
		len = skip;
	}
    //计算并更新权重矩阵中一个元素的权重梯度
	if(tid == 0)
	{
        float sq_sum = w_sq_sum[wid / inputSize];//计算并更新权重矩阵中一个元素的权重梯度
        //使用线程块的总和除以样本的数量batch计算出该元素的权重梯度。
        //再将该线程将权重梯度与正则化项相加，得到最终的权重梯度，并将其存储在wgrad数组
		wgrad[wid] = _sum[0] / batch + lambda*beta*(w[wid]/limit)*__expf(beta*(sq_sum - 1));
	}
}

//计算当前层的权重梯度
void Spiking::getGrad()
{
    dim3 thread = dim3(min(1024, inputSize));
    dim3 block  = dim3(batch, outputSize);
    //计算sum(w*e/o) /vth
    g_Spiking_wgrad_sideEffect<<<block, thread, sizeof(float) * min(1024, inputSize)>>>(
        w->getDev(),
        fireCount->getDev(),
        accEffect->getDev(),
        threshold,
        inputSize,
        outputSize,
        sideEffect->getDev());
    checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Spiking_wgrad_sideEffect");
   
    cudaFuncSetCacheConfig(g_Spiking_wgrad_spiketime,cudaFuncCachePreferL1);

    //E对w求梯度=当前层Delta * e *（1+sum(w*e/o) /vth）,公式23
    g_Spiking_wgrad_spiketime<<<block, thread>>>(
        sideEffect->getDev(),
        accEffect->getDev(),
        curDelta->getDev(),
        lateralFactor == NULL ? NULL : lateralFactor->getDev(),
        wgradTmp->getDev(),
        inputSize,
        outputSize);

    checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Spiking_wgrad_spiketime");

    block = dim3(outputSize);
    thread = dim3(min(inputSize, 1024));

    g_Spiking_calSquareSum<<<block, thread, sizeof(float) * min(inputSize, 1024)>>>(
        w->getDev(),
        weightSqSum->getDev(),
        outputSize,
        inputSize,
        weightLimit);
    checkCudaErrors(cudaStreamSynchronize(0));    
	getLastCudaError("g_Spiking_calSquareSum");
 
	block  = dim3(outputSize * inputSize);
	thread = dim3(batch);

	g_Spiking_gradAdd<<<block, thread, sizeof(float) * batch>>>(
		wgradTmp->getDev(),
		wgrad->getDev(),
		w->getDev(),
        weightSqSum->getDev(),
		batch,
		lambda,
        beta,
        weightLimit,
        inputSize,
		w->getArea());

	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Spiking_gradAdd");
    
    // add the bias derivation here:
}	

//使用Adam或者SGD根据之前求得的权重变化值更新权重,根据计算出的权重梯度更新权重矩阵
void Spiking::updateWeight()
{
    dim3 block  = min((w->getLen() + 255)/ 256, 5120);
    dim3 thread = 256;

    if(Config::instance()->getOptimizerType() == std::string("adam")){
        g_adam_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
            g1_w->getDev(),
            g2_w->getDev(),
            b1_t,
            b2_t,
            wgrad->getDev(),
            w->getDev(),
            w->getLen(),
            Config::instance()->getLrate());
            b1_t *= 0.9f; b2_t *= 0.999f;
    } 
    else{
        g_sgd_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
            momentum_w->getDev(),
            wgrad->getDev(), 
            w->getDev(),
            w->getLen(), 
            Config::instance()->getMomentum(),
            Config::instance()->getLrate());
    }
    // handle the bias here
}


Spiking::Spiking(std::string name)
{
	m_name = name;
	ConfigSpiking* config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
	SpikingLayerBase * preLayer = (SpikingLayerBase*)Layers::instance()->get(config->m_input);

	inputs = preLayer->getSpikingOutputs();
    inputs_time = preLayer->getSpikingTimeOutputs();
    inputs_time_format = new cuMatrix<int>(inputs_time->rows, inputs_time->cols * inputs_time->channels, 1);
	preDelta = preLayer->getCurDelta();
	preDelta_format = NULL;
    if(preDelta != NULL)
        preDelta_format = new cuMatrix<float>(preDelta->rows, preDelta->cols * preDelta->channels, 1);

    preFireCount = preLayer->getFireCount();
    preFireCount_format = new cuMatrix<int>(preFireCount->rows, preFireCount->cols * preFireCount->channels, 1);
	
    endTime   = Config::instance()->getEndTime(); 
	batch     = Config::instance()->getBatchSize();
	lambda    = Config::instance()->getLambda();
    beta      = Config::instance()->getBeta();
    T_REFRAC  = config->m_t_ref;
    TAU_M     = config->m_tau_m;
    TAU_S     = config->m_tau_s;    

	inputSize  = inputs->cols * inputs->channels / endTime;
	outputSize = config->m_numNeurons;

    weightLimit = Config::instance()->getWeightLimit();

    UNDESIRED_LEVEL = config->m_undesired_level;
    DESIRED_LEVEL   = config->m_desired_level;
    MARGIN          = config->m_margin; 

    outputs  = new cuMatrix<bool>(batch, outputSize * endTime, 1);
    outputs_time = new cuMatrix<int>(batch, outputSize * endTime, 1);

    // for fast input response
    inputs_resp_tmp = new cuMatrix<float>(outputSize, endTime * batch, 1);
    inputs_resp = new cuMatrix<float>(batch, outputSize * endTime, 1);
    inputs_float = new cuMatrix<float>(inputSize, endTime * batch, 1);

	curDelta = new cuMatrix<float>(batch, outputSize, 1);
    fireCount= new cuMatrix<int>(batch, outputSize, 1);
    weightSqSum = new cuMatrix<float>(outputSize, 1, 1);
    maxCount    = new cuMatrix<int>(batch, 1, 1);
    accEffect   = new cuMatrix<float>(batch, outputSize * inputSize, 1); 
    sideEffect  = new cuMatrix<float>(batch, outputSize, 1);

    predict = NULL;


    //梯度求导计算
    // only for the output
    //可在隐藏层增加误差
    if(config->m_name == std::string("output")){
        groundTruth   = new cuMatrix<float>(batch, outputSize, 1);
        cost          = new cuMatrix<float>(1, 1, 1);
    }
    else{
        groundTruth   = NULL;
        cost          = NULL;
    }
    assert(outputSize > 0 && inputSize > 0);

    w        = new cuMatrix<float>(outputSize, inputSize, 1);
    b        = new cuMatrix<float>(outputSize, 1, 1);
    wgrad    = new cuMatrix<float>(outputSize, inputSize, 1);
    bgrad    = new cuMatrix<float>(outputSize, 1, 1);
    wgradTmp = new cuMatrix<float>(batch, outputSize * inputSize, 1);
    if(config->hasLaterialWeight() == true){
        w_laterial = new cuMatrix<float>(outputSize, outputSize, 1);
    }
    else
        w_laterial = NULL;
    
    threshold = config->m_vth;
   
    // lateral inihibition factor for the output
    //输出层的横向抑制因子
    lateralFactor = NULL;
    lateralW = 0.0f;
    if(config->hasLaterialInh() == true && config->m_name == std::string("output")){
        lateralFactor = new cuMatrix<float>(batch, outputSize, 1);
        lateralW = config->m_localInbStrength;
    }

    if(Config::instance()->useEffectRatio()){
        if(batch > 1){
            printf("Must set batch size to 1 if use effect ratio for grad of synaptic effect.\n");
            printf("Current batch size: %d\n", batch);
            assert(batch <= 1);
        }
        effectRatio = new cuMatrix<float>(outputSize, inputSize, 1);
    }


    //Adam优化器更新权重
    momentum_w = new cuMatrix<float>(outputSize, inputSize, 1);
    momentum_b = new cuMatrix<float>(outputSize, 1, 1);
    g1_w       = new cuMatrix<float>(outputSize, inputSize, 1); // for adam
    g1_b       = new cuMatrix<float>(outputSize, 1, 1);
    g2_w       = new cuMatrix<float>(outputSize, inputSize, 1);
    g2_b       = new cuMatrix<float>(outputSize, 1, 1);
    b1_t = 0.9;
    b2_t = 0.999;
 
	this->initRandom();
    w_ref = NULL;
    w_laterial_ref = NULL;
    b_ref = NULL; 

    if(Config::instance()->getIsGradientChecking())
        this->loadRef(); // for verification purpose

    Layers::instance()->set(m_name, this);
}

void Spiking::save(FILE* file)
{
    w->toCpu();
    b->toCpu();

    for(int c = 0; c < w->channels; c++){
        for(int i = 0; i < w->rows; i++){
            for(int j = 0; j < w->cols; j++){
                fprintf(file, "%f ", w->get(i, j, c));
            }
        }
    }
    if(w_laterial != NULL){
        for(int c = 0; c < w_laterial->channels; c++){
            for(int i = 0; i < w_laterial->rows; i++){
                for(int j = 0; j < w_laterial->cols; j++){
                    fprintf(file, "%f ", w_laterial->get(i, j, c));
                }
            }
        } 
    }

    for(int c = 0; c < b->channels; c++){
        for(int i = 0; i < b->rows; i++){
            for(int j = 0; j < b->cols; j++){
                fprintf(file, "%f ", b->get(i, j, c));
            }
        }
    }
}

void Spiking::clearMomentum()
{
    momentum_b->gpuClear();
    momentum_w->gpuClear();
}

void Spiking::verify(const std::string& phrase)
{
    printf("Verify for the layer: %s at %s phrase.\n", m_name.c_str(), phrase.c_str());
    if(phrase == std::string("train"))
    {
        if(!output_train_ref.empty()){
            outputs->toCpu();
            checkMatrixIsSame(output_train_ref[0], outputs, outputSize);
        }
        
    }
    else if(phrase == std::string("test"))
    {
        if(w_ref != NULL){
            w->toCpu();
            checkMatrixIsSame(w_ref, w);
        }
        if(w_laterial_ref != NULL && w_laterial != NULL){
            w_laterial->toCpu();
            checkMatrixIsSame(w_laterial_ref, w_laterial);
        }
 
        if(b_ref != NULL){
            b->toCpu();
            checkMatrixIsSame(b_ref, b);
        }
    
        if(!output_test_ref.empty()){
            outputs->toCpu();
            checkMatrixIsSame(output_test_ref[0], outputs, outputSize);
        }
    }
    printf("Verification for the layer: %s at %s phrase. Pased!!\n", m_name.c_str(), phrase.c_str());
}

//* load the reference weights and output spikes for verification
void Spiking::loadRef()
{
    if(batch != 1){
        printf("Only do the verification for one batch and one sample!\n");
        exit(0);
    }
    ConfigSpiking * config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
    if(config->m_ref_weight_path != std::string("NULL")){
        w_ref = new cuMatrix<float>(outputSize, inputSize, 1);
        initFromDumpfile(config->m_ref_weight_path, w_ref);
        if(config->hasBias()){
            b_ref = new cuMatrix<float>(outputSize, 1, 1);
            initBiasFromDumpfile(config->m_ref_weight_path, b_ref);
        }
    }

    if(config->m_ref_lweight_path != std::string("NULL")){
        w_laterial_ref = new cuMatrix<float>(outputSize, outputSize, 1);
        initFromDumpfile(config->m_ref_lweight_path, w_laterial_ref);
    }

    if(config->m_ref_output_train_path != std::string("NULL")){
        read_each_speech_dump(config->m_ref_output_train_path, output_train_ref, endTime, outputSize);
        assert(output_train_ref.size() == 1 && output_train_ref[0] != NULL);
        output_train_ref[0]->rows = 1;
        output_train_ref[0]->cols = endTime * outputSize;
    }

    if(config->m_ref_output_test_path != std::string("NULL")){
        read_each_speech_dump(config->m_ref_output_test_path, output_test_ref, endTime, outputSize);
        assert(output_test_ref.size() == 1 && output_test_ref[0] != NULL);
        output_test_ref[0]->rows = 1;
        output_test_ref[0]->cols = endTime * outputSize;
   }

}

void Spiking::initRandom()
{
    ConfigSpiking * config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
    float initW = config->m_initW;
 
    if(config->isGaussian()){
        float epsilon = initW;
        for(int c = 0; c < w->channels; c++)
        {
            createGaussian(w->getHost() + c * w->getArea(),
                    outputSize, inputSize, w->channels, epsilon);
        }
        w->toGpu();
    }
    else if(config->isBernoulli()){
        for(int j = 0; j < w->getLen(); j++){
            w->getHost()[j] =  initW * (2.0f * rand() / RAND_MAX - 1.0f);
            //printf("%f ", w->getHost()[j]);
        }//printf("\n");
        w->toGpu();
    }
    else if(config->isFixed()){
        // one input connects to nconnect randomly selected outputs, with initW/-initW
        int nconnect = config->m_weightConnect;
        assert(nconnect > 0);
        for(int c = 0; c < w->channels; ++c){
            for(int i = 0; i < w->rows; ++i){
                for(int t = 0; t < nconnect; ++t){
                    int j = rand() % inputSize;
                    if(rand() % 2 == 0)
                        w->set(i, j, c, initW);
                    else
                        w->set(i, j, c, -1.0*initW);
                    //printf("input_%d to reservoir_%d : %f\n", j, i, w->get(i, j, c));
                }
            }
        }
        w->toGpu();
    }
    else if(config->isExternal()){
        initFromDumpfile(config->m_weightPath, w);
    }
    if(config->hasLaterialWeight()){
        initLaterial();
    }

}

void Spiking::initFromCheckpoint(FILE* file)
{
    float val = 0;
    for(int c = 0; c < w->channels; c++){
        for(int i = 0; i < w->rows; i++){
            for(int j = 0; j < w->cols; j++){
                if(fscanf(file, "%f", &val) == EOF)
                {
                    char logStr[256];
                    sprintf(logStr, "scanf fail for layer: %s\n", m_name.c_str());
                    LOG(logStr, "Result/log.txt");
                    assert(0);
                }
                w->set(i, j, c, val);
            }
        }
    }

    if(w_laterial != NULL){
        for(int c = 0; c < w_laterial->channels; c++){
            for(int i = 0; i < w_laterial->rows; i++){
                for(int j = 0; j < w_laterial->cols; j++){
                    if(fscanf(file, "%f", &val) == EOF)
                    {
                        char logStr[256];
                        sprintf(logStr, "scanf fail for layer: %s\n", m_name.c_str());
                        LOG(logStr, "Result/log.txt");
                    }
                    w_laterial->set(i, j, c, val);
                }
            }
        } 
    }

    for(int c = 0; c < b->channels; c++){
        for(int i = 0; i < b->rows; i++){
            for(int j = 0; j < b->cols; j++){
                if(fscanf(file, "%f", &val) == EOF)
                {
                    char logStr[256];
                    sprintf(logStr, "scanf fail for layer: %s\n", m_name.c_str());
                    LOG(logStr, "Result/log.txt");
                    assert(0);
                }
                b->set(i, j, c, val);
            }
        }
    }

    w->toGpu();
    b->toGpu();
}

//* initial the weights from the dumped file by the CPU sim
void Spiking::initFromDumpfile(const std::string& filename, cuMatrix<float>*& cuW)
{
    ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        printf("Cannot open the file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
 
    assert(cuW != NULL);
    std::vector<std::vector<float> > weights(cuW->rows, std::vector<float>(cuW->cols, 0.0f));
   
    int idx; 
    float weight;
    std::string pre_name, post_name;
    while(f_in>>idx>>pre_name>>post_name>>weight){
        int pre = extractNeuronIndex(pre_name);
        int post = extractNeuronIndex(post_name);
        if(post >= weights.size() || pre >= weights[0].size()){
            if(pre == weights[0].size() && post < weights.size()){ // this is related to bias    
                continue;
            }
            else{
                printf("Read the file: %s, in line: %d\n", filename.c_str(), idx);
                printf("Post: %d, OutputDim: %d\n Pre: %d, InputDim: %d\n", post, (int)weights.size(), pre, (int)weights[0].size());
                assert(post < weights.size() && pre < weights[0].size());
            }
        }
        weights[post][pre] += weight;
    }

    for(int c = 0; c < cuW->channels; c++){
        for(int i = 0; i < cuW->rows; i++){
            for(int j = 0; j < cuW->cols; j++){
                cuW->set(i, j, c, weights[i][j]);
            }
        }
    }
    cuW->toGpu();
    // verify that the weights is correctly copied!
    for(int i = 0; i < weights.size(); ++i){
        for(int j = 0; j < weights[0].size(); ++j){
            assert(fabsf(cuW->get(i, j, 0) - weights[i][j]) < 1e-4);
        }
    }
}

//* initial the bias weights from the dumped file by the CPU sim
void Spiking::initBiasFromDumpfile(const std::string& filename, cuMatrix<float>*& cuW)
{
    ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        printf("Cannot open the file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
    assert(cuW != NULL);

    int idx; 
    float weight;
    std::string pre_name, post_name;
    while(f_in>>idx>>pre_name>>post_name>>weight){
        int pre = extractNeuronIndex(pre_name);
        int post = extractNeuronIndex(post_name);
        if(pre == inputSize && post < outputSize){ // this is related to bias
            cuW->set(post, 0, 0, weight); 
        }
    }
    cuW->toGpu();
}

void Spiking::initLaterial()
{
    ConfigSpiking* config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
    if(config->m_laterialType == "RESERVOIR"){
        initFromDumpfile(config->m_lweightPath, w_laterial);
        //initReservoirConnection(config->m_reservoirDim);
    }
    else if(config->m_laterialType == "LOCAL_INHIBITION"){
        initLocalInhibition(config->m_localInbStrength); 
    }
}

// intialize the reservoir connections
// TODO: improve the randomness of the reservoir (the bad random seed we used now!)
void Spiking::initReservoirConnection(const std::vector<int>& reservoirDim)
{
    assert(reservoirDim.size() == 3);
    assert(w_laterial != NULL);
    int d1 = reservoirDim[0], d2 = reservoirDim[1], d3 = reservoirDim[2];
    int num = d1 * d2 * d3;
    if(num != outputSize){
        printf("The reservoir dim: %d x %d x %d = %d does not match the number neuron: %d!\n",d1, d2, d3, num, outputSize);
        exit(EXIT_FAILURE);
    }
    // adopted from the CPU code:
    srand(5);
    std::vector<bool> excitatory(num, false);
    std::vector<dim3> coordinates;
    for(int i = 0; i < excitatory.size(); ++i){
        if(rand() % 100 < 20) excitatory[i] = false;
        else    excitatory[i] = true;
    }
    for(int i = 0; i < d1; ++i){
        for(int j = 0; j < d2; ++j){
            for(int k = 0; k < d3; ++k){
                int index = (i * d2 + j) * d3 + k;
                assert(index < excitatory.size());
                coordinates.push_back(dim3(i, j, k));
            }
        }
    }
    double c, a;
    double distsq, dist;
    const double factor2 = 1.5;
    for(int i = 0; i < num; ++i){
        for(int j = 0; j < num; ++j){
            if(excitatory[i]){
                if(excitatory[j]){
                    c = 0.3 * factor2;
                    a = 1;
                }
                else{
                    c = 0.2 * factor2;
                    a = 1;
                }
            }
            else{
                if(excitatory[j]){
                    c = 0.4 * factor2;
                    a = -1;
                }
                else{
                    c = 0.1 * factor2;
                    a = -1;
                }
            }
            distsq = 0;
            dist = coordinates[i].x -  coordinates[j].x;
            distsq += dist * dist;
            dist = coordinates[i].y -  coordinates[j].y;
            distsq += dist * dist;
            dist = coordinates[i].z -  coordinates[j].z;
            distsq += dist * dist;
            if(rand() % 100000 < 100000 * c * exp(-distsq / 4)){
                //printf("reservoir_%d to reservoir_%d %f\n", i , j, a);
                w_laterial->set(j, i, 0, a);
            }
        }
    }
    w_laterial->toGpu();
}

void Spiking::initLocalInhibition(float strength)
{
    assert(w_laterial != NULL);
    for(int c = 0; c < w_laterial->channels; c++){
        for(int i = 0; i < w_laterial->rows; i++){
            for(int j = 0; j < w_laterial->cols; j++){
                if(i == j)  continue;
                w_laterial->set(i, j, c, -1*strength);
            }
        }
    }
    w_laterial->toGpu();
}

/* the device function to realize: weights * spikes(:, t - 1) + recurrent_weights * o_spikes(t - 1)
 * I only consider the first order dynamics 一阶突触模型
 * inputSize  : number of input neurons
 * outputSize : number of output neurons
*/
//(inputSize, outputSize, curInput, curOutput, o_idx, w, w_l, b, t, dummyFreq, endTime);
//计算突触后膜电势
__device__ float d_Spiking_accumulate_spikes(
    int inputSize,
    int outputSize,
    float* input_response,
    bool* output,
    int o_idx,
    float* weights,
    float* weights_lat,
    float* biases,
    int t,
    int dummyFreq,//控制何时将偏差添加到响应的参数
    int endTime)
{
    int idx = threadIdx.x;
    if(idx >= outputSize * inputSize){
        return 0;
    }  
    float response = 0.0f;
    // 前向连接响应
    response = input_response[(t - 1) + o_idx * endTime];

    // 偏差响应
    if(t % dummyFreq == 0){
        response += biases[idx];
    }    

    if(weights_lat != NULL){
        // 循环连接响应，来自其他输出神经元的横向连接的输入
        for(int i = 0; i < outputSize; ++i)
            response += output[i + (t - 1) * outputSize] ? weights_lat[i + o_idx * outputSize] : 0;
    }
    return response;
}


/* given each input and output spike train, 
 * compute the accumulative synaptic effect as the gradient
 计算梯度过程中的累积突触效应delta*e_{i|j}
 * input: input spikes: endTime * inputSize
 * output: output spikes: endTime * outputSize
 */
__device__ float d_Spiking_gradient(
    bool* output,
    bool* input,
    float delta,
    int o_idx,
    int i_idx,
    int outputSize,
    int inputSize,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    float acc_response = 0.0f;
    int t_post_last = 1;
    for(int t_post = 1; t_post < endTime; t_post++){
        if(output[o_idx + t_post * outputSize] != true) continue;
        float sum = 0.0f;

        int ub = t_post;
        int lb = max(1, int(t_post - 4*TAU_M));
        //归一化突触后电势计算
        for(int t_pre = lb; t_pre < ub; ++t_pre){
            if(input[i_idx + t_pre * inputSize] != true)    continue;
            int pre_time = t_pre + T_REFRAC;
            if(pre_time > t_post)   continue;
            int s = t_post - t_post_last;
            int t = t_post - pre_time;
            float factor = exp(-1*max(t - s, 0)/TAU_S)/(1 - TAU_S/TAU_M);
            sum += factor * (exp(-1*min(s, t)/TAU_M) - exp(-1*min(s, t)/TAU_S));
        }
        
        t_post_last = t_post + T_REFRAC;
        acc_response += sum;//e_{i|j}，S-PSP
    }
    float delta_w = delta * acc_response;
    return delta_w;
 
}

/* given each input and output spike train of spike times, 
 * compute the accumulative synaptic effect as the gradient
 * input: input spikes: endTime * inputSize
 * output: output spikes: endTime * outputSize
 */
//未使用
__device__ float d_Spiking_gradient_spiketime(
    int* output_time,
    int* input_time,
    int n_ospikes,
    int n_ispikes,
    float delta,
    int o_idx,
    int i_idx,
    float lat_factor,
    int outputSize,
    int inputSize,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    
}


/* compute the gradient for the bias
 * input: input spikes: endTime * inputSize
 * output: output spikes: endTime * outputSize
 */
//未使用
__device__ float d_Spiking_bias_gradient_spiketime(
    int* output_time,
    int n_ospikes,
    float delta,
    int o_idx,
    int dummyFreq,
    int outputSize,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    float acc_response = 0.0f;
    int t_post_last = 1;
    for(int i = 0; i < n_ospikes; ++i){
        int t_post = output_time[o_idx * endTime + i];
        float sum = 0.0f;
        
        int ub = t_post;
        int lb = max(1, int(t_post - 4*TAU_M));
        for(int j = dummyFreq; j < endTime; j += dummyFreq){
            int t_pre = j;
            if(t_pre < lb || t_pre >= ub)    continue;

            int pre_time = t_pre + T_REFRAC;
            if(pre_time > t_post)   continue;
            int s = t_post - t_post_last;
            int t = t_post - pre_time;

            float factor = exp(-1*max(t - s, 0)/TAU_S)/(1 - TAU_S/TAU_M);
            sum += factor * (exp(-1*min(s, t)/TAU_M) - exp(-1*min(s, t)/TAU_S));
        }
        t_post_last = t_post + T_REFRAC;
        acc_response += sum;
    }
    float delta_b = delta * acc_response;
    return delta_b;
 
}


/*
 * dim3 block = dim3(1);
 * dim3 thread= dim3(256);
 */
//计算神经网络输出和实际标签值之间的损失，计算方式是将脉冲计数与每个类别的期望计数之间的差值平方，
//然后对每个批次中的所有神经元进行求和



__global__ void g_getCost_output(
    int*   fireCount, 
    float* groundTruth, 
    float* cost, 
    int*   y, //predict
    int    batch, 
    int    cols,//输出层的神经元数量
    float  UNDESIRED_LEVEL,//5
    float  DESIRED_LEVEL,//35
    float  MARGIN)//5
{
    extern __shared__ float _sum[];
    int len = batch * cols;
    //将整个groundTruth数组初始化为UNDESIRED_LEVEL
    for(int i = 0; i < len; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < len){
            groundTruth[id] =  UNDESIRED_LEVEL;
        }
    }
    
    __syncthreads();
    //将每个批次的正确标签的位置设为DESIRED_LEVEL
    for(int i = 0; i < batch; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < batch){
            int yy = y[id];
            groundTruth[id * cols + yy] = DESIRED_LEVEL;
        }
    }

    _sum[threadIdx.x] = 0;
    __syncthreads();

    //计算脉冲计数和期望计数之间的差值，并将其存储在_sum数组的每个线程索引处 
    for(int i = 0; i < len; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < len)
        {
            float diff = fabsf(float(fireCount[id]) - groundTruth[id]);
            _sum[threadIdx.x] += diff > MARGIN ? diff * diff : 0; 
        }
    }

    len = blockDim.x;
    //通过对_sum数组进行并行归约，计算出所有线程的成本总和，并将其存储在cost数组中
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1)>>1;
        if(threadIdx.x < skip && (threadIdx.x + skip) < len)
        {
            _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = skip;
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        cost[0] = _sum[0];
    }
}


__global__ void g_getDelta_output(float* outputDelta, int* fireCount, float* groundTruth, int len, float MARGIN)
{
    for(int i = 0; i < len; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < len)
        {
            float diff = fabsf(float(fireCount[id]) - groundTruth[id]);
            outputDelta[id] = diff > MARGIN ? fireCount[id] - groundTruth[id] : 0;
        }
    }
}






/*
__global__ void g_getCost_output(
    int*   fireCount, 
    float* groundTruth, 
    float* cost, 
    int*   y, //predict
    int    batch, 
    int    cols,//输出层的神经元数量
    float  UNDESIRED_LEVEL,//5
    float  DESIRED_LEVEL,//35
    float  MARGIN)//5
{
    extern __shared__ float _sum[];
    int len = batch * cols;
    //将整个groundTruth数组初始化为UNDESIRED_LEVEL
    for(int i = 0; i < len; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < len){
            groundTruth[id] =  UNDESIRED_LEVEL;
        }
    }
    
    __syncthreads();
    //将每个批次的正确标签的位置设为DESIRED_LEVEL
    for(int i = 0; i < batch; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < batch){
            int yy = y[id];
            groundTruth[id * cols + yy] = DESIRED_LEVEL;
        }
    }

    _sum[threadIdx.x] = 0;
    __syncthreads();

    //计算脉冲计数和期望计数之间的差值，并将其存储在_sum数组的每个线程索引处 
    for(int i = 0; i < len; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < len)
        {
            float diff = fabsf(float(fireCount[id]) - groundTruth[id]);
            // _sum[threadIdx.x] += diff > MARGIN ? diff * diff : 0; 

            float loss_mse = diff * diff;
            
            float a = 1.05;
            float b = 0.3;
            float c1 = 64;
            float c2 = 8;

            float a1 = __expf(-1 * diff * diff / c1 );
            float a2 = __expf(-1 * diff * diff / c2 );

            float loss_mcc = b * a1 + ( 1 - b ) * a2;
            float sum_loss = a * loss_mse + (1-a) * loss_mcc;

            _sum[threadIdx.x] += diff > MARGIN ? sum_loss : 0;
            
        }
    }

    len = blockDim.x;
    //通过对_sum数组进行并行归约，计算出所有线程的成本总和，并将其存储在cost数组中
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1)>>1;
        if(threadIdx.x < skip && (threadIdx.x + skip) < len)
        {
            _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = skip;
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        cost[0] = _sum[0];
    }
}

 
__global__ void g_getDelta_output(float* outputDelta, int* fireCount, float* groundTruth, int len, float MARGIN)
{
    for(int i = 0; i < len; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < len)
        {
            float diff = fabsf(float(fireCount[id]) - groundTruth[id]);
            // outputDelta[id] = diff > MARGIN ? fireCount[id] - groundTruth[id] : 0;
            float a = 1.05;
            float b = 0.3;
            float c1 = 64;
            float c2 = 8;

            float a1_grad =  2 * (fireCount[id] - groundTruth[id]) * __expf(-1 * diff * diff / c1 ) / c1 ;
            float a2_grad =  2 * (fireCount[id] - groundTruth[id]) * __expf(-1 * diff * diff / c2 ) / c2;

            float loss_mcc_grad = b * a1_grad + ( 1 - b ) * a2_grad;
            // float loss_mcc_grad = b * a1_grad;

            float sum_loss_grad = a * (fireCount[id] - groundTruth[id]) + (1-a) * loss_mcc_grad;
            outputDelta[id] = diff > MARGIN ? sum_loss_grad : 0;
           
        }
    }
}
*/







/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(outputSize);
 */
//apply the sample weights
//根据样本权重缩放误差。它通过遍历所有当前批次的样本，并使用线程索引threadIdx.x来确定当前线程负责计算哪个神经元的误差。
__global__ void g_boostWeight_output(float* outputDelta, float* sample_weights, int len)
{
    int batchId = blockIdx.x;
    float sample_weight = sample_weights[batchId];
    int outputSize = blockDim.x;
    int tid = threadIdx.x;
    int target = tid + batchId * outputSize;
    if(target < len)
        //函数使用当前样本的权重将相应的误差值进行缩放，即将误差乘以样本权重。缩放后的误差值被存储在outputDelta数组中
        outputDelta[target] *= sample_weight;

}


/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, outputSize);
 */
//计算侧向抑制因子
__global__ void g_getLateralFactor_output(
    int* outputs_time,
    int* batchFireCount,
    float w0,//lateralW
    int* y,//predict
    float* batchLFactor,//lateralFactor
    float vth,
    int outputSize,
    int endTime,
    int T_REFRAC,//不应期
    float TAU_M,
    float TAU_S)
{
    extern __shared__ float d_sum[];
    int tid = threadIdx.x;//使用线程索引threadIdx.x来确定当前线程负责计算哪一对神经元
    d_sum[tid] = 0;
    __syncthreads();

    int batchId = blockIdx.x;
    int j_idx   = blockIdx.y;
    
    int outputSize2 = endTime * outputSize;
    int* output_time = outputs_time + batchId * outputSize2;
    int* output_fireCount = batchFireCount + batchId * outputSize;
    int cls = y[batchId];

    float * lateral_factors = batchLFactor + batchId * outputSize;

    int f_cnt_j = output_fireCount[j_idx];
    float d_j = (f_cnt_j > 0 || (f_cnt_j == 0 && j_idx == cls)) ? 1 / vth : 0;

    //遍历输出层中的每对神经元
    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int l_idx = i + tid;
        if(l_idx < outputSize && j_idx != l_idx)
        {
            int f_cnt_l = output_fireCount[l_idx];//输出脉冲计数
            float d_l = (f_cnt_l > 0 || (f_cnt_l == 0 && l_idx == cls)) ? 1 / vth : 0;
            

            
            float e_jl = d_Spiking_accumulate_effect(output_time, output_time, f_cnt_l, f_cnt_j, l_idx, j_idx, outputSize, outputSize, endTime, T_REFRAC, TAU_M, TAU_S);
            float effect_ratio_jl = (f_cnt_j == 0 || f_cnt_l == 0) ? 1 : e_jl / f_cnt_j;
            float e_lj = d_Spiking_accumulate_effect(output_time, output_time, f_cnt_j, f_cnt_l, j_idx, l_idx, outputSize, outputSize, endTime, T_REFRAC, TAU_M, TAU_S);
            float effect_ratio_lj = (f_cnt_l == 0 || f_cnt_j == 0) ? 1 : e_lj / f_cnt_l;
            d_sum[tid] += effect_ratio_jl * d_l * effect_ratio_lj * d_j; 
             
            
            // float w_stdp= single_stdp(output_time,output_time,f_cnt_l, f_cnt_j, l_idx,j_idx,endTime, T_REFRAC,TAU_M,TAU_S);
            // d_sum[tid] += w_stdp;
        }
    }
    
    
    int len = blockDim.x;  
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(tid < skip && (tid + skip) < len)
        {
            d_sum[tid] += d_sum[tid + skip];//一个指向包含当前线程块计算出来的侧向抑制因子的浮点数组的指针
        }
        len = skip;
    }
    if(tid == 0)
    {
        //lateral_factors是一个指向包含当前批次的侧向抑制因子的浮点数组的指针
        lateral_factors[j_idx] = 1.0f / (1 - d_sum[0] * w0 * w0);//w0是侧向抑制的权重
    }
}
 


/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(outputSize);
 */
//未使用
__global__ void g_getMaxCount(int* fireCount, int* maxCount, int cols)
{
    extern __shared__ int _max[];
    int batchId = blockIdx.x;
    int len = blockDim.x;
    int id = threadIdx.x;
    
    _max[id] = 0;
    __syncthreads();
    
    for(int tid = 0; tid < cols; tid += blockDim.x){
        int ttid = tid + id;
        if(ttid < cols){
            _max[threadIdx.x] = max(_max[threadIdx.x], fireCount[ttid + batchId * cols]);
        }
    }

    _max[id] = fireCount[id + batchId * cols];
    while(len != 1)
    { 
        __syncthreads();
        int skip = (len + 1)>>1;
        if(id < skip && (id + skip) < len)
        {
            _max[id] = max(_max[id], _max[id + skip]);
        }
        len = skip;
    }
    __syncthreads();
    if(id == 0)
    {
        maxCount[batchId] = _max[0];
    } 
}


/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(1024, outputSize));
 */
//如果目标神经元不放电的时候，修改目标神经元的输出脉冲序列和输出脉冲放电计数,使其在目标级别时间步内发放一定数量的脉冲
__global__ void g_modifySpikes(bool* outputs, int* y, int* fireCount, int target_level, int endTime, int outputSize)
{
    //使用线程索引threadIdx.x和块索引blockIdx.x来确定当前线程负责计算哪个批次的神经元
    int batchId = blockIdx.x;
    //使用目标级别和真实标签值来确定目标神经元
    int target = y == NULL ? -1 : y[batchId];
    int mCnt = target_level; 
    //使用目标级别和真实标签值来确定目标神经元
    bool* outputSpikes = outputs + batchId * endTime * outputSize;


    for(int id = 0; id < outputSize; id += blockDim.x){
        int o_idx = id + threadIdx.x;
        if(o_idx < outputSize)
        {
            if(o_idx != target)
                return;

             //对于目标神经元，函数检查其输出脉冲计数是否为零
            if(fireCount[o_idx + batchId * outputSize] == 0)
            {
                //计算需要发放的脉冲数量和脉冲间隔，并将这些脉冲存储在outputs数组
                int count = 0;
                int interval = endTime / mCnt;
                for(int t = interval; t < endTime; t += interval)
                {
                    outputSpikes[o_idx + t * outputSize] = true;
                    count++;
                }
                //函数更新fireCount数组中目标神经元的输出脉冲计数
                fireCount[o_idx + batchId * outputSize] = count;
            }
        }
    }
}


/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(outputSize, 1024));
 */

//LIF神经元前向计算
__global__ void g_Spiking_feedforward(
    float* inputs_resp,//一个指向包含当前批次的神经元输入响应的浮点数组的指针。
	float* w,
	float* w_l,
    float* b,
	bool*  outputs,
    int* fireCount,
	int inputSize,
	int outputSize,
    int endTime,
    float vth,
    int dummyFreq, //模拟的虚拟脉冲频率
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
	int batchId = blockIdx.x;
    int outputSize2 = endTime * outputSize;

	bool* curOutput   = outputs + batchId * outputSize2;
    float* curInput   = inputs_resp + batchId * outputSize2;//inputs_resp:batch * outputSize*endTime 
    int* curFireCount = fireCount + batchId * outputSize; 

    // 模拟脉冲序列:用一个时间循环来模拟脉冲传播和响应过程
    for(int tidx = 0; tidx < outputSize; tidx += blockDim.x)
    {
        int o_idx = tidx + threadIdx.x;
        if(o_idx < outputSize)
        {
            float v  = 0.0f;
            float ep = 0.0f;
            float threshold = vth - 1e-6; // 由于响应速度快，减小了数值差异
            int t_ref= 0;
            float response = 0.0f;
            int fire_count = 0;
            for(int t = 0; t < endTime; t++){
                // 1. leakage,膜电位的漏电处理
                v  -= v / TAU_M;
                ep -= ep / TAU_S;
                if(t == 0)
                {
                    curOutput[o_idx + t * outputSize] = false;
                    continue;
                }

                // 2. 接收接收脉冲输入
                __syncthreads(); // make sure all the threads has generated the spikes for the last time step
                //weights * spikes(:, t - 1) + recurrent_weights * o_spikes(t - 1)一阶突出模型
                //根据输入脉冲和权重计算膜电位的变化量，并将其累加到膜电位上
                response = d_Spiking_accumulate_spikes(inputSize, outputSize, curInput, curOutput, o_idx, w, w_l, b, t, dummyFreq, endTime);
                
                // 3. Add up the response to ep (state variable)
                ep += response;

                // 4. Update the vmem accordingly
                v += ep/TAU_S;
                if(t_ref > 0){
                    v = 0;
                    t_ref--;
                }
            
                // 5. Fire or not
                curOutput[o_idx + t * outputSize] = v > threshold ?  true : false;
                t_ref = v > threshold ? T_REFRAC : t_ref;//T_REFRAC是不应期
                fire_count += v > threshold ? 1 : 0;
                v = v > threshold ? 0 : v;
            }
            curFireCount[o_idx] = fire_count; 
        }
    }
}


/*
 * dim3 block = dim3(batch, inputSize);
 * dim3 thread= min(1024, outputSize);
 */
//计算delta*e,计算神经元之间的权重梯度
__global__ void g_Spiking_wgrad(
        bool* inputs,//指向包含当前批次的神经元输入脉冲序列的布尔数组的指针
        bool* outputs,
        float* curDelta,//指向包含当前批次的神经元输入脉冲序列的布尔数组的指
        float* wgradTmp,//指向包含当前批次的神经元权重梯度的浮点数组的指针
        int inputSize,
        int outputSize,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S)
{
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;

    int wSize        = outputSize * inputSize;
    int inputSize2   = endTime * inputSize;
    int outputSize2  = endTime * outputSize;
    int curDeltaSize = outputSize;

    float* wgrad  = wgradTmp + batchId * wSize;
    bool* input    = inputs + batchId * inputSize2;
    bool* output   = outputs + batchId * outputSize2;
    float* cDelta = curDelta + batchId * curDeltaSize;
    
    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize)
        {
            float delta_w = d_Spiking_gradient(output, input, cDelta[o_idx], o_idx, i_idx, outputSize, inputSize, endTime, T_REFRAC, TAU_M, TAU_S);
            wgrad[i_idx + o_idx * inputSize] = delta_w;//修改后的神经元权重梯度存储在wgradTmp指针所指向的数组中
        }
    }

}

/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, inputSize);
 * 计算sum(w*e/o)/v
 */
//计算隐藏层神经元输出对当前层神经元输出的影响
__global__ void g_Spiking_wgrad_sideEffect(
        float* weights,
        int* batchFireCount,
        float* batchAccEffect,
        float  vth,
        int inputSize,
        int outputSize,
        float * batchSideEffect)
{
    int batchId = blockIdx.x;
    int o_idx = blockIdx.y;
    int tid     = threadIdx.x;
    extern __shared__ float _sum[];
    _sum[tid] = 0;
    __syncthreads();

    int wSize        = outputSize * inputSize;
    int* fireCount   = batchFireCount + batchId * outputSize;
    float* acc_effect= batchAccEffect + batchId * wSize;
    float* side_effect = batchSideEffect + batchId * outputSize;
    int o_cnt = fireCount[o_idx];

    //计算当前输出神经元的输出脉冲计数，并根据其是否为零来设置隐藏层神经元输出对当前层神经元输出
    for(int i = 0; i < inputSize; i += blockDim.x)
    {
        int idx = i + tid;
        if(idx < inputSize)
        {
            float w = weights[idx + o_idx * inputSize];
            float e = acc_effect[idx + o_idx * inputSize];
            //隐藏层反向传播计算，公式23，e对o求梯度，在解耦中表示，e/i
            float ratio = o_cnt == 0 ? 0.5 : e/o_cnt;
            _sum[tid] += w * ratio;
        }
    }
    int len = blockDim.x;
    //函数根据输入权重、权重梯度累加值和输出神经元的输出脉冲计数计算当前输出神经元对当前神经元输出的影响
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(tid < skip && (tid + skip) < len)
        {
            _sum[tid] += _sum[tid + skip];
        }
        len = skip;
    }
    if(tid == 0){
        side_effect[o_idx] = _sum[0]/vth;
    }
}


/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, inputSize);
 */
//计算神经元之间的权重梯度
__global__ void g_Spiking_wgrad_spiketime(
        float* batchSideEffect,
        float* batchAccEffect,
        float* curDelta,
        float* latFactor,
        float* wgradTmp,
        int inputSize,
        int outputSize)
{
    int batchId = blockIdx.x;
    int o_idx   = blockIdx.y;
    int tid     = threadIdx.x;
    
    int wSize        = outputSize * inputSize;
    int curDeltaSize = outputSize;

    float* wgrad  = wgradTmp + batchId * wSize;
    float* acc_effect     = batchAccEffect + batchId * wSize;
    float* side_effect = batchSideEffect + batchId * outputSize;
    float* cDelta = curDelta + batchId * curDeltaSize;
    float* lFactor = latFactor == NULL ? NULL : latFactor + batchId * curDeltaSize;

    float s_effect = side_effect[o_idx];
    float latFac = lFactor == NULL ? 1.0f : lFactor[o_idx];
    float delta = cDelta[o_idx];

    //使用一个时间循环来计算神经元之间的权重梯度
    for(int i = 0; i < inputSize; i += blockDim.x)
    {
        int i_idx = i + tid;
        if(i_idx < inputSize)
        {
            //首先根据当前神经元的侧面效应、潜伏期因子和误差项计算当前神经元对目标神经元的影响
            float compen_effect = acc_effect[i_idx + o_idx * inputSize] * (1 + s_effect);
            //根据累加的权重梯度值和当前神经元对目标神经元的影响计算当前神经元和目标神经元之间的权重梯度
            float delta_w = delta * compen_effect * latFac;
            //将权重梯度其存储在wgradTmp数组中相应位置上
            wgrad[i_idx + o_idx * inputSize] = delta_w;
        }
    }
}

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(1024, outputSize));
 */
//未使用
__global__ void g_Spiking_bgrad_spiketime(
        int* outputs_time,
        int* batchFireCount,
        float* curDelta,
        float* bgradTmp,
        int outputSize,
        int endTime,
        int dummyFreq,
        int T_REFRAC,
        float TAU_M,
        float TAU_S)
{
    int batchId = blockIdx.x;

    int bSize = outputSize;
    int outputSize2  = endTime * outputSize;
    int curDeltaSize = outputSize;

    float* bgrad  = bgradTmp + batchId * bSize;
    int* output_time      = outputs_time + batchId * outputSize2;
    int* output_fireCount = batchFireCount + batchId * outputSize;
    float* cDelta = curDelta + batchId * curDeltaSize;
    
    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize)
        {
            float delta_b = d_Spiking_bias_gradient_spiketime(output_time, output_fireCount[o_idx], cDelta[o_idx], o_idx, dummyFreq, outputSize, endTime, T_REFRAC, TAU_M, TAU_S);
            bgrad[o_idx] = delta_b;
        }
    }

}


/*
 * dim3 block = dim3(batch, inputSize);
 * dim3 thread= min(1024, outputSize);
 */
//计算神经元之间的影响，即突触效应
// __global__ void g_Spiking_synaptic_effect(
//         int* inputs_time,
//         int* outputs_time,
//         int* batchPreFireCount,
//         int* batchFireCount,
//         float* w,
//         float* batchAccEffect,
//         float* effectRatio,//指向包含当前批次的神经元突触效应比率的浮点数组的指针。如果不需要计算比率，则为NULL
//         int inputSize,
//         int outputSize,
//         int endTime,
//         int T_REFRAC,
//         float TAU_M,
//         float TAU_S)
// {
//     int batchId = blockIdx.x;
//     int i_idx   = blockIdx.y;

//     int wSize        = outputSize * inputSize;
//     int inputSize2   = endTime * inputSize;
//     int outputSize2  = endTime * outputSize;

//     int* input_time       = inputs_time + batchId * inputSize2;
//     int* output_time      = outputs_time + batchId * outputSize2;
//     int* input_fireCount  = batchPreFireCount + batchId * inputSize;
//     int* output_fireCount = batchFireCount + batchId * outputSize;
//     float* acc_effect     = batchAccEffect + batchId * wSize;

//     for(int i = 0; i < outputSize; i += blockDim.x)
//     {
//         int o_idx = i + threadIdx.x;
//         if(o_idx < outputSize)
//         {
//             float e = d_Spiking_accumulate_effect(output_time, input_time, output_fireCount[o_idx], input_fireCount[i_idx], o_idx, i_idx, outputSize, inputSize, endTime, T_REFRAC, TAU_M, TAU_S);
//             acc_effect[i_idx + o_idx * inputSize] = e;
//             if(effectRatio != NULL){
//                 int o_cnt = output_fireCount[o_idx];
//                 int i_cnt = input_fireCount[i_idx];
//                 float ratio = i_cnt == 0 || o_cnt == 0 ? 1 : e / float(i_cnt);
//                 effectRatio[i_idx + o_idx * inputSize] = ratio * w[i_idx + o_idx * inputSize];
//             }
//         }
//     }
// }
__global__ void g_Spiking_synaptic_effect(
        int* inputs_time,
        int* outputs_time,
        int* batchPreFireCount,
        int* batchFireCount,
        float* w,
        float* batchAccEffect,
        float* effectRatio,//指向包含当前批次的神经元突触效应比率的浮点数组的指针。如果不需要计算比率，则为NULL
        int inputSize,
        int outputSize,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S)
{
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;

    int wSize        = outputSize * inputSize;
    int inputSize2   = endTime * inputSize;
    int outputSize2  = endTime * outputSize;

    int* input_time       = inputs_time + batchId * inputSize2;
    int* output_time      = outputs_time + batchId * outputSize2;
    int* input_fireCount  = batchPreFireCount + batchId * inputSize;
    int* output_fireCount = batchFireCount + batchId * outputSize;
    float* acc_effect     = batchAccEffect + batchId * wSize;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize)
        {
            float e = d_Spiking_accumulate_effect(output_time, input_time, output_fireCount[o_idx], input_fireCount[i_idx], o_idx, i_idx, outputSize, inputSize, endTime, T_REFRAC, TAU_M, TAU_S);
            acc_effect[i_idx + o_idx * inputSize] = e;
            // if(effectRatio != NULL){
            //     int o_cnt = output_fireCount[o_idx];
            //     int i_cnt = input_fireCount[i_idx];
            //     float ratio = i_cnt == 0 || o_cnt == 0 ? 1 : e / float(i_cnt);          
            //     effectRatio[i_idx + o_idx * inputSize] = ratio * w[i_idx + o_idx * inputSize];
            // }
            float w_stdp = single_stdp(output_time, input_time, output_fireCount[o_idx], input_fireCount[i_idx], o_idx, i_idx, endTime, T_REFRAC, TAU_M, TAU_S);
            int o_cnt = output_fireCount[o_idx];
            int i_cnt = input_fireCount[i_idx];
            float ratio = i_cnt == 0 || o_cnt == 0 ? 1 : w_stdp / float(i_cnt);
            effectRatio[i_idx + o_idx * inputSize] = ratio * w[i_idx + o_idx * inputSize];
        }
    }
}
          

/*
 * dim3 block = dim3(batch, inputSize);
 * dim3 thread= min(1024, outputSize);
 */
__global__ void g_Spiking_debug_spiketime(
        int* inputs_time,
        int* outputs_time,
        int* batchPreFireCount,
        int* batchFireCount,
        int inputSize,
        int outputSize,
        int endTime)
{
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;

    int inputSize2   = endTime * inputSize;
    int outputSize2  = endTime * outputSize;
 
    int* input_time       = inputs_time + batchId * inputSize2;
    int* output_time      = outputs_time + batchId * outputSize2;
    int* input_fireCount  = batchPreFireCount + batchId * outputSize;
    int* output_fireCount = batchFireCount + batchId * outputSize;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize)
        {
            if(i_idx == I_IDX && o_idx == O_IDX){
                printf("Input %d fires: ", i_idx);
                for(int i = 0; i < input_fireCount[i_idx]; i++)    printf("%d\t", input_time[i_idx * endTime + i]);
                printf("\n");
                printf("Output %d fires: ", o_idx);
                for(int j = 0; j < output_fireCount[o_idx]; j++)    printf("%d\t", output_time[o_idx * endTime + j]);
                printf("\n");
            }
        }
    } 
}
