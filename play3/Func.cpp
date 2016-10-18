/*************************************************************************
    > File Name: Func.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 12时24分33秒
 ************************************************************************/

#include "Func.hpp"
double curTime()
{
	struct timespec tp[1];
	int r = clock_gettime(CLOCK_REALTIME, tp);

	assert(r==0);

	return tp->tv_sec + tp->tv_nsec * 1.0e-9;
}

void genRandInt(std::vector<int>& array, int len, int min, int max)
{
	srand(19920403);
	for (int i = 0; i < len; i ++)
	{
		array.push_back((rand()%(max-min+1))+min);
	}

	return;
}

void genRandIntDynamic(std::vector<int>& array, int len, int min, int max)
{
	for (int i = 0; i < len; i ++)
	{
		array.push_back((rand()%(max-min+1))+min);
	}

	return;
}

double getMeanIntArray(std::vector<int>& array)
{
	int sum = 0;
	for (int i = 0; i < array.size(); i ++)
	{
		sum += array[i];
	}

	return (double)sum/array.size();
}
void forward(const VecD& xt, const State* prev, State* curr, Master* master)
{
	// Part 1
	curr->i = master->bi;
	curr->i.noalias() += master->Wxi*xt + master->Whi*prev->h;
	curr->f = master->bf;
	curr->f.noalias() += master->Wxf*xt + master->Whf*prev->h;
	curr->o = master->bo;
	curr->o.noalias() += master->Wxo*xt + master->Who*prev->h;
	curr->u = master->bu;
	curr->u.noalias() += master->Wxu*xt + master->Whu*prev->h;

	// Part 2
	ActFunc::logistic(curr->i);
	ActFunc::logistic(curr->f);
	ActFunc::logistic(curr->o);
	ActFunc::tanh(curr->u);

	curr->c = curr->i.array()*curr->u.array() + curr->f.array()*prev->c.array();
	curr->cTanh = curr->c;
	ActFunc::tanh(curr->cTanh);
	curr->h = curr->o.array()*curr->cTanh.array();

	return;
}
void forwardPart1(const VecD& xt, const State* prev, State* curr, Master* master)
{
	// std::cout << "forwardPart1 start " << std::endl;
	curr->i = master->bi;
	curr->i.noalias() += master->Wxi*xt + master->Whi*prev->h;
	// std::cout << "forwardPart1 I: " << std::endl;
	curr->f = master->bf;
	curr->f.noalias() += master->Wxf*xt + master->Whf*prev->h;
	// std::cout << "forwardPart1 II: " << std::endl;
	curr->o = master->bo;
	curr->o.noalias() += master->Wxo*xt + master->Who*prev->h;
	// std::cout << "forwardPart III: " << std::endl;
	curr->u = master->bu;
	curr->u.noalias() += master->Wxu*xt + master->Whu*prev->h;
	// std::cout << "forwardPart IV: " << std::endl;
	// std::cout << "forwardPart1 end" << std::endl;
}

void forwardPart2(const VecD& xt, const State* prev, State* curr, Master* master)
{
	ActFunc::logistic(curr->i);
	ActFunc::logistic(curr->f);
	ActFunc::logistic(curr->o);
	ActFunc::tanh(curr->u);
	
	curr->c = curr->i.array()*curr->u.array() + curr->f.array()*prev->c.array();
	curr->cTanh = curr->c;
	ActFunc::tanh(curr->cTanh);
	curr->h = curr->o.array()*curr->cTanh.array();
}

void backward(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master)
{
	VecD delo, deli, delu, delf;

	curr->delc.array() += ActFunc::tanhPrime(curr->cTanh).array() * curr->delh.array() * curr->o.array();
	prev->delc.array() += curr->delc.array() * curr->f.array();

	delo = ActFunc::logisticPrime(curr->o).array() * curr->delh.array() * curr->cTanh.array();
	deli = ActFunc::logisticPrime(curr->i).array() * curr->delc.array() * curr->u.array();
	delf = ActFunc::logisticPrime(curr->f).array() * curr->delc.array() * prev->c.array();
	delu = ActFunc::tanhPrime(curr->u).array() * curr->delc.array() * curr->i.array();

	curr->delx.noalias() = 
		master->Wxi.transpose() * deli + 
		master->Wxf.transpose() * delf + 
		master->Who.transpose() * delo + 
		master->Wxu.transpose() * delu;

	prev->delh.noalias() += 
		master->Whi.transpose() * deli + 
		master->Whf.transpose() * delf + 
		master->Who.transpose() * delo + 
		master->Whu.transpose() * delu;
	
	grad.Wxi.noalias() += deli * xt.transpose();
	
	grad.Whi.noalias() += deli * prev->h.transpose();

	grad.Wxf.noalias() += delf * xt.transpose();
	grad.Whf.noalias() += delf * prev->h.transpose();

	grad.Wxo.noalias() += delo * xt.transpose();
	grad.Who.noalias() += delo * prev->h.transpose();

	grad.Wxu.noalias() += delu * xt.transpose();
	grad.Whu.noalias() += delu * prev->h.transpose();
	
	grad.bi += deli;
	grad.bf += delf;
	grad.bo += delo;
	grad.bu += delu;
	
	return;
}

void newBackward(State* prev, State* curr,  Master* master, VecD& delo, VecD& deli, VecD& delu, VecD& delf)
{
	// VecD delo, deli, delu, delf;

	curr->delc.array() += ActFunc::tanhPrime(curr->cTanh).array() * curr->delh.array() * curr->o.array();
	prev->delc.array() += curr->delc.array() * curr->f.array();

	delo = ActFunc::logisticPrime(curr->o).array() * curr->delh.array() * curr->cTanh.array();
	deli = ActFunc::logisticPrime(curr->i).array() * curr->delc.array() * curr->u.array();
	delf = ActFunc::logisticPrime(curr->f).array() * curr->delc.array() * prev->c.array();
	delu = ActFunc::tanhPrime(curr->u).array() * curr->delc.array() * curr->i.array();

	curr->delx.noalias() = 
		master->Wxi.transpose() * deli + 
		master->Wxf.transpose() * delf + 
		master->Who.transpose() * delo + 
		master->Wxu.transpose() * delu;

	prev->delh.noalias() += 
		master->Whi.transpose() * deli + 
		master->Whf.transpose() * delf + 
		master->Who.transpose() * delo + 
		master->Whu.transpose() * delu;
/*	
	grad.Wxi.noalias() += deli * xt.transpose();
	
	grad.Whi.noalias() += deli * prev->h.transpose();

	grad.Wxf.noalias() += delf * xt.transpose();
	grad.Whf.noalias() += delf * prev->h.transpose();

	grad.Wxo.noalias() += delo * xt.transpose();
	grad.Who.noalias() += delo * prev->h.transpose();

	grad.Wxu.noalias() += delu * xt.transpose();
	grad.Whu.noalias() += delu * prev->h.transpose();
	
	grad.bi += deli;
	grad.bf += delf;
	grad.bo += delo;
	grad.bu += delu;
	*/	
	return;
}


void backwardPart1(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master)
{
	// std::cout << "backward part 1 start " << std::endl;
	VecD deli, delf, delo, delu;
	
	curr->delc.array() += ActFunc::tanhPrime(curr->cTanh).array()*curr->delh.array()*curr->o.array();
	// std::cout << "hahha" << std::endl;
	prev->delc.array() += curr->delc.array()*curr->f.array();
	// std::cout << "backward part 1 middle" << std::endl;
	deli = ActFunc::logisticPrime(curr->i).array() * curr->delc.array() * curr->u.array();
	// std::cout << "after deli" << std::endl;
	delf = ActFunc::logisticPrime(curr->f).array() * curr->delc.array() * prev->c.array();
	// std::cout << "after delf" << std::endl;
	delo = ActFunc::logisticPrime(curr->o).array() * curr->delh.array() * curr->cTanh.array();
	// std::cout << "after delo" << std::endl;
	delu = ActFunc::tanhPrime(curr->u).array() * curr->delc.array() * curr->i.array();
	// std::cout << "after delu" << std::endl;
	return;
}

void backwardPart2(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master, int dim)
{
	VecD deli = VecD::Random(dim);
	VecD delf = VecD::Random(dim);
	VecD delo = VecD::Random(dim);
	VecD delu = VecD::Random(dim);

	curr->delx.noalias() = 
		master->Wxi.transpose() * deli +
		master->Wxf.transpose() * delf +
		master->Wxo.transpose() * delo + 
		master->Wxu.transpose() * delu;

	curr->delh.noalias() = 
		master->Whi.transpose() * deli +
		master->Whf.transpose() * delf + 
		master->Who.transpose() * delo + 
		master->Whu.transpose() * delu;

	return;
}

void backwarPart3(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master, int dim)
{
	VecD deli = VecD::Random(dim);
	VecD delf = VecD::Random(dim);
	VecD delo = VecD::Random(dim);
	VecD delu = VecD::Random(dim);

	grad.Wxi.noalias() += deli * xt.transpose();
	grad.Whi.noalias() += deli * prev->h.transpose();
	
	grad.Wxf.noalias() += delf * xt.transpose();
	grad.Whf.noalias() += delf * prev->h.transpose();

	grad.Wxo.noalias() += delo * xt.transpose();
	grad.Who.noalias() += delo * prev->h.transpose();

	grad.Wxu.noalias() += delu * xt.transpose();
	grad.Whu.noalias() += delu * prev->h.transpose();

	grad.bi += deli;
	grad.bf += delf;
	grad.bo += delo;
	grad.bu += delu;

	return;
}

void backwardVersion2Part1(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master)
{
	// std::cout << "backward part 1 start " << std::endl;
	VecD deli, delf, delo, delu;
	
	curr->delc.array() += ActFunc::tanhPrime(curr->cTanh).array()*curr->delh.array()*curr->o.array();
	// std::cout << "hahha" << std::endl;
	prev->delc.array() += curr->delc.array()*curr->f.array();
	// std::cout << "backward part 1 middle" << std::endl;
	deli = ActFunc::logisticPrime(curr->i).array() * curr->delc.array() * curr->u.array();
	// std::cout << "after deli" << std::endl;
	delf = ActFunc::logisticPrime(curr->f).array() * curr->delc.array() * prev->c.array();
	// std::cout << "after delf" << std::endl;
	delo = ActFunc::logisticPrime(curr->o).array() * curr->delh.array() * curr->cTanh.array();
	// std::cout << "after delo" << std::endl;
	delu = ActFunc::tanhPrime(curr->u).array() * curr->delc.array() * curr->i.array();
	// std::cout << "after delu" << std::endl;

	curr->delx.noalias() = 
		master->Wxi.transpose() * deli +
		master->Wxf.transpose() * delf +
		master->Wxo.transpose() * delo + 
		master->Wxu.transpose() * delu;

	curr->delh.noalias() = 
		master->Whi.transpose() * deli +
		master->Whf.transpose() * delf + 
		master->Who.transpose() * delo + 
		master->Whu.transpose() * delu;
}
void backwardVersion3Part1(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master)
{

	VecD deli, delf, delo, delu;
	
	curr->delc.array() += ActFunc::tanhPrime(curr->cTanh).array()*curr->delh.array()*curr->o.array();
	// std::cout << "hahha" << std::endl;
	prev->delc.array() += curr->delc.array()*curr->f.array();
	// std::cout << "backward part 1 middle" << std::endl;
	deli = ActFunc::logisticPrime(curr->i).array() * curr->delc.array() * curr->u.array();
	// std::cout << "after deli" << std::endl;
	delf = ActFunc::logisticPrime(curr->f).array() * curr->delc.array() * prev->c.array();
	// std::cout << "after delf" << std::endl;
	delo = ActFunc::logisticPrime(curr->o).array() * curr->delh.array() * curr->cTanh.array();
	// std::cout << "after delo" << std::endl;
	delu = ActFunc::tanhPrime(curr->u).array() * curr->delc.array() * curr->i.array();
	// std::cout << "after delu" << std::endl;
	
	grad.Wxi.noalias() += deli * xt.transpose();
	grad.Whi.noalias() += deli * prev->h.transpose();
	
	grad.Wxf.noalias() += delf * xt.transpose();
	grad.Whf.noalias() += delf * prev->h.transpose();

	grad.Wxo.noalias() += delo * xt.transpose();
	grad.Who.noalias() += delo * prev->h.transpose();

	grad.Wxu.noalias() += delu * xt.transpose();
	grad.Whu.noalias() += delu * prev->h.transpose();

	grad.bi += deli;
	grad.bf += delf;
	grad.bo += delo;
	grad.bu += delu;
}
void softmaxCalcDist(const VecD& input, VecD& output, Softmax* softmax)
{
	output = softmax->bias;
	output.noalias() += softmax->weight*input;
	output.array() -= output.maxCoeff();
	output = output.array().exp();
	output /= output.array().sum();
}

Real softmaxCalcLoss(const VecD& output, const int label)
{
	return -log(output.coeff(label, 0));	
}

void softmaxBackward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* softmax)
{
	VecD delta = output;

	delta.coeffRef(label, 0) -= 1.0;
	deltaFeature = softmax->grad_weight.transpose()*delta; // hotpot 1
	softmax->grad_weight += delta*input.transpose(); // hotpot 2
	softmax->grad_bias += delta;
}

void softmaxBackward1(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* softmax)
{	
	VecD delta = output;
	
	delta.coeffRef(label, 0) -= 1.0;
	//deltaFeature = softmax->grad_weight.transpose()*delta;
	deltaFeature = (delta.transpose()*softmax->grad_weight).transpose();
	//softmax->grad_weight += delta*input.transpose();
	softmax->grad_bias += delta;
}

void softmaxBackward2(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* softmax)
{
	deltaFeature = VecD::Random(512);
}

void softmaxBackward3(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* softmax)
{
	VecD delta = output;
	delta.coeffRef(label, 0) -= 1.0;
	deltaFeature = softmax->grad_weight.transpose()*delta;

	softmax->grad_bias += delta;
}

void softmaxBackward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* sharedSoftmax, Softmax* privateSoftmax)
{
	VecD delta = output;
	
	delta.coeffRef(label, 0) -= 1.0;
	deltaFeature = sharedSoftmax->grad_weight.transpose()*delta;
	privateSoftmax->grad_weight += delta*input.transpose();
	privateSoftmax->grad_bias += delta;
}

Real softmaxOperation(const VecD& input, VecD& output, const int label, VecD& deltaFeature, Softmax* sharedSoftmax, Softmax* privateSoftmax)
{
	
	output = sharedSoftmax->bias;
	output.noalias() += sharedSoftmax->weight*input;
	output.array() -= output.maxCoeff();
	output = output.array().exp();
	output /= output.array().sum();
	
	//Real loss = 0.0;
		
	Real loss = -log(output.coeff(label, 0));
	
	output.coeffRef(label, 0) -= 1.0;
	//deltaFeature = sharedSoftmax->grad_weight.transpose()*output;
	deltaFeature.noalias() = sharedSoftmax->grad_weight.transpose()*output;
	//privateSoftmax->grad_weight += output*input.transpose();
	privateSoftmax->grad_weight.noalias() += output*input.transpose();
	//privateSoftmax->grad_bias += output;
	privateSoftmax->grad_bias.noalias() += output;
	

	return loss;
}

Real softmaxOperation(const VecD& input, const int label, VecD& deltaFeature, Softmax* sharedSoftmax, Softmax* privateSoftmax)
{

	VecD output = sharedSoftmax->bias;
	output.noalias() += sharedSoftmax->weight*input;
	output.array() -= output.maxCoeff();
	output = output.array().exp();
	output /= output.array().sum();
	
	// Real loss = 0.0;
	
	Real loss = -log(output.coeff(label, 0));
	
	output.coeffRef(label, 0) -= 1.0;
	deltaFeature = sharedSoftmax->grad_weight.transpose()*output;
	privateSoftmax->grad_weight += output*input.transpose();
	privateSoftmax->grad_bias += output;

	return loss;
}

// for new version
void newSoftmaxCalcDist(const VecD& input, VecD& output, Softmax* sharedSoftmax)
{
    output = sharedSoftmax->bias;
    output.noalias() += sharedSoftmax->weight*input;
    output.array() -= output.maxCoeff();
    output = output.array().exp();
    output /= output.array().sum();

    return;
}

Real newSoftmaxCalcLoss(VecD& output, const int label)
{
    return -log(output.coeff(label, 0));
}

void newSoftmaxBackward(VecD& output, const int label, Softmax* privateSoftmax)
{
    output.coeffRef(label, 0) -= 1.0;

    return;
}

void newSoftmaxBackwardPart1(VecD& output, const int label, Softmax* privateSoftmax)
{
    output.coeffRef(label, 0) -= 1.0;
    privateSoftmax->grad_bias += output;
}

void newSoftmaxBackwardPart2(VecD& output, VecD& deltaFeature, Softmax* sharedSoftmax)
{
    deltaFeature = sharedSoftmax->grad_weight.transpose() * output;
}

void newSoftmaxBackwardPart3(VecD& input, VecD& output, Softmax* privateSoftmax, int index)
{
    privateSoftmax->grad_weight.col(index) += output*input(index, 0);     
}
Real newSoftmaxOperation(const VecD& input, VecD& output, const int label, VecD& deltaFeature, Softmax* sharedSoftmax, Softmax* privateSoftmax, int setSize)
{



}



