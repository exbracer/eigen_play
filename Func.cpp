/*************************************************************************
    > File Name: Func.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 12时24分33秒
 ************************************************************************/

#include "Func.hpp"

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

void backward(State* prev, State* curr, Grad& grad, const VecD& xt)
{

	return;
}

void backwardPart1(State* prev, State* curr, Grad& grad, const VecD& xt)
{

}

void backwardPart2(State* prev, State* curr, Grad& grad, const VecD& xt)
{

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
	deltaFeature = softmax->grad_weight.transpose()*delta;
	softmax->grad_weight += delta*input.transpose();
	softmax->grad_bias += delta;
}
