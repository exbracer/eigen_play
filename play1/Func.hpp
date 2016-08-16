/*************************************************************************
    > File Name: Func.hpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 12时24分33秒
 ************************************************************************/
#ifndef _FUNC_HPP_
#define _FUNC_HPP_
#include "Include.hpp"
#include "State.hpp"
#include "Master.hpp"
#include "Grad.hpp"
#include "Softmax.hpp"


void forward(const VecD& xt, const State* prev, State* curr, Master* master);
void forwardPart1(const VecD& xt, const State* prev, State* curr, Master* master);
void forwardPart2(const VecD& xt, const State* prev, State* curr, Master* master);
void backward(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master);
void backwardPart1(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master);
void backwardPart2(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master, int dim);
void backwardPart3(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master, int dim);
void backwardVersion2Part1(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master);
void backwardVersion3Part1(State* prev, State* curr, Grad& grad, const VecD& xt, Master* master);
void softmaxCalcDist(const VecD& input, VecD& output, Softmax* softmax);
Real softmaxCalcLoss(const VecD& output, const int label);
void softmaxBackward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* softmax);
void softmaxBackward1(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* softmax);
void softmaxBackward2(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* softmax);
void softmaxBackward3(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* softmax);
void softmaxBackward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, Softmax* sharedSoftmax, Softmax* privateSoftmax);
Real softmaxOperation(const VecD& input, VecD& output, const int label, VecD& deltaFeature, Softmax* sharedSoftmax, Softmax* privateSoftmax);
Real softmaxOperation(const VecD& input, const int label, VecD& deltaFeature, Softmax* sharedSoftmax, Softmax* privateSoftmax);

#endif

