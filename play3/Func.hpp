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

double curTime();
void genRandInt(std::vector<int>& array, int len, int min, int max);
void genRandIntDynamic(std::vector<int>& array, int len, int min, int max);
double getMeanIntArray(std::vector<int>& array);

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

void newBackward(State* prev, State* curr, Master* master, VecD& delo, VecD& deli, VecD& delu, VecD& delf);

void newSoftmaxCalcDist(const VecD& input, VecD& output, Softmax* sharedSoftmax);
Real newSoftmaxCalcLoss(VecD& output, const int label);
void newSoftmaxBackward(VecD& output, const int label, Softmax* privateSoftmax);

void newSoftmaxBackwardPart1(VecD& output, const int label, Softmax* privateSoftmax);
void newSoftmaxBackwardPart2(VecD& output, VecD& deltaFeature, Softmax* sharedSoftmax);
void newSoftmaxBackwardPart3(VecD& input, VecD& output, Softmax* privateSoftmax, int index);

Real newSoftmaxOperation(const VecD& input, VecD& output, const int label, VecD& deltaFeature, Softmax* sharedSoftmax, Softmax* privateSoftmax, int setSize);

#endif

