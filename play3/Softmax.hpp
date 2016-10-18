/*************************************************************************
    > File Name: Softmax.hpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月05日 星期五 01时43分36秒
 ************************************************************************/

#ifndef _SOFTMAX_HPP_
#define _SOFTMAX_HPP_
#include "Include.hpp"

class Softmax
{
public:
	Softmax(){

	};
	Softmax(const int input_dim, const int class_num):
		weight(MatD::Zero(class_num, input_dim)), bias(VecD::Zero(class_num))
	{
		grad_weight = MatD::Zero(weight.rows(), weight.cols());
		grad_bias = VecD::Zero(bias.rows());
	};
	MatD weight; VecD bias;
	MatD grad_weight; VecD grad_bias;
};

#endif
