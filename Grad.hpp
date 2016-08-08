/*************************************************************************
    > File Name: Grad.hpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月04日 星期四 22时09分42秒
 ************************************************************************/

#ifndef _GRAD_HPP_
#define _GRAD_HPP_
#include "Include.hpp"

class Grad
{
public:
	Grad(){}
	Grad(const int input_dim, const int hidden_dim);
	void init();	
	MatD Wxi, Whi; VecD bi;
	MatD Wxf, Whf; VecD bf;
	MatD Wxo, Who; VecD bo;
	MatD Wxu, Whu; VecD bu;

};

#endif
