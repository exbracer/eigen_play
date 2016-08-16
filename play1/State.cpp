/*************************************************************************
    > File Name: State.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 14时42分10秒
 ************************************************************************/
#include "State.hpp"

void State::clear() 
{
	this->h = VecD();
	this->c = VecD();
	this->u = VecD();
	this->i = VecD();
	this->f = VecD();
	this->o = VecD();
	this->cTanh = VecD();
	this->delh = VecD();
	this->delc = VecD();
	this->delx = VecD();
	this->dela = VecD();
}

