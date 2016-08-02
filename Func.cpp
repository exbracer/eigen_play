/*************************************************************************
    > File Name: Func.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 12时24分33秒
 ************************************************************************/

#include "Func.hpp"

void forwardPart1(const VecD& xt, const State* prev, State* curr, Master* master)
{
	curr->i = master->bi;
	curr->i.noalias() += master->Wxi*xt + master->Whi*prev->h;
	curr->f = master->bf;
	curr->f.noalias() += master->Wxf*xt + master->Whf*prev->h;
	curr->o = master->bo;
	curr->o.noalias() += master->Wxo*xt + master->Who*prev->h;
	curr->u = master->bu;
	curr->u.noalias() += master->Wxu*xt + master->Whu*prev->h;
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


