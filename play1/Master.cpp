/*************************************************************************
    > File Name: Master.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 17时30分00秒
 ************************************************************************/

#include "Master.hpp"

Master::Master(const int input_dim, const int hidden_dim)
{
	this->Wxi = MatD(hidden_dim, input_dim);
	this->Whi = MatD(hidden_dim, hidden_dim);
	this->bi = VecD::Zero(hidden_dim);

	this->Wxf = MatD(hidden_dim, input_dim);
	this->Whf = MatD(hidden_dim, hidden_dim);
	this->bf = VecD::Zero(hidden_dim);

	this->Wxo = MatD(hidden_dim, input_dim);
	this->Who = MatD(hidden_dim, hidden_dim);
	this->bo = VecD::Zero(hidden_dim);

	this->Wxu = MatD(hidden_dim, input_dim);
	this->Whu = MatD(hidden_dim, hidden_dim);
	this->bu = VecD::Zero(hidden_dim);
}

void Master::init(Rand& rnd, const Real scale)
{
	rnd.uniform(this->Wxi, scale);
	rnd.uniform(this->Whi, scale);

	rnd.uniform(this->Wxf, scale);
	rnd.uniform(this->Whf, scale);

	rnd.uniform(this->Wxo, scale);
	rnd.uniform(this->Who, scale);

	rnd.uniform(this->Wxo, scale);
	rnd.uniform(this->Who, scale);
	
	rnd.uniform(this->Wxu, scale);
	rnd.uniform(this->Whu, scale);

	return;
}
