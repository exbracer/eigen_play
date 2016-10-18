/*************************************************************************
    > File Name: Grad.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月08日 星期一 16时07分11秒
 ************************************************************************/

#include "Grad.hpp"

Grad::Grad(const int input_dim, const int hidden_dim)
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

void Grad::init()
{
	this->Wxi.setZero();
	this->Whi.setZero();
	this->bi.setZero();

	this->Wxf.setZero();
	this->Whf.setZero();
	this->bi.setZero();

	this->Wxo.setZero();
	this->Who.setZero();
	this->bo.setZero();

	this->Wxu.setZero();
	this->Whu.setZero();
	this->bu.setZero();
}
