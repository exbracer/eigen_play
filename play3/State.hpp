/*************************************************************************
    > File Name: State.hpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 14时33分55秒
 ************************************************************************/
#ifndef _STATE_HPP_
#define _STATE_HPP_
#include "Include.hpp"

class State 
{
public:
	virtual ~State() 
	{
		this->clear();
	}
	 
	VecD h, c, u, i, f, o;
	VecD cTanh;

	VecD delh, delc, delx, dela; // for backward propagation

	virtual void clear();

};

#endif
