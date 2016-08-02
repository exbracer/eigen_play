/*************************************************************************
    > File Name: Master.hpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 17时24分01秒
 ************************************************************************/

#ifndef _MASTER_HPP_
#define _MASTER_HPP_

#include "Include.hpp"
class Master
{
public:
	Master(){};
	Master(const int input_dim, const int hidden_dim);
	
	MatD Wxi, Whi; VecD bi; // for the input gate
	MatD Wxf, Whf; VecD bf; // for the forget gate
	MatD Wxo, Who; VecD bo; // for the output gate
	MatD Wxu, Whu; VecD bu; // for the memory cell
};

#endif

