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

void forwardPart1(const VecD& xt, const State* prev, State* curr, Master* master);
void forwardPart2(const VecD& xt, const State* prev, State* curr, Master* master);

#endif

