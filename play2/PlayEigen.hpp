/*************************************************************************
    > File Name: PlayEigen.hpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月16日 星期二 11时38分03秒
 ************************************************************************/

#ifndef _PLAYEIGEN_HPP_
#define _PLAYEIGEN_HPP_

#include "Include.cpp"

double playCurrTime();
void playGenRandInt(std::vector<int>& array, int len, int min, int max);
void playGenRandIntDynamic(std::vector<int>& array, int len, int min, int max);
double playGetMeanIntArray(std::vector<int>& array);

void playEigen();
#endif

