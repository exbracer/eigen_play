/*************************************************************************
    > File Name: play_eigen.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月16日 星期二 11时46分35秒
 ************************************************************************/

#include "PlayEigen.hpp"

double playCurrTime()
{
	struct timespec tp[1];
	int r = clock_gettime(CLOCK_REALTIME, tp);
	assert(r == 0);

	return tp->tv_sec + tp->tv_nsec * 1.0e-9;
}

void playGenRandInd(std::vector<int>& array, int len, int min, int max)
{
	srand(19920403);
	for (int i = 0; i < len; i ++)
	{
		array.push_back((rand()%(max-min+1))+min);
	}

	return;
}

void playGenRandIntDynamic(std::vector<int>& array, int len, int min, int max)
{
	for (int i = 0; i < len; i ++)
	{
		array.push_back((rand()%(max-min+1))+min);
	}
	return;
}

double playGetMeanIntArray(std::vector<int>& array)
{
	int sum = 0;

	for (int i = 0; i < array.size(); i ++)
	{
		sum += array[i];
	}

	return (double)sum/array.size();
}

void playEigen()
{
	
}
