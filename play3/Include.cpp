/*************************************************************************
	> File Name: Include.cpp
	> Author: Yuchen Qiao
	> Mail: qiaoyc14@mails.tsinghua.edu.cn
	> Created Time: Wed 05 Oct 2016 08:29:52 PM CST
 ************************************************************************/

#include "Include.hpp"

double playCurrTime()
{
    struct timespec tp[1];
    int r = clock_gettime(CLOCK_REALTIME, tp);
    assert(r == 0);

    return tp->tv_sec + tp->tv_nsec * 1.0e-9;

}

void playGenRandInt(std::vector<int>& array, int len, int min, int max)
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
