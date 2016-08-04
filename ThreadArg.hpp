/*************************************************************************
    > File Name: ThreadArgs.hpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月04日 星期四 19时55分53秒
 ************************************************************************/
#ifndef _THREADARG_HPP_
#define _THREADARG_HPP_
#include "Include.hpp"
#include "Master.hpp"
#include "State.hpp"

class ThreadArg
{
public:
	ThreadArg(Master& master_):master(master_)
	{

	};

	Master& master;
	std::vector<State*> states;
};

class ThreadArg1
{
public:
	ThreadArg1(Master& master_):master(master_)
	{

	};

	Master& master;
	std::vector<State*> encStates, decStates;
};

#endif
