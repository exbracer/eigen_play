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
#include "Grad.hpp"
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

class ThreadArg2
{
public:
	ThreadArg2(Master& master_):master(master_)
	{

	};

	Master& master;
	std::vector<State*> encStates, decStates;
	Grad src_grad, tgt_grad;
};

class ThreadArg3
{
public:
	ThreadArg3(Master& master_1, Master& master_2):master_enc(master_1), master_dec(master_2)
	{

	};

	Master& master_enc;
	Master& master_dec;
	std::vector<State*> encStates, decStates;
	Grad src_grad, tgt_grad;
	Softmax* softmax;
};

class ThreadArg4
{
public:
	ThreadArg4(Master& master_1, Master& master_2):master_enc(master_1), master_dec(master_2)
	{

	};

	Master& master_enc;
	Master& master_dec;
	std::vector<State*> encStates, decStates;
	Grad src_grad, tgt_grad;
	Softmax* softmax;
	VecD target_dist;
};

#endif
