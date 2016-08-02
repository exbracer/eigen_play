/*************************************************************************
    > File Name: main.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 11时34分18秒
 ************************************************************************/

#include<iostream>
#include "Include.hpp"
#include "Func.hpp"

double curTime()
{
	struct timespec tp[1];
	int r = clock_gettime(CLOCK_REALTIME, tp);
	assert(r == 0);

	return tp->tv_sec + tp->tv_nsec * 1.0e-9;
}

void testFunc1(int argc, char* argv[])
{
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int num_batches = NUM_BATCHES;
	int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;

	// 
	std::vector<State*> states;
	Master * master = new Master(hidden_dim, input_dim);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);

	// variables for time recording
	

	// initialization
	for (int i = 0; i < max_num_states; i ++)
	{
		states.push_back(new State);	
	}

	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
	for (int i = 0; i < num_batches; i ++)
	{
		states[0]->h = zeros;
		states[0]->c = zeros;
		for (int j = 0; j < num_src_terms; j ++)
		{
			forwardPart1(xt, states[i], states[i+1], master);
		}
	}
	
	double t1 = curTime();


	return;
}

// the main function
int main(int argc, char* argv[])
{
	
	testFunc1(argc, argv);

	return 0;
}
