/*************************************************************************
    > File Name: main.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月02日 星期二 11时34分18秒
 ************************************************************************/

#include "Include.hpp"
#include "Func.hpp"
#include "ThreadArg.hpp"
// tool function
double curTime()
{
	struct timespec tp[1];
	int r = clock_gettime(CLOCK_REALTIME, tp);
	assert(r == 0);

	return tp->tv_sec + tp->tv_nsec * 1.0e-9;
}

void genRandInt(std::vector<int>& array, int len, int min, int max)
{
	srand(19920403);
	for (int i = 0; i < len; i ++)
	{
		array.push_back((rand()%(max-min+1))+min);
	}

	return;
}

void genRandIntDynamic(std::vector<int>& array, int len, int min, int max)
{
	//srand(time(NULL));
	for (int i = 0; i < len; i ++)
	{
		array.push_back((rand()%(max-min+1))+min);
	}

	return;
}

double getMeanIntArray(std::vector<int>& array)
{
	int sum = 0; 
	for (int i = 0; i < array.size(); i++)
	{
		sum += array[i];
	}

	return (double)sum/array.size();
}

// test function
void testFunc1(int argc, char* argv[])
{
	std::cout << "test func 1" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;

	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;

	// 
	// std::vector<State*> states;
	std::vector<State*> states[num_threads];
	Master * master = new Master(hidden_dim, input_dim);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);

	// variables for time recording
	
	std::cout << "xt size = " << xt.size() << std::endl;
	// initialization
	for (int i = 0; i < num_threads; i ++)
	{
		for (int j = 0; j < max_num_states; j ++)
		{
			states[i].push_back(new State);
		}
	}
	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();
		states[id][0]->h = zeros;
		states[id][0]->c = zeros;
		for (int j = 0; j < num_src_terms; j ++)
		{
			forwardPart1(xt, states[id][j], states[id][j+1], master);
			forwardPart2(xt, states[id][j], states[id][j+1], master);
		}
		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
}

// test function 2
void testFunc2(int argc, char* argv[])
{
	std::cout << "test func 2" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;

	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;

	// 
	// std::vector<State*> states;
	std::vector<ThreadArg*> args;
	Master * master = new Master(hidden_dim, input_dim);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);

	// variables for time recording
	
	
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->states.push_back(new State);
			args[i]->states[j]->h = zeros;
			args[i]->states[j]->c = zeros;
		}
	}
	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();

		for (int j = 0; j < num_src_terms; j ++)
		{
			forwardPart1(xt, args[id]->states[j], args[id]->states[j+1], master);
			forwardPart2(xt, args[id]->states[j], args[id]->states[j+1], master);
		}
		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 2

// test function 3
void testFunc3(int argc, char* argv[])
{
	std::cout << "test func 3" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	// int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;
	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;

	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	// std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;

	// 
	// std::vector<State*> states;
	std::vector<ThreadArg*> args;
	Master * master = new Master(hidden_dim, input_dim);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);

	// variables for time recording
	

	// variables 
	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->states.push_back(new State);
			args[i]->states[j]->h = zeros;
			args[i]->states[j]->c = zeros;
		}
	}
	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();

		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forwardPart1(xt, args[id]->states[j], args[id]->states[j+1], master);
			forwardPart2(xt, args[id]->states[j], args[id]->states[j+1], master);
		}
		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 3

void testFunc4(int argc, char* argv[])
{
	std::cout << "test func 4" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	// int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;
	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;
	int src_voc_size = SRC_VOC_SIZE;
	int tgt_voc_size = TGT_VOC_SIZE;


	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	// std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;

	// 
	// std::vector<State*> states;
	const Real scale = 0.1;
	std::vector<ThreadArg*> args;
	Rand rnd;
	Master * master = new Master(hidden_dim, input_dim);
	master->init(rnd, scale);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);
	//MatD src_embd = MatD::Random(input_dim, src_voc_size);
	MatD src_embd = MatD(input_dim, src_voc_size);
	rnd.uniform(src_embd, scale);
	// variables for time recording
	

	// variables 
	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	srand(20101024);
	for (int i = 0; i < num_threads; i++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	
	// for temp test
	/*
	for (int i = 0; i < num_threads; i ++)
	{
		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			std::cout << src_data[i][j] << " ";
		}
		std::cout << std::endl;
	}
	*/
	// for temp test
	
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->states.push_back(new State);
			args[i]->states[j]->h = zeros;
			args[i]->states[j]->c = zeros;
		}
	}
	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();

		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forwardPart1(src_embd.col(src_data[id][j]), args[id]->states[j], args[id]->states[j+1], master);
			forwardPart2(src_embd.col(src_data[id][j]), args[id]->states[j], args[id]->states[j+1], master);
		}
		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 4

void testFunc5(int argc, char* argv[])
{
	std::cout << "test func 5" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	// int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;
	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;
	int src_voc_size = SRC_VOC_SIZE;
	int tgt_voc_size = TGT_VOC_SIZE;


	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	// std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;

	// 
	// std::vector<State*> states;
	const Real scale = 0.1;
	std::vector<ThreadArg*> args;
	Rand rnd;
	Master * master = new Master(hidden_dim, input_dim);
	master->init(rnd, scale);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);
	//MatD src_embd = MatD::Random(input_dim, src_voc_size);
	MatD src_embd = MatD(input_dim, src_voc_size);
	rnd.uniform(src_embd, scale);
	// variables for time recording
	
	// variables 
	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	srand(20101024);
	for (int i = 0; i < num_threads; i++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->states.push_back(new State);
			args[i]->states[j]->h = zeros;
			args[i]->states[j]->c = zeros;
		}
	}
	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();

		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forward(src_embd.col(src_data[id][j]), args[id]->states[j], args[id]->states[j+1], master);
		}
		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 5

void testFunc6(int argc, char* argv[])
{
	std::cout << "test func 6" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	// int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;
	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;
	int max_num_tgt_terms = MAX_NUM_TGT_TERMS;
	int min_num_tgt_terms = MIN_NUM_TGT_TERMS;
	int src_voc_size = SRC_VOC_SIZE;
	int tgt_voc_size = TGT_VOC_SIZE;


	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	// std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;
	std::cout << "max_num_src_terms = " << max_num_src_terms << std::endl;
	std::cout << "min_num_src_terms = " << min_num_src_terms << std::endl;
	std::cout << "max_num_tgt_terms = " << max_num_tgt_terms << std::endl;
	std::cout << "min_num_tgt_terms = " << min_num_tgt_terms << std::endl;
	std::cout << "src_voc_size = " << src_voc_size << std::endl;
	std::cout << "tgt_voc_size = " << tgt_voc_size << std::endl;
	// 
	// std::vector<State*> states;
	const Real scale = 0.1;
	std::vector<ThreadArg1*> args;
	Rand rnd;
	Master * master = new Master(hidden_dim, input_dim);
	master->init(rnd, scale);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);
	//MatD src_embd = MatD::Random(input_dim, src_voc_size);
	MatD src_embd = MatD(input_dim, src_voc_size);
	MatD tgt_embd = MatD(input_dim, tgt_voc_size);
	rnd.uniform(src_embd, scale);
	rnd.uniform(tgt_embd, scale);
	// variables for time recording
	
	// variables 
	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> num_tgt_terms;
	genRandInt(num_tgt_terms, batch_size, min_num_tgt_terms, max_num_tgt_terms);
	std::cout << "mean of num_tgt_terms = " << getMeanIntArray(num_tgt_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	std::vector<int> tgt_data[num_threads];
	srand(20101024);
	for (int i = 0; i < num_threads; i++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(tgt_data[i], max_num_tgt_terms, 1, tgt_voc_size);
	}
	
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg1(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->encStates.push_back(new State);
			args[i]->encStates[j]->h = zeros;
			args[i]->encStates[j]->c = zeros;
			args[i]->decStates.push_back(new State);
			
		}
	}

	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();

		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forward(src_embd.col(src_data[id][j]), args[id]->encStates[j], args[id]->encStates[j+1], master);
		}

		args[id]->decStates[0]->h = args[id]->encStates[num_src_terms[i]]->h;
		args[id]->decStates[0]->c = args[id]->encStates[num_src_terms[i]]->c;
		
		for (int j = 1; j < num_tgt_terms[i]; j ++)
		{
			forward(tgt_embd.col(tgt_data[id][j]), args[id]->decStates[j-1], args[id]->decStates[j], master);
		}


		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 6

void testFunc7(int argc, char* argv[])
{
	std::cout << "test func 7" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	// int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;
	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;
	int max_num_tgt_terms = MAX_NUM_TGT_TERMS;
	int min_num_tgt_terms = MIN_NUM_TGT_TERMS;
	int src_voc_size = SRC_VOC_SIZE;
	int tgt_voc_size = TGT_VOC_SIZE;


	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	// std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;
	std::cout << "max_num_src_terms = " << max_num_src_terms << std::endl;
	std::cout << "min_num_src_terms = " << min_num_src_terms << std::endl;
	std::cout << "max_num_tgt_terms = " << max_num_tgt_terms << std::endl;
	std::cout << "min_num_tgt_terms = " << min_num_tgt_terms << std::endl;
	std::cout << "src_voc_size = " << src_voc_size << std::endl;
	std::cout << "tgt_voc_size = " << tgt_voc_size << std::endl;
	// 
	// std::vector<State*> states;
	const Real scale = 0.1;
	std::vector<ThreadArg1*> args;
	Rand rnd;
	Master * master = new Master(hidden_dim, input_dim);
	master->init(rnd, scale);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);
	//MatD src_embd = MatD::Random(input_dim, src_voc_size);
	MatD src_embd = MatD(input_dim, src_voc_size);
	MatD tgt_embd = MatD(input_dim, tgt_voc_size);
	rnd.uniform(src_embd, scale);
	rnd.uniform(tgt_embd, scale);

	Softmax * softmax = new Softmax(hidden_dim, tgt_voc_size);

	// variables for time recording
	
	// variables 
	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> num_tgt_terms;
	genRandInt(num_tgt_terms, batch_size, min_num_tgt_terms, max_num_tgt_terms);
	std::cout << "mean of num_tgt_terms = " << getMeanIntArray(num_tgt_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	std::vector<int> tgt_data[num_threads];
	srand(20101024);
	for (int i = 0; i < num_threads; i++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(tgt_data[i], max_num_tgt_terms, 1, tgt_voc_size);
	}
	
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg1(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->encStates.push_back(new State);
			args[i]->encStates[j]->h = zeros;
			args[i]->encStates[j]->c = zeros;
			args[i]->decStates.push_back(new State);
			
		}
	}

	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();
		VecD target_dist;

		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forward(src_embd.col(src_data[id][j]), args[id]->encStates[j], args[id]->encStates[j+1], master);
		}

		args[id]->decStates[0]->h = args[id]->encStates[num_src_terms[i]]->h;
		args[id]->decStates[0]->c = args[id]->encStates[num_src_terms[i]]->c;
		softmaxCalcDist(args[id]->decStates[0]->h, target_dist, softmax);

		for (int j = 1; j < num_tgt_terms[i]; j ++)
		{
			forward(tgt_embd.col(tgt_data[id][j]), args[id]->decStates[j-1], args[id]->decStates[j], master);
			
			softmaxCalcDist(args[id]->decStates[j]->h, target_dist, softmax);

		}


		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 7

void testFunc8(int argc, char* argv[])
{
	std::cout << "test func 8" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	// int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;
	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;
	int max_num_tgt_terms = MAX_NUM_TGT_TERMS;
	int min_num_tgt_terms = MIN_NUM_TGT_TERMS;
	int src_voc_size = SRC_VOC_SIZE;
	int tgt_voc_size = TGT_VOC_SIZE;


	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	// std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;
	std::cout << "max_num_src_terms = " << max_num_src_terms << std::endl;
	std::cout << "min_num_src_terms = " << min_num_src_terms << std::endl;
	std::cout << "max_num_tgt_terms = " << max_num_tgt_terms << std::endl;
	std::cout << "min_num_tgt_terms = " << min_num_tgt_terms << std::endl;
	std::cout << "src_voc_size = " << src_voc_size << std::endl;
	std::cout << "tgt_voc_size = " << tgt_voc_size << std::endl;
	// 
	// std::vector<State*> states;
	const Real scale = 0.1;
	std::vector<ThreadArg1*> args;
	Rand rnd;
	Master * master = new Master(hidden_dim, input_dim);
	master->init(rnd, scale);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);
	//MatD src_embd = MatD::Random(input_dim, src_voc_size);
	MatD src_embd = MatD(input_dim, src_voc_size);
	MatD tgt_embd = MatD(input_dim, tgt_voc_size);
	rnd.uniform(src_embd, scale);
	rnd.uniform(tgt_embd, scale);

	Softmax * softmax = new Softmax(hidden_dim, tgt_voc_size);

	// variables for time recording
	
	// variables 
	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> num_tgt_terms;
	genRandInt(num_tgt_terms, batch_size, min_num_tgt_terms, max_num_tgt_terms);
	std::cout << "mean of num_tgt_terms = " << getMeanIntArray(num_tgt_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	std::vector<int> tgt_data[num_threads];
	srand(20101024);
	for (int i = 0; i < num_threads; i++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(tgt_data[i], max_num_tgt_terms, 1, tgt_voc_size);
	}
	
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg1(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->encStates.push_back(new State);
			args[i]->encStates[j]->h = zeros;
			args[i]->encStates[j]->c = zeros;
			args[i]->decStates.push_back(new State);
			
		}
	}

	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();
		VecD target_dist;

		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forward(src_embd.col(src_data[id][j]), args[id]->encStates[j], args[id]->encStates[j+1], master);
		}

		args[id]->decStates[0]->h = args[id]->encStates[num_src_terms[i]]->h;
		args[id]->decStates[0]->c = args[id]->encStates[num_src_terms[i]]->c;
		softmaxCalcDist(args[id]->decStates[0]->h, target_dist, softmax);
		softmaxBackward(args[id]->decStates[0]->h, target_dist, tgt_data[id][0], args[id]->decStates[0]->delh, softmax);

		for (int j = 1; j < num_tgt_terms[i]; j ++)
		{
			forward(tgt_embd.col(tgt_data[id][j]), args[id]->decStates[j-1], args[id]->decStates[j], master);
			
			softmaxCalcDist(args[id]->decStates[j]->h, target_dist, softmax);

			softmaxBackward(args[id]->decStates[j]->h, target_dist, tgt_data[id][j], args[id]->decStates[j]->delh, softmax);
		}


		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 8

void testFunc9(int argc, char* argv[])
{
	std::cout << "test func 9" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	// int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;
	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;
	int max_num_tgt_terms = MAX_NUM_TGT_TERMS;
	int min_num_tgt_terms = MIN_NUM_TGT_TERMS;
	int src_voc_size = SRC_VOC_SIZE;
	int tgt_voc_size = TGT_VOC_SIZE;


	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	// std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;
	std::cout << "max_num_src_terms = " << max_num_src_terms << std::endl;
	std::cout << "min_num_src_terms = " << min_num_src_terms << std::endl;
	std::cout << "max_num_tgt_terms = " << max_num_tgt_terms << std::endl;
	std::cout << "min_num_tgt_terms = " << min_num_tgt_terms << std::endl;
	std::cout << "src_voc_size = " << src_voc_size << std::endl;
	std::cout << "tgt_voc_size = " << tgt_voc_size << std::endl;
	// 
	// std::vector<State*> states;
	const Real scale = 0.1;
	std::vector<ThreadArg1*> args;
	Rand rnd;
	Master * master = new Master(hidden_dim, input_dim);
	master->init(rnd, scale);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);
	//MatD src_embd = MatD::Random(input_dim, src_voc_size);
	MatD src_embd = MatD(input_dim, src_voc_size);
	MatD tgt_embd = MatD(input_dim, tgt_voc_size);
	rnd.uniform(src_embd, scale);
	rnd.uniform(tgt_embd, scale);

	Softmax * softmax = new Softmax(hidden_dim, tgt_voc_size);

	// variables for time recording
	
	// variables 
	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> num_tgt_terms;
	genRandInt(num_tgt_terms, batch_size, min_num_tgt_terms, max_num_tgt_terms);
	std::cout << "mean of num_tgt_terms = " << getMeanIntArray(num_tgt_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	std::vector<int> tgt_data[num_threads];
	srand(20101024);
	for (int i = 0; i < num_threads; i++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(tgt_data[i], max_num_tgt_terms, 1, tgt_voc_size);
	}
	
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg1(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->encStates.push_back(new State);
			args[i]->encStates[j]->h = zeros;
			args[i]->encStates[j]->c = zeros;
			args[i]->decStates.push_back(new State);
			
		}
	}

	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();
		VecD target_dist;

		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forward(src_embd.col(src_data[id][j]), args[id]->encStates[j], args[id]->encStates[j+1], master);
		}

		args[id]->decStates[0]->h = args[id]->encStates[num_src_terms[i]]->h;
		args[id]->decStates[0]->c = args[id]->encStates[num_src_terms[i]]->c;

		for (int j = 1; j < num_tgt_terms[i]; j ++)
		{
			forward(tgt_embd.col(tgt_data[id][j]), args[id]->decStates[j-1], args[id]->decStates[j], master);
		}

		for (int j = 0; j < num_tgt_terms[i]; j ++)
		{
			softmaxCalcDist(args[id]->decStates[j]->h, target_dist, softmax);

			softmaxBackward(args[id]->decStates[j]->h, target_dist, tgt_data[id][j], args[id]->decStates[j]->delh, softmax);
		}


		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 9

void testFunc10(int argc, char* argv[])
{
	std::cout << "test func 10" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	// int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;
	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;
	int max_num_tgt_terms = MAX_NUM_TGT_TERMS;
	int min_num_tgt_terms = MIN_NUM_TGT_TERMS;
	int src_voc_size = SRC_VOC_SIZE;
	int tgt_voc_size = TGT_VOC_SIZE;


	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	// std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;
	std::cout << "max_num_src_terms = " << max_num_src_terms << std::endl;
	std::cout << "min_num_src_terms = " << min_num_src_terms << std::endl;
	std::cout << "max_num_tgt_terms = " << max_num_tgt_terms << std::endl;
	std::cout << "min_num_tgt_terms = " << min_num_tgt_terms << std::endl;
	std::cout << "src_voc_size = " << src_voc_size << std::endl;
	std::cout << "tgt_voc_size = " << tgt_voc_size << std::endl;
	// 
	// std::vector<State*> states;
	const Real scale = 0.1;
	std::vector<ThreadArg1*> args;
	Rand rnd;
	Master * master = new Master(hidden_dim, input_dim);
	master->init(rnd, scale);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);
	//MatD src_embd = MatD::Random(input_dim, src_voc_size);
	MatD src_embd = MatD(input_dim, src_voc_size);
	MatD tgt_embd = MatD(input_dim, tgt_voc_size);
	rnd.uniform(src_embd, scale);
	rnd.uniform(tgt_embd, scale);

	Softmax * softmax = new Softmax(hidden_dim, tgt_voc_size);

	// variables for time recording
	
	// variables 
	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> num_tgt_terms;
	genRandInt(num_tgt_terms, batch_size, min_num_tgt_terms, max_num_tgt_terms);
	std::cout << "mean of num_tgt_terms = " << getMeanIntArray(num_tgt_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	std::vector<int> tgt_data[num_threads];
	srand(20101024);
	for (int i = 0; i < num_threads; i++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(tgt_data[i], max_num_tgt_terms, 1, tgt_voc_size);
	}
	
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg1(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->encStates.push_back(new State);
			args[i]->encStates[j]->h = zeros;
			args[i]->encStates[j]->c = zeros;
			args[i]->decStates.push_back(new State);
			
		}
	}

	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();
		VecD target_dist;

		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forward(src_embd.col(src_data[id][j]), args[id]->encStates[j], args[id]->encStates[j+1], master);
		}

		args[id]->decStates[0]->h = args[id]->encStates[num_src_terms[i]]->h;
		args[id]->decStates[0]->c = args[id]->encStates[num_src_terms[i]]->c;

		for (int j = 1; j < num_tgt_terms[i]; j ++)
		{
			forward(tgt_embd.col(tgt_data[id][j]), args[id]->decStates[j-1], args[id]->decStates[j], master);
		}

		for (int j = 0; j < num_tgt_terms[i]; j ++)
		{
			softmaxCalcDist(args[id]->decStates[j]->h, target_dist, softmax);

			softmaxBackward1(args[id]->decStates[j]->h, target_dist, tgt_data[id][j], args[id]->decStates[j]->delh, softmax);
		}


		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 10

void testFunc11(int argc, char* argv[])
{
	std::cout << "test func 11" << std::endl;
	// parameters of NN
	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	// int num_src_terms = NUM_SRC_TERMS;
	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;
	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;
	int max_num_tgt_terms = MAX_NUM_TGT_TERMS;
	int min_num_tgt_terms = MIN_NUM_TGT_TERMS;
	int src_voc_size = SRC_VOC_SIZE;
	int tgt_voc_size = TGT_VOC_SIZE;

	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	// std::cout << "num_src_terms = " << num_src_terms << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;
	std::cout << "max_num_src_terms = " << max_num_src_terms << std::endl;
	std::cout << "min_num_src_terms = " << min_num_src_terms << std::endl;
	std::cout << "max_num_tgt_terms = " << max_num_tgt_terms << std::endl;
	std::cout << "min_num_tgt_terms = " << min_num_tgt_terms << std::endl;
	std::cout << "src_voc_size = " << src_voc_size << std::endl;
	std::cout << "tgt_voc_size = " << tgt_voc_size << std::endl;
	// 
	// std::vector<State*> states;
	const Real scale = 0.1;
	std::vector<ThreadArg1*> args;
	Rand rnd;
	Master * master = new Master(hidden_dim, input_dim);
	master->init(rnd, scale);
	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);
	//MatD src_embd = MatD::Random(input_dim, src_voc_size);
	MatD src_embd = MatD(input_dim, src_voc_size);
	MatD tgt_embd = MatD(input_dim, tgt_voc_size);
	rnd.uniform(src_embd, scale);
	rnd.uniform(tgt_embd, scale);

	Softmax * softmax = new Softmax(hidden_dim, tgt_voc_size);

	// variables for time recording
	
	// variables 
	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> num_tgt_terms;
	genRandInt(num_tgt_terms, batch_size, min_num_tgt_terms, max_num_tgt_terms);
	std::cout << "mean of num_tgt_terms = " << getMeanIntArray(num_tgt_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	std::vector<int> tgt_data[num_threads];
	srand(20101024);
	for (int i = 0; i < num_threads; i++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(tgt_data[i], max_num_tgt_terms, 1, tgt_voc_size);
	}
	
	// initialization
	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg1(*master));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->encStates.push_back(new State);
			args[i]->encStates[j]->h = zeros;
			args[i]->encStates[j]->c = zeros;
			args[i]->decStates.push_back(new State);
			
		}
	}

	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();
		VecD target_dist;

		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forward(src_embd.col(src_data[id][j]), args[id]->encStates[j], args[id]->encStates[j+1], master);
		}

		args[id]->decStates[0]->h = args[id]->encStates[num_src_terms[i]]->h;
		args[id]->decStates[0]->c = args[id]->encStates[num_src_terms[i]]->c;

		for (int j = 1; j < num_tgt_terms[i]; j ++)
		{
			forward(tgt_embd.col(tgt_data[id][j]), args[id]->decStates[j-1], args[id]->decStates[j], master);
		}

		for (int j = 0; j < num_tgt_terms[i]; j ++)
		{
			softmaxCalcDist(args[id]->decStates[j]->h, target_dist, softmax);

			softmaxBackward(args[id]->decStates[j]->h, target_dist, tgt_data[id][j], args[id]->decStates[j]->delh, softmax);
		}
		
		args[id]->decStates[num_tgt_terms[i]]->delc = zeros;

		for (int j = num_tgt_terms[i]; j >= 1; j --)
		{
			args[id]->decStates[i-1]->delc = zeros;
			
		}

		asm volatile("# end");
	}
	
	double t1 = curTime();
	
	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
} // end of test function 11

// the main function
int main(int argc, char* argv[])
{
	// srand(19920403);
	std::cout << "start" << std::endl;	
	std::cout << std::endl;
	// for test function 1
	/*
	std::cout << "test function 1" << std::endl;
	testFunc1(argc, argv);
	std::cout << std::endl;
	// for test function 2
	std::cout << "test function 2" << std::endl;
	testFunc2(argc, argv);
	std::cout << std::endl;

	// for test function 3
	std::cout << "test function 3" << std::endl;
	testFunc3(argc, argv);
	std::cout << std::endl;
	// for test function 4
	std::cout << "test function 4" << std::endl;
	testFunc4(argc, argv);
	std::cout << std::endl;
	// for test function 5
	std::cout << "test function 5" << std::endl;
	testFunc5(argc, argv);
	std::cout << std::endl;
	// for test function 6
	std::cout << "test function 6" << std::endl;
	testFunc6(argc, argv);
	std::cout << std::endl;
	// for test function 7
	std::cout << "test function 7" << std::endl;
	testFunc7(argc, argv);
	std::cout << std::endl;
	
	// for test function 8
	std::cout << "test function 8" << std::endl;
	testFunc8(argc, argv);
	std::cout << std::endl;
	*/
	// for test function 9
	std::cout << "test function 9" << std::endl;
	testFunc9(argc, argv);
	std::cout << std::endl;
	// for test function 10
	std::cout << "test function 10" << std::endl;
	testFunc10(argc, argv);
	std::cout << std::endl;
	// for test function 11
	std::cout << "test function 11" << std::endl;
	testFunc11(argc, argv);
	std::cout << std::endl;
	// for test function 12
	std::cout << "test function 12" << std::endl;
	testFunc12(argc, argv);
	std::cout << std::endl;

	std::cout << "end" << std::endl;

	return 0;
}
