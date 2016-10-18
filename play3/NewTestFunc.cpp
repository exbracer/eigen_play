/*************************************************************************
    > File Name: NewTestFunc.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月26日 星期五 15时10分56秒
 ************************************************************************/

#include "NewTestFunc.hpp"

void newTestFunc1(int argc, char** argv)
{
	std::cout << "new test func 1" << std::endl;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;

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
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;
	std::cout << "max_num_src_terms = " << max_num_src_terms << std::endl;
	std::cout << "min_num_src_terms = " << min_num_src_terms << std::endl;
	std::cout << "max_num_tgt_terms = " << max_num_tgt_terms << std::endl;
	std::cout << "min_num_tgt_terms = " << min_num_tgt_terms << std::endl;
	std::cout << "src_voc_size = " << src_voc_size << std::endl;
	std::cout << "tgt_voc_size = " << tgt_voc_size << std::endl;

	const Real scale = 0.1;
	std::vector<ThreadArg4*> args;
	Rand rnd;
	Master* master_enc = new Master(hidden_dim, input_dim);
	master_enc->init(rnd, scale);
	Master* master_dec = new Master(hidden_dim, input_dim);
	master_dec->init(rnd, scale);

	VecD xt = VecD(input_dim);
	VecD zeros = VecD::Zero(hidden_dim);
	VecD random_vec = VecD::Random(hidden_dim);

	MatD src_embd = MatD(input_dim, src_voc_size);
	MatD tgt_embd = MatD(input_dim, tgt_voc_size);
	rnd.uniform(src_embd, scale);
	rnd.uniform(tgt_embd, scale);

	Softmax * softmax = new Softmax(hidden_dim, tgt_voc_size);
	Softmax * softmax_2 = new Softmax(hidden_dim, tgt_voc_size);

	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> num_tgt_terms;
	genRandInt(num_tgt_terms, batch_size, min_num_tgt_terms, max_num_tgt_terms);
	std::cout << "mean of num_tgt_terms = " << getMeanIntArray(num_tgt_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	std::vector<int> tgt_data[num_threads];
	srand(20101024);

	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(tgt_data[i], max_num_tgt_terms, 1, tgt_voc_size);
	}

	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg4(*master_enc, *master_dec));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->encStates.push_back(new State);
			args[i]->encStates[j]->h = zeros;
			args[i]->encStates[j]->c = zeros;
			args[i]->decStates.push_back(new State);
		}
		args[i]->softmax = new Softmax(hidden_dim, tgt_voc_size);
		args[i]->target_dist = VecD::Zero(tgt_voc_size);
	}

	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# start");
		int id = omp_get_thread_num();
		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forward(src_embd.col(src_data[id][j]), args[id]->encStates[j], args[id]->encStates[j+1], master_enc);
		}

		args[id]->decStates[0]->h = args[id]->encStates[num_src_terms[i]]->h;
		args[id]->decStates[0]->c = args[id]->encStates[num_src_terms[i]]->c;

		for (int j = 1; j < num_tgt_terms[i]; j ++)
		{
			forward(tgt_embd.col(tgt_data[id][j]), args[id]->decStates[j-1], args[id]->decStates[j], master_dec);
		}

		for (int j = 0; j < num_tgt_terms[i]; j ++)
		{
            // do softmax operation
		}
		args[id]->decStates[num_tgt_terms[i]]->delc = zeros;
		args[id]->decStates[num_tgt_terms[i]]->delh = random_vec;
		args[id]->decStates[num_tgt_terms[i]-1]->delh = random_vec;
		
		std::vector<VecD> tgt_delos;
		std::vector<VecD> tgt_delis;
		std::vector<VecD> tgt_delus;
		std::vector<VecD> tgt_delfs;
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			tgt_delos.push_back(zeros);
			tgt_delis.push_back(zeros);
			tgt_delus.push_back(zeros);
			tgt_delfs.push_back(zeros);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->decStates[j-1]->delc = zeros;
			args[id]->decStates[j-1]->delh = random_vec;

			// backward(args[id]->decStates[j-1], args[id]->decStates[j], args[id]->tgt_grad, tgt_embd.col(tgt_data[id][j-1]), master_dec);
			newBackward(args[id]->decStates[j-1], args[id]->decStates[j], master_dec, tgt_delos[j-1], tgt_delis[j-1], tgt_delus[j-1], tgt_delfs[j-1]);
			
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Wxi.noalias() += tgt_delis[j-1] * tgt_embd.col(tgt_data[id][j-1]);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Whi.noalias() += tgt_delis[j-1] * args[id]->decStates[j-1]->h.transpose();
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Wxf.noalias() += tgt_delfs[j-1] * tgt_embd.col(tgt_data[id][j-1]);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Whf.noalias() += tgt_delfs[j-1] * args[id]->decStates[j-1]->h.transpose();
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Wxo.noalias() += tgt_delos[j-1] * tgt_embd.col(tgt_data[id][j-1]);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Who.noalias() += tgt_delos[j-1] * args[id]->decStates[j-1]->h.transpose();
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Wxu.noalias() += tgt_delus[j-1] * tgt_embd.col(tgt_data[id][j-1]);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Whu.noalias() += tgt_delus[j-1] * args[id]->decStates[j-1]->h.transpose();
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.bi += tgt_delis[j-1];
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.bf += tgt_delfs[j-1];
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.bo += tgt_delos[j-1];
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.bu += tgt_delus[j-1];
		}
		std::vector<VecD> src_delos;
		std::vector<VecD> src_delis;
		std::vector<VecD> src_delfs;
		std::vector<VecD> src_delus;

		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			src_delos.push_back(zeros);
			src_delis.push_back(zeros);
			src_delus.push_back(zeros);
			src_delfs.push_back(zeros);
		}

		args[id]->encStates[num_src_terms[i]]->delc = args[id]->decStates[0]->delc;
		args[id]->encStates[num_src_terms[i]]->delh = args[id]->decStates[0]->delh;

		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->encStates[j-1]->delh = zeros;
			args[id]->encStates[j-1]->delc = zeros;
			//backward(args[id]->encStates[j-1], args[id]->encStates[j], args[id]->src_grad, src_embd.col(src_data[id][j-1]), master_enc);
			newBackward(args[id]->encStates[j-1], args[id]->encStates[j], master_enc, src_delos[j-1], src_delis[j-1], src_delus[j-1], src_delfs[j-1]);
		}

		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Wxi.noalias() += src_delis[j-1] * src_embd.col(src_data[id][j-1]);
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Whi.noalias() += src_delis[j-1] * args[id]->encStates[j-1]->h.transpose();
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Wxf.noalias() += src_delfs[j-1] * src_embd.col(src_data[id][j-1]);
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Whf.noalias() += src_delfs[j-1] * args[id]->encStates[j-1]->h.transpose();
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Wxo.noalias() += src_delos[j-1] * src_embd.col(src_data[id][j-1]);
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Who.noalias() += src_delos[j-1] * args[id]->encStates[j-1]->h.transpose();
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Wxu.noalias() += src_delus[j-1] * src_embd.col(src_data[id][j-1]);
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Whu.noalias() += src_delus[j-1] * args[id]->encStates[j-1]->h.transpose();
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.bi += src_delis[j-1];
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.bf += src_delfs[j-1];
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.bo += src_delos[j-1];
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.bu += src_delus[j-1];
		}
		asm volatile("# end");
	}

	double t1 = curTime();

	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
}


void newTestFunc2(int argc, char** argv)
{
    std::cout << "play 3: new test func 2" << std::endl;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;

	int max_num_states = MAX_NUM_STATES;
	int input_dim = INPUT_DIM;
	int hidden_dim = HIDDEN_DIM;

	int max_num_src_terms = MAX_NUM_SRC_TERMS;
	int min_num_src_terms = MIN_NUM_SRC_TERMS;
	int max_num_tgt_terms = MAX_NUM_TGT_TERMS;
	int min_num_tgt_terms = MIN_NUM_TGT_TERMS;
    int max_num_terms = playIntMax(max_num_src_terms, max_num_tgt_terms);

	int src_voc_size = SRC_VOC_SIZE;
	int tgt_voc_size = TGT_VOC_SIZE;
    
    int set_size = SET_SIZE;

	std::cout << "num_threads = " << num_threads << std::endl;
	std::cout << "batch_size = " << batch_size << std::endl;
	std::cout << "max_num_states = " << max_num_states << std::endl;
	std::cout << "input_dim = " << input_dim << std::endl;
	std::cout << "hidden_dim = " << hidden_dim << std::endl;
	std::cout << "max_num_src_terms = " << max_num_src_terms << std::endl;
	std::cout << "min_num_src_terms = " << min_num_src_terms << std::endl;
	std::cout << "max_num_tgt_terms = " << max_num_tgt_terms << std::endl;
	std::cout << "min_num_tgt_terms = " << min_num_tgt_terms << std::endl;
	std::cout << "src_voc_size = " << src_voc_size << std::endl;
	std::cout << "tgt_voc_size = " << tgt_voc_size << std::endl;

	const Real scale = 0.1;
	std::vector<ThreadArg5*> args;
	Rand rnd;
	Master* master_enc = new Master(hidden_dim, input_dim);
	master_enc->init(rnd, scale);
	Master* master_dec = new Master(hidden_dim, input_dim);
	master_dec->init(rnd, scale);

	VecD xt = VecD(input_dim);
	VecD zeros_vec = VecD::Zero(hidden_dim);
	VecD random_vec = VecD::Random(hidden_dim);

	MatD src_embd = MatD(input_dim, src_voc_size);
	MatD tgt_embd = MatD(input_dim, tgt_voc_size);
	rnd.uniform(src_embd, scale);
	rnd.uniform(tgt_embd, scale);

	Softmax * softmax = new Softmax(hidden_dim, tgt_voc_size);
	Softmax * softmax_2 = new Softmax(hidden_dim, tgt_voc_size);

	std::vector<int> num_src_terms;
	genRandInt(num_src_terms, batch_size, min_num_src_terms, max_num_src_terms);
	std::cout << "mean of num_src_terms = " << getMeanIntArray(num_src_terms) << std::endl;

	std::vector<int> num_tgt_terms;
	genRandInt(num_tgt_terms, batch_size, min_num_tgt_terms, max_num_tgt_terms);
	std::cout << "mean of num_tgt_terms = " << getMeanIntArray(num_tgt_terms) << std::endl;

	std::vector<int> src_data[num_threads];
	std::vector<int> tgt_data[num_threads];
	srand(20101024);

	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(src_data[i], max_num_src_terms, 1, src_voc_size);
	}
	for (int i = 0; i < num_threads; i ++)
	{
		genRandIntDynamic(tgt_data[i], max_num_tgt_terms, 1, tgt_voc_size);
	}

	args.clear();
	for (int i = 0; i < num_threads; i ++)
	{
		args.push_back(new ThreadArg5(*master_enc, *master_dec));
		for (int j = 0; j < max_num_states; j ++)
		{
			args[i]->encStates.push_back(new State);
			args[i]->encStates[j]->h = zeros_vec;
			args[i]->encStates[j]->c = zeros_vec;
			args[i]->decStates.push_back(new State);
		}
		args[i]->softmax = new Softmax(hidden_dim, tgt_voc_size);
		// args[i]->target_dist = VecD::Zero(tgt_voc_size);
        for (int j = 0; j < max_num_terms; j ++)
        {
            args[i]->delosBuffer.push_back(zeros_vec);
            args[i]->delisBuffer.push_back(zeros_vec);
            args[i]->delusBuffer.push_back(zeros_vec);
            args[i]->delfsBuffer.push_back(zeros_vec);
            args[i]->target_dist.push_back(VecD::Zero(tgt_voc_size));
        }

        for (int j = 0; j < max_num_tgt_terms; j ++)
        {
            args[i]->deltaFeatureBuffer;
            args[i]->gradWeightBuffer;
        }
	}

	double t0 = curTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic) shared(args, src_embd)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# start");
		int id = omp_get_thread_num();
		for (int j = 0; j < num_src_terms[i]; j ++)
		{
			forward(src_embd.col(src_data[id][j]), args[id]->encStates[j], args[id]->encStates[j+1], master_enc);
		}

		args[id]->decStates[0]->h = args[id]->encStates[num_src_terms[i]]->h;
		args[id]->decStates[0]->c = args[id]->encStates[num_src_terms[i]]->c;

		for (int j = 1; j < num_tgt_terms[i]; j ++)
		{
			forward(tgt_embd.col(tgt_data[id][j]), args[id]->decStates[j-1], args[id]->decStates[j], master_dec);
		}

		for (int j = 0; j < num_tgt_terms[i]; j ++)
		{
            // do softmax calculation here
            newSoftmaxCalcDist(args[id]->decStates[j]->h, args[id]->target_dist[j], softmax);
            newSoftmaxCalcLoss(args[id]->target_dist[j], tgt_data[id][j]);

            // assign value for priavteSoftmax grad_bias
            newSoftmaxBackwardPart1(args[id]->target_dist[j], tgt_data[id][j], args[id]->softmax);
        }
    
        for (int j = 0; j < num_tgt_terms[i]; j ++)
        {
            newSoftmaxBackwardPart2(args[id]->target_dist[j], args[id]->decStates[j]->delh, softmax);
        }
        /*
        while(1)
        {
            for (int j = 0; j < num_tgt_terms[i]; j ++)
            {
               // softmaxOperation(args[id]->decStates[j]->h, args[id]->target_dist, tgt_data[id][j], args[id]->decSates[j]->delh, softmax, args[id]->softmax); 
            }
        }
        */
        for (int index = 0; index < hidden_dim; index ++)
        {
            for (int j = 0; j < num_tgt_terms[i]; j ++)
            {
                newSoftmaxBackwardPart3(args[id]->decStates[j]->h, args[id]->target_dist[j], args[id]->softmax, index);
            }
        }
		args[id]->decStates[num_tgt_terms[i]]->delc = zeros_vec;
		args[id]->decStates[num_tgt_terms[i]]->delh = random_vec;
		args[id]->decStates[num_tgt_terms[i]-1]->delh = random_vec;
		
        /*
		std::vector<VecD> tgt_delos;
		std::vector<VecD> tgt_delis;
		std::vector<VecD> tgt_delus;
		std::vector<VecD> tgt_delfs;
        
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			tgt_delos.push_back(zeros);
			tgt_delis.push_back(zeros);
			tgt_delus.push_back(zeros);
			tgt_delfs.push_back(zeros);
		}
        */
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->decStates[j-1]->delc = zeros_vec;
			args[id]->decStates[j-1]->delh = random_vec;

			// backward(args[id]->decStates[j-1], args[id]->decStates[j], args[id]->tgt_grad, tgt_embd.col(tgt_data[id][j-1]), master_dec);
			// newBackward(args[id]->decStates[j-1], args[id]->decStates[j], master_dec, tgt_delos[j-1], tgt_delis[j-1], tgt_delus[j-1], tgt_delfs[j-1]);
			newBackward(args[id]->decStates[j-1], args[id]->decStates[j], master_dec, args[id]->delosBuffer[j-1], args[id]->delisBuffer[j-1], args[id]->delusBuffer[j-1], args[id]->delfsBuffer[j-1]);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Wxi.noalias() += args[id]->delisBuffer[j-1] * tgt_embd.col(tgt_data[id][j-1]);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Whi.noalias() += args[id]->delisBuffer[j-1] * args[id]->decStates[j-1]->h.transpose();
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Wxf.noalias() += args[id]->delfsBuffer[j-1] * tgt_embd.col(tgt_data[id][j-1]);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Whf.noalias() += args[id]->delfsBuffer[j-1] * args[id]->decStates[j-1]->h.transpose();
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Wxo.noalias() += args[id]->delosBuffer[j-1] * tgt_embd.col(tgt_data[id][j-1]);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Who.noalias() += args[id]->delosBuffer[j-1] * args[id]->decStates[j-1]->h.transpose();
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Wxu.noalias() += args[id]->delusBuffer[j-1] * tgt_embd.col(tgt_data[id][j-1]);
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.Whu.noalias() += args[id]->delusBuffer[j-1] * args[id]->decStates[j-1]->h.transpose();
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.bi += args[id]->delisBuffer[j-1];
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.bf += args[id]->delfsBuffer[j-1];
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.bo += args[id]->delosBuffer[j-1];
		}
		for (int j = num_tgt_terms[i]-1; j >= 1; j --)
		{
			args[id]->tgt_grad.bu += args[id]->delusBuffer[j-1];
		}
        /*
		std::vector<VecD> src_delos;
		std::vector<VecD> src_delis;
		std::vector<VecD> src_delfs;
		std::vector<VecD> src_delus;

		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			src_delos.push_back(zeros);
			src_delis.push_back(zeros);
			src_delus.push_back(zeros);
			src_delfs.push_back(zeros);
		}
        */

		args[id]->encStates[num_src_terms[i]]->delc = args[id]->decStates[0]->delc;
		args[id]->encStates[num_src_terms[i]]->delh = args[id]->decStates[0]->delh;

		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->encStates[j-1]->delh = zeros_vec;
			args[id]->encStates[j-1]->delc = zeros_vec;
			// backward(args[id]->encStates[j-1], args[id]->encStates[j], args[id]->src_grad, src_embd.col(src_data[id][j-1]), master_enc);
			// newBackward(args[id]->encStates[j-1], args[id]->encStates[j], master_enc, src_delos[j-1], src_delis[j-1], src_delus[j-1], src_delfs[j-1]);
		    newBackward(args[id]->encStates[j-1], args[id]->encStates[j], master_enc, args[id]->delosBuffer[j-1], args[id]->delisBuffer[j-1], args[id]->delusBuffer[j-1], args[id]->delfsBuffer[j-1]);
        }

		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Wxi.noalias() += args[id]->delisBuffer[j-1] * src_embd.col(src_data[id][j-1]);
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Whi.noalias() += args[id]->delisBuffer[j-1] * args[id]->encStates[j-1]->h.transpose();
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Wxf.noalias() += args[id]->delfsBuffer[j-1] * src_embd.col(src_data[id][j-1]);
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Whf.noalias() += args[id]->delfsBuffer[j-1] * args[id]->encStates[j-1]->h.transpose();
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Wxo.noalias() += args[id]->delosBuffer[j-1] * src_embd.col(src_data[id][j-1]);
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Who.noalias() += args[id]->delosBuffer[j-1] * args[id]->encStates[j-1]->h.transpose();
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Wxu.noalias() += args[id]->delusBuffer[j-1] * src_embd.col(src_data[id][j-1]);
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.Whu.noalias() += args[id]->delusBuffer[j-1] * args[id]->encStates[j-1]->h.transpose();
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.bi += args[id]->delisBuffer[j-1];
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.bf += args[id]->delfsBuffer[j-1];
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.bo += args[id]->delosBuffer[j-1];
		}
		for (int j = num_src_terms[i]; j >= 1; j --)
		{
			args[id]->src_grad.bu += args[id]->delusBuffer[j-1];
		}
		asm volatile("# end");
	}

	double t1 = curTime();

	std::cout << "time used = " << t1 - t0 << std::endl;

	return;
}
