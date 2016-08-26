/*************************************************************************
    > File Name: Bench.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月23日 星期二 18时30分56秒
 ************************************************************************/

#include "Bench.hpp"

void bench1(int argc, char** argv)
{
	std::cout << "bench1" << std::endl;
	// int num_rows = INPUT_DIM;
	// int num_cols = TGT_VOC_SIZE;

	int num_rows = 6*1024;
	int num_cols = INPUT_DIM;


	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	Rand rnd;
	const Real scale = 0.1;
	int max_batch_count= 32;

	std::vector<MatD*> mats;
	std::vector<VecD*> vec1s;
	std::vector<VecD*> vec2s;
	VecD random_vec1 = VecD::Random(num_rows);
	VecD random_vec2 = VecD::Random(num_cols);
	
	for (int i = 0; i < num_threads; i ++)
	{
		mats.push_back(new MatD(num_rows, num_cols));
		rnd.uniform(*mats[i], scale);
		vec1s.push_back(new VecD(num_rows));
		*vec1s[i] = random_vec1;
		vec2s.push_back(new VecD(num_cols));
		*vec2s[i] = random_vec2;
	}

	double t0 = playCurrTime();
	
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
		for (int j = 0; j < batch_size; j ++)
		{
	for (int i = 0; i < max_batch_count; i ++)
	{

			asm volatile("# begin");
			int id = omp_get_thread_num();

			//(*mats[id]) += *vec1s[id] * (*vec2s[id]).transpose();
			(*mats[id]) *= 0.1;
			asm volatile("# end");
		}
	}
	double t1 = playCurrTime();

	std::cout << "time used = " << t1 - t0 << std::endl;

	return;

}

void bench2(int argc, char** argv)
{
	std::cout << "bench2" << std::endl;
	int num_rows = INPUT_DIM;
	int num_cols = TGT_VOC_SIZE;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	Rand rnd;
	const Real scale = 0.1;
	int max_batch_count= 40;

	std::vector<MatD*> mats;
	std::vector<VecD*> vec1s;
	std::vector<VecD*> vec2s;
	VecD random_vec1 = VecD::Random(num_rows);
	VecD random_vec2 = VecD::Random(num_cols);
	
	for (int i = 0; i < num_threads; i ++)
	{
		mats.push_back(new MatD(num_rows, num_cols));
		rnd.uniform(*mats[i], scale);
		vec1s.push_back(new VecD(num_rows));
		*vec1s[i] = random_vec1;
		vec2s.push_back(new VecD(num_cols));
		*vec2s[i] = random_vec2;
	}

	double t0 = playCurrTime();
	for (int i = 0; i < max_batch_count; i ++)
	{
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
		for (int j = 0; j < batch_size; j ++)
		{
			asm volatile("# begin");
			int id = omp_get_thread_num();

			for (int ii = 0; ii < num_cols; ii ++)
			{
				for (int jj = 0; jj < num_rows; jj ++)
				{
					mats[id]->coeffRef(jj, ii) += vec1s[id]->coeff(jj,0) * vec2s[id]->coeff(ii,0);
				}
			}
		
			asm volatile("# end");
		}
	}
	double t1 = playCurrTime();

	std::cout << "time used = " << t1 - t0 << std::endl;

	return;

}

void bench3(int argc, char** argv)
{
	std::cout << "bench4" << std::endl;
	int num_rows = INPUT_DIM;
	int num_cols = TGT_VOC_SIZE;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	Rand rnd;
	const Real scale = 0.1;
	int max_batch_count= 40;

	std::vector<MatD*> mats;
	std::vector<VecD*> vec1s;
	std::vector<VecD*> vec2s;
	VecD random_vec1 = VecD::Random(num_rows);
	VecD random_vec2 = VecD::Random(num_cols);
	
	for (int i = 0; i < num_threads; i ++)
	{
		mats.push_back(new MatD(num_rows, num_cols));
		rnd.uniform(*mats[i], scale);
		vec1s.push_back(new VecD(num_rows));
		*vec1s[i] = random_vec1;
		vec2s.push_back(new VecD(num_cols));
		*vec2s[i] = random_vec2;
	}

	double t0 = playCurrTime();
	for (int i = 0; i < max_batch_count; i ++)
	{
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
		for (int j = 0; j < batch_size; j ++)
		{
			asm volatile("# begin");
			int id = omp_get_thread_num();

			for (int ii = 0; ii < num_cols; ii ++)
			{
				for (int jj = 0; jj < num_rows; jj ++)
				{
					*(mats[id]->data()+ii*num_rows+jj) += vec1s[id]->coeff(jj,0) * vec2s[id]->coeff(ii,0);
				}
			}
		
			asm volatile("# end");
		}
	}
	double t1 = playCurrTime();

	std::cout << "time used = " << t1 - t0 << std::endl;

	return;

}

void bench4(int argc, char** argv)
{
	std::cout << "bench4" << std::endl;
	int num_rows = TGT_VOC_SIZE;
	int num_cols = INPUT_DIM;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	Rand rnd;
	const Real scale = 0.1;
	int max_batch_count= 40;

	std::vector<MatD*> mats;
	std::vector<VecD*> vec1s;
	std::vector<VecD*> vec2s;
	VecD random_vec1 = VecD::Random(num_rows);
	VecD random_vec2 = VecD::Random(num_cols);
	
	for (int i = 0; i < num_threads; i ++)
	{
		mats.push_back(new MatD(num_rows, num_cols));
		rnd.uniform(*mats[i], scale);
		vec1s.push_back(new VecD(num_rows));
		*vec1s[i] = random_vec1;
		vec2s.push_back(new VecD(num_cols));
		*vec2s[i] = random_vec2;
	}

	double t0 = playCurrTime();
	for (int i = 0; i < max_batch_count; i ++)
	{
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
		for (int j = 0; j < batch_size; j ++)
		{
			asm volatile("# begin");
			int id = omp_get_thread_num();

			for (int ii = 0; ii < num_cols; ii ++)
			{
				for (int jj = 0; jj < num_rows; jj ++)
				{
					*(mats[id]->data()+ii*num_rows+jj) += vec1s[id]->coeff(jj,0) * vec2s[id]->coeff(ii,0);
				}
			}
		
			asm volatile("# end");
		}
	}
	double t1 = playCurrTime();

	std::cout << "time used = " << t1 - t0 << std::endl;

	return;

}

void bench5(int argc, char** argv)
{
	std::cout << "bench 5" << std::endl;

	int num_rows = 6*1024;
	int num_cols = HIDDEN_DIM;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	int offset1 = atoi(argv[2]);
	int offset2 = atoi(argv[3]);

	int max_batch_count = 32;

	std::vector<Real*> mats;
	std::vector<Real*> vec1s;
	std::vector<Real*> vec2s;
	std::vector<Real> x_set;
	unsigned short rg[3] = {
		1, 2, 3
	};
	for (int i = 0;i < num_threads; i ++)
	{
		mats.push_back(new Real[num_rows*num_cols]);
		for (int ii = 0; ii < num_rows; ii ++)
		{
			for (int jj = 0; jj < num_cols; jj ++)
			{
				*(mats[i]+ii*num_cols+jj) = erand48(rg);
			}
		}
		vec1s.push_back(new Real[num_rows]);
		for (int ii = 0; ii < num_rows; ii ++)
		{
			*(vec1s[i]+ii) = erand48(rg);
		}
		vec2s.push_back(new Real[num_cols]);
		for (int ii = 0; ii < num_cols; ii ++)
		{
			*(vec2s[i]+ii) = erand48(rg);
		}
		x_set.push_back(erand48(rg));
	}
	std::cout << "init finished" << std::endl;
	double t0 = playCurrTime();

	//for (int i = 0;i < max_batch_count; i ++)
	//{
#pragma omp parallel for num_threads(num_threads) schedule(static)
		for (int j = 0; j < batch_size; j ++)
		{
	for (int i = 0; i < max_batch_count; i ++)
	{

			asm volatile("# begin");
			int id = omp_get_thread_num();
			Real* tmp_mat = mats[id];
			Real* tmp_vec1 = vec1s[id];
			Real* tmp_vec2 = vec2s[id];
			/*
			for (int ii = 0; ii < num_cols; ii += offset2)
			{
				for (int jj = 0; jj < num_rows; jj += offset1)
				{
					// tmp_mat[ii*num_rows+jj] += tmp_vec1[jj] * tmp_vec2[ii];
					// tmp_mat[ii*num_rows+jj]+=x_set[id];
					for (int kk = ii; kk < ii+offset2; kk ++)
					{
						for (int ll = jj; ll < jj+offset1; ll ++)
						{
							tmp_mat[kk*num_rows+ll] += tmp_vec1[ll] * tmp_vec2[kk];
						}
					}
				}
			}
			*/

			for (int ii = 0; ii < num_cols; ii ++)
			{
				for (int jj = 0; jj < num_rows; jj ++)
				{
					tmp_mat[ii*num_rows+jj] += 0.1;
				}
			}
			asm volatile("# end");
		}
	}

	double t1 = playCurrTime();

	std::cout << "time used = " << t1 - t0 << std::endl;


	return;
}


void bench6(int argc, char** argv)
{
	std::cout << "bench6" << std::endl;
	// int num_rows = INPUT_DIM;
	// int num_cols = TGT_VOC_SIZE;

	int num_rows = 6*1024;
	int num_cols = INPUT_DIM;


	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	Rand rnd;
	const Real scale = 0.1;
	int max_batch_count= 32;

	std::vector<MatD*> mats;
	std::vector<VecD*> vec1s;
	std::vector<VecD*> vec2s;
	VecD random_vec1 = VecD::Random(num_rows);
	VecD random_vec2 = VecD::Random(num_cols);
	
	for (int i = 0; i < num_threads; i ++)
	{
		// mats.push_back(new MatD(num_rows, num_cols));
		// rnd.uniform(*mats[i], scale);
		MatD * mat = NULL;
		VecD * vec1 = NULL;
		VecD * vec2 = NULL;
		mats.push_back(mat);
		vec1s.push_back(vec1);
		vec2s.push_back(vec2);
	}
#pragma omp parallel num_threads(num_threads)
	{
		int id = omp_get_thread_num();
		mats[id] = new MatD(num_rows, num_cols);
		rnd.uniform(*mats[id], scale);
		vec1s[id] = new VecD(num_rows);
		vec2s[id] = new VecD(num_cols);
	}

	double t0 = playCurrTime();
	
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
		for (int j = 0; j < batch_size; j ++)
		{
	for (int i = 0; i < max_batch_count; i ++)
	{

			asm volatile("# begin");
			int id = omp_get_thread_num();

			//(*mats[id]) += *vec1s[id] * (*vec2s[id]).transpose();
			(*mats[id]) *= 0.1;
			asm volatile("# end");
		}
	}
	double t1 = playCurrTime();

	std::cout << "time used = " << t1 - t0 << std::endl;

	return;

}


void bench7(int argc, char** argv)
{
	std::cout << "bench 7" << std::endl;

	int num_rows = 6*1024;
	int num_cols = HIDDEN_DIM;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	int offset1 = atoi(argv[2]);
	int offset2 = atoi(argv[3]);

	int max_batch_count = 32;

	std::vector<Real*> mats;
	std::vector<Real*> vec1s;
	std::vector<Real*> vec2s;
	std::vector<Real> x_set;
	unsigned short rg[3] = {
		1, 2, 3
	};

	for (int i = 0; i < num_threads; i ++)
	{
		Real * mat = NULL;
		Real * vec1 = NULL;
		Real * vec2 = NULL;
		mats.push_back(mat);
		vec1s.push_back(vec1);
		vec2s.push_back(vec2);
	}
#pragma omp parallel num_threads(num_threads)
	{
		int id = omp_get_thread_num();
		mats[id] = new Real[num_rows*num_cols];
		for (int ii = 0; ii < num_rows; ii ++)
		{
			for (int jj = 0; jj < num_cols; jj ++)
			{
				*(mats[id]+ii*num_cols+jj) = erand48(rg);
			}
		}
		vec1s[id] = new Real[num_rows];
		for (int ii = 0; ii < num_rows; ii ++)
		{
			*(vec1s[id]+ii) = erand48(rg);
		}
		vec2s[id] = new Real[num_cols];
		for (int ii = 0; ii < num_cols; ii ++)
		{
			*(vec2s[id]+ii) = erand48(rg);
		}
	}
	std::cout << "init finished" << std::endl;
	double t0 = playCurrTime();

	//for (int i = 0;i < max_batch_count; i ++)
	//{
#pragma omp parallel for num_threads(num_threads) schedule(static)
		for (int j = 0; j < batch_size; j ++)
		{
	for (int i = 0; i < max_batch_count; i ++)
	{

			asm volatile("# begin");
			int id = omp_get_thread_num();
			Real* tmp_mat = mats[id];
			Real* tmp_vec1 = vec1s[id];
			Real* tmp_vec2 = vec2s[id];
			
			for (int ii = 0; ii < num_cols; ii += offset2)
			{
				for (int jj = 0; jj < num_rows; jj += offset1)
				{
					// tmp_mat[ii*num_rows+jj] += tmp_vec1[jj] * tmp_vec2[ii];
					// tmp_mat[ii*num_rows+jj]+=x_set[id];
					for (int kk = ii; kk < ii+offset2; kk ++)
					{
						for (int ll = jj; ll < jj+offset1; ll ++)
						{
							tmp_mat[kk*num_rows+ll] += tmp_vec1[ll] * tmp_vec2[kk];
						}
					}
				}
			}
			
			/*
			for (int ii = 0; ii < num_cols; ii ++)
			{
				for (int jj = 0; jj < num_rows; jj ++)
				{
					tmp_mat[ii*num_rows+jj] *= 0.1;
				}
			}
			*/
			asm volatile("# end");
		}
	}

	double t1 = playCurrTime();

	std::cout << "time used = " << t1 - t0 << std::endl;


	return;
}
