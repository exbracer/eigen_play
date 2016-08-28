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

void printStart()
{
	std::cout << "start" << std::endl;
}
void playEigen1(int argc, char** argv)
{
	MatD mat1(3, 3);
	MatD mat2(3, 3);
	MatD mat3(3, 3);
	MatD mat4(3, 3);
	mat1 << 1,2,3,4,5,6,7,8,9;
	mat2 << 1,2,3,4,5,6,7,8,9;
	mat3 << 1,2,3,4,5,6,7,8,9;
	mat4 << 1,2,3,4,5,6,7,8,9;
	std::cout << "mat1 is " << std::endl;
	std::cout << mat1 << std::endl;
	std::cout << "mat2 is " << std::endl;
	std::cout << mat2 << std::endl;
	std::cout << "mat3 is " << std::endl;
	std::cout << mat3 << std::endl;
	std::cout << "mat4 is " << std::endl;
	std::cout << mat4 << std::endl;
	std::cout << "******************" << std::endl;
	printStart();
	
	// mat1 = mat2 + mat3; 
	// malloc() not called
	
	// mat1 = mat2 * mat3; 
	// malloc() called, hit 1 times, not call central cache, 
	// tc_malloc ()size = 68)
	
	// mat1.noalias() = mat2 * mat3; 
	// malloc not called
	
	// mat1 = mat2 + mat3 * mat4; 
	// malloc called, hit 1 times, not call central cache
	// tc_malloc (size = 68) 
	
	// mat1 = mat1 + mat3 * mat4; 
	// malloc called, hit 1 times, not call central cache
	// tc_malloc (size = 68) 

	// mat1.noalias() = mat2 + mat3 * mat4;
	// malloc not called
	
	// mat1.noalias() = mat1 + mat3 * mat4;
	// malloc not called
	
	// mat1 += mat3 * mat4;
	// malloc called, hit 1 time

	mat1.noalias() += mat3 * mat4;
	// malloc not called

	std::cout << "mat1 = " << std::endl;
	std::cout << mat1 << std::endl;

	return;
}

void playEigen2(int argc, char** argv)
{
	MatD mat1(3, 3);
	MatD mat2(3, 3);
	MatD mat3(3, 3);

	VecD vec1(3);
	VecD vec2(3);

	mat1 << 1,2,3,4,5,6,7,8,9;
	mat2 << 1,2,3,4,5,6,7,8,9;
	mat3 << 1,2,3,4,5,6,7,8,9;

	vec1 << 1, 2, 3;
	vec2 << 1, 2, 3;

	std::cout << "mat1 is " << std::endl;
	std::cout << mat1 << std::endl;
	std::cout << "mat2 is " << std::endl;
	std::cout << mat2 << std::endl;
	std::cout << "mat3 is " << std::endl;
	std::cout << mat3 << std::endl;

	std::cout << "vec1 is" << std::endl;
	std::cout << vec1 << std::endl;
	std::cout << "vec2 is" << std::endl;
	std::cout << vec2 << std::endl;

	std::cout << "***********************" << std::endl;
	printStart();

	// mat1 = vec1 * vec2.transpose();
	// malloc called 1 time
	// tc_malloc (size = 68)
	
	// mat1 = mat2.transpose();
	// malloc not called
	
	mat1 += vec1 * vec2.transpose();
	// malloc called 1 time
	// tc_malloc (size = 68)
	std::cout << "after calculation" << std::endl;
	std::cout << "mat1 is " << std::endl;
	std::cout << mat1 << std::endl;
	/*
	std::cout << "in memory :" << std::endl;
	for (int i = 0; i < mat1.size(); i ++)
	{
		std::cout << *(mat1.data()+i) << " ";

	}
	std::cout << std::endl;
	*/
	return;
}

void playEigen3(int argc, char** argv)
{
	int num_rows = INPUT_DIM;
	int num_cols = HIDDEN_DIM;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	Rand rnd;
	const Real scale = 0.1;

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
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();

		(*mats[id]).noalias() += *vec1s[id] * (*vec2s[id]).transpose();
		asm volatile("# end");
	}
	double t1 = playCurrTime();

	std::cout << "time used = " << t1 - t0 << std::endl;
	return;
}

void playEigen4(int argc, char** argv)
{
	int num_rows = INPUT_DIM;
	int num_cols = HIDDEN_DIM;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	Rand rnd;
	const Real scale = 0.1;

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
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# begin");
		int id = omp_get_thread_num();

		// (*mats[id]).noalias() += *vec1s[id] * (*vec2s[id]).transpose();
		
		for (int j = 0; j < num_cols; j ++)
		{
			mats[id]->col(j).array() += vec1s[id]->array() * vec2s[id]->coeff(j, 0);
		}
		
		asm volatile("# end");
	}
	double t1 = playCurrTime();

	std::cout << "time used = " << t1 - t0 << std::endl;
	return;
}

void playEigen5(int argc, char ** argv)
{
	int num_rows = TGT_VOC_SIZE;
	int num_cols = HIDDEN_DIM;

	int num_threads = atoi(argv[1]);
	int batch_size = BATCH_SIZE;
	
	Rand rnd;
	const Real scale = 0.1;

	std::vector<MatD*> mats;
	std::vector<VecD*> vec1s;
	std::vector<VecD*> vec2s;
	VecD random_vec1 = VecD::Random(num_rows);
	VecD random_vec2 = VecD::Random(num_cols);

	MatD mat1 = MatD::Zero(num_rows, num_cols);
	MatD mat2 = MatD::Zero(num_rows, num_cols);

	for (int i = 0; i < num_threads; i ++)
	{
		mats.push_back(new MatD(num_rows, num_cols));
		//rnd.uniform(*mats[i], scale);
		*mats[i] = mat1;
		vec1s.push_back(new VecD(num_rows));
		*vec1s[i] = random_vec1;
		vec2s.push_back(new VecD(num_cols));
		*vec2s[i] = random_vec2;
	}
	std::cout << "==================================" << std::endl;
	std::cout << "private method" << std::endl;
	double t0 = playCurrTime();

#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# start 1");
		int id = omp_get_thread_num();
		(*mats[id]).noalias() += (*vec1s[id]) * (*vec2s[id]).transpose(); 
		asm volatile("# end 1");
	}
	double t1 = playCurrTime();
	std::cout << "time used = " << t1 - t0 << std::endl;

	for (int i = 0; i < num_threads; i ++)
	{
		mat1 += (*mats[i]);
	}

	std::cout << "==================================" << std::endl;
	std::cout << "shared method" << std::endl;

	double t2 = playCurrTime();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
	for (int i = 0; i < batch_size; i ++)
	{
		asm volatile("# start 2");
		int id = omp_get_thread_num();
		mat2.noalias() += (*vec1s[id]) * (*vec2s[id]).transpose();
		asm volatile("# end 2");
	}
	double t3 = playCurrTime();
	std::cout << "time used = " << t3 - t2 << std::endl;

	std::cout << "===================================" << std::endl;
	std::cout << "compare the result" << std::endl;

	int size1 = mat1.size();
	int size2 = mat2.size();
	std::cout << "size1 = " << size1 << std::endl;
	std::cout << "size2 = " << size2 << std::endl;
	Real sum = 0.0;
	for (int i = 0; i < size1; i ++)
	{
		Real diff = *(mat1.data() + i) - *(mat2.data()+i);
		if (fabs(diff) >= 1e-3)
		{
			std::cout << "diff = " << diff << std::endl;
			std::cout << "i = " << i << std::endl;
			std::cout << "mat1 coeff is " << *(mat1.data()+i) << std::endl;
			std::cout << "mat2 coeff is " << *(mat2.data()+i) << std::endl;
		}
		sum += diff*diff;

	}
	std::cout << "sum = " << sum << std::endl;

	return;
}
