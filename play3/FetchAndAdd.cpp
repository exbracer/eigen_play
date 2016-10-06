/*************************************************************************
	> File Name: FetchAndAdd.cpp
	> Author: Yuchen Qiao
	> Mail: qiaoyc14@mails.tsinghua.edu.cn
	> Created Time: Wed 05 Oct 2016 05:28:06 PM CST
 ************************************************************************/

#include "FetchAndAdd.hpp"

Real atomic_real_fetch_and_add(Real* operand, Real incr)
{
    union 
    {
        Real d;
        uint64_t i;
    } old_val, new_val, ret_val;

    do
    {
        old_val.d = *(volatile Real *)operand;
        new_val.d = old_val.d + incr;
        __asm__ __volatile__ ("lock; cmpxchgq %1, (%2)"
                              :"=a" (ret_val.i)
                              :"r" (new_val.i), "r" (operand), "0" (old_val.i)
                              :"memory");
    } while (ret_val.i != old_val.i);

    return old_val.d;
}

void fetchAndAdd1(int argc, char** argv)
{
    int num_rows = HIDDEN_DIM;
    int num_cols = TGT_VOC_SIZE;

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
        MatD * mat = new MatD(num_rows, num_cols);
        *mat = MatD::Zero(num_rows, num_cols);
        mats.push_back(mat);
        // rnd.uniform(*mats[i], scale);
        vec1s.push_back(new VecD(num_rows));
        *vec1s[i] = random_vec1;
        vec2s.push_back(new VecD(num_cols));
        *vec2s[i] = random_vec2;
    }
    
    std::cout << "====================================" << std::endl;
    std::cout << "private method" << std::endl;

    double t0 = playCurrTime();
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0;i < batch_size; i ++)
    {
        asm volatile("# begin 1");
        int id = omp_get_thread_num();

        (*mats[id]).noalias() += *vec1s[id] * (*vec2s[id]).transpose();
        asm volatile("# end 1");
    }

    // double t1 = playCurrTime();
    // std::cout << "time used = " << t1 - t0 << std::endl;

    for (int i = 0; i < num_threads; i ++)
    {
        mat1 += *(mats[i]);
    }
    double t1 = playCurrTime();
    std::cout << "time used = " << t1 - t0 << std::endl;

    std::cout << "====================================" << std::endl;
    std::cout << "shared method" << std::endl;

    double t2 = playCurrTime();
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < batch_size; i ++)
    {
        asm volatile("# start 2");
        int id = omp_get_thread_num();
        // using sync_fetch_and_add instruction
        // 
        /*
        for (int j = 0; j < mat2.size(); j ++)
        {
            Real tmp = ((*vec1s[id])[j%num_rows])*((*vec2s[id])[j/num_rows]);
            // atomic_real_fetch_and_add(mat2.data()+i, tmp);
            *(mat2.data()+j) += tmp;
            // __sync_fetch_and_add(mat2.data()+i, tmp);
            // __sync_val_compare_and_swap(mat2.data()+i, *(mat2.data()+i), *(mat2.data()+i)+tmp);
        }
        */
        //
        for (int jj = 0; jj < num_cols; jj ++)
        {
            for (int ii = 0; ii < num_rows; ii ++)
            {
                Real tmp = (*vec1s[id])[ii] * (*vec2s[id])[jj];
                // *(mat2.data()+jj*num_rows+ii) += tmp;
                atomic_real_fetch_and_add(mat2.data()+jj*num_rows+ii, tmp);
            }
        }
        asm volatile("# end 2");
    }

    double t3 = playCurrTime();
    std::cout << "time used = " << t3 - t2 << std::endl;

    std::cout << "====================================" << std::endl;
    std::cout << "compare the result" << std::endl;

    int size1 = mat1.size();
    int size2 = mat2.size();
    std::cout << "size1 = " << size1 << std::endl;
    std::cout << "size2 = " << size2 << std::endl;
    
    Real sum = 0.0;
    for (int i = 0; i < size1; i ++)
    {
        Real diff = *(mat1.data() + i) - *(mat2.data()+i);
        /*
        if (fabs(diff) >= 1e-6)
        {
            std::cout << "diff = " << diff << std::endl;
            std::cout << "i = " << i << std::endl;
            std::cout << "mat1 coeff is " << *(mat1.data()+i) << std::endl;
            std::cout << "mat2 coeff is " << *(mat2.data()+i) << std::endl;
        }
        */
        
        sum += fabs(diff);
        
    }

    std::cout << "sum = " << sum << std::endl;
    
    return;
}
