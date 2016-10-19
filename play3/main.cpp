/*************************************************************************
    > File Name: main.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月16日 星期二 14时34分42秒
 ************************************************************************/

#include "PlayEigen.hpp"
#include "Bench.hpp"
#include "FetchAndAdd.hpp"
#include "NewTestFunc.hpp"

int main(int argc, char** argv)
{
    // fetchAndAdd1(argc, argv);
    std::cout << "new test function 1" << std::endl;
    newTestFunc1(argc, argv);
    std::cout << std::endl;
    std::cout << "new test function 2" << std::endl;
    newTestFunc2(argc, argv);
    std::cout << std::endl;
	std::cout << "new test function 3" << std::endl;
	newTestFunc3(argc, argv);
	std::cout << std::endl;
	return 0;
}
