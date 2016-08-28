/*************************************************************************
    > File Name: main.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: 2016年08月26日 星期五 16时42分12秒
 ************************************************************************/

#include "Include.hpp"
#include "Func.hpp"
#include "TestFunc.hpp"
#include "NewTestFunc.hpp"

// main function here
int main(int argc, char** argv)
{
	
	std::cout << "test function 32" << std::endl;
	testFunc32(argc, argv);
	std::cout << std::endl;
	
	std::cout << "new test function 1 " << std::endl;
	newTestFunc1(argc, argv);
	std::cout << std::endl;

	return 0;
}
