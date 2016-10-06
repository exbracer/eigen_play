/*******************************************
 *
 *
 *
 *
 ******************************************/
#ifndef __INCLUDE_HPP__
#define __INCLUDE_HPP__
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>

#include <vector>
#include <map>

#include "omp.h"
#include "Matrix.hpp"
#include "ActFunc.hpp"
#include "BlackOut.hpp"

#define INPUT_DIM 512
#define HIDDEN_DIM 512
#define NUM_SRC_TERMS 12
#define NUM_TGT_TERMS 15
#define MAX_NUM_STATES 200
#define BATCH_SIZE 128
#define MAX_NUM_SRC_TERMS 16
#define MIN_NUM_SRC_TERMS 10
#define MAX_NUM_TGT_TERMS 20
#define MIN_NUM_TGT_TERMS 10

#define TRAIN_DATA_SIZE 4096
#define TEST_DATA_SIZE 4096
#define SRC_VOC_SIZE 6812
#define TGT_VOC_SIZE 6354

double playCurrTime();
void playGenRandInt(std::vector<int>& array, int len, int min, int max);
void playGenRandIntDynamic(std::vector<int>& array, int len, int min, int max);
double playGetMeanIntArray(std::vector<int>& array);

#endif


