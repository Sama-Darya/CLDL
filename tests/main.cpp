#include "../include/cldl/Net.h"
#include "unitTest.h"
#include <algorithm>

using namespace std;

int main(){

    unitTest* unitest;
    unitest = new unitTest();
    unitest->test_net_setInputs();
    unitest->test_net_porpInputs();
    unitest->test_net_masterPropagation();
    return 0;

}
