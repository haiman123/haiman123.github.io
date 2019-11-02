#include<iostream>
using namespace std;
int main(){
    int i = 42;
    int &r1 = i;
    const int &r2 = i;
    printf("r2 = %d",r2);
    r1 = 0;
    printf("r2 = %d",r2);
    printf("i=%d",i);
    return 0;
}
