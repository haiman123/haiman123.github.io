#include<iostream>
#include<string>
using namespace std;
int main()
{
    const string hexDigits = "0123456789abcdef";
    cout << "Enter a series of number between 0 and 15"
	 << " separated by spaces. Hit ENTER when finished:"
	 << endl;
    string result;
    string::size_type n;
    while(cin>>n){
	if(n==0){
	    break;
	}
	if(n<hexDigits.size()){
            result += hexDigits[n];
	}
    }
    cout<<"Your hex number is:"<<result<<endl;
    return 0;
}
