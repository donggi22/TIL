/*
    홍정모 연구소 https://honglab.co.kr
*/

#include <iostream> // iostream이라는 헤더를 포함(include)

using namespace std; // 네임스페이스 설명 std::cout

int main() // entry point
{
    // 주석(comment) 다는 방법

    cout << "Hello, World" << endl; // 자료형 신경 덜 쓰면서 콘솔 출력 가능
    // printf("Hello World!!! by printf");
    // 기본 출력 여기서 설명해야 함

    char user_input[100];
    cin >> user_input;
    cout << user_input;

    return 0;
}

/*
g++ c:\Users\dev\Desktop\TIL\Cpp\Ex01_HelloWorld\main.cpp -o HelloWorld.exe

C++ 메모리 레이아웃 https://www.geeksforgeeks.org/cpp/memory-layout-of-cpp-program/

(TIL) PS C:\Users\dev\Desktop\TIL> size .\Cpp\Ex01_HelloWorld\main.exe
   text    data     bss     dec     hex filename
  11252    1920     384   13556    34f4 .\Cpp\Ex01_HelloWorld\main.exe
*/