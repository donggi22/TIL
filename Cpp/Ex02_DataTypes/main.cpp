#include <iostream>

using namespace std;

int main()
{
    // 변수를 정의할 때 자료형을 미리 지정해야 합니다.
    // 자료형은 바꿀 수 없습니다.

    // 내부적으로 메모리를 이미 갖고 있습니다.
    int i;   // 변수 정의
    i = 123; // 변수에 값 지정 (객체 레퍼런스 아님)


    // sizeof 소개
    // 정수는 4byte
    // cout << i << " " << sizeof(i) << endl; // 추측해보세요.

    // cout << sizeof(int) << endl;

    // cout << 123 + 4 << " " << sizeof(123 + 4) << endl;

    float f = 123.456f; // 마지막 f 주의
    double d = 123.456; // f 불필요

    // cout << f << " " << sizeof(f) << endl; // 123.456 4
    // cout << d << " " << sizeof(f) << endl; // 123.456 8


    // C++는 글자 하나와 문자열을 구분합니다.
    char c = 'a'; // 글자 하나 홑따옴표
    char str[] = "Hello, World!"; // 문자열(문자의 배열) 겹따옴표, std::string

    // cout << c << " " << sizeof(c) << endl; // a 1
    // cout << str << " " << sizeof(str) << endl; // Hello, World! 14

    // 그 외에도 다양한 자료형이 존재합니다.


    // 형변환
    i = 987.654; // double을 int에 강제로 저장

    // cout << "int from double " << i << endl;

    f = 567.89; // 이것도 형변환. f 없으면 double


    // 기본 연산자
    
    // i = 987;
    i += 100; // i = i + 100;
    i++;      // i = i + 1;

    // cout << i << endl; // 추측해보세요.


    // 불리언
    bool is_good = true;
    is_good = false;

    // cout << is_good << endl; // 0

    // is_good2 = true;
    // cout << is_good2 << endl; // 1

    // cout << boolalpha << true << endl;    // true
    // cout << is_good << endl;              // false
    // cout << noboolalpha << true << endl;  // 1
    // cout << is_good << endl;              // 0


    // 논리 연산 몇 가지 소개
    // https://en.cppreference.com/w/cpp/language/operator_precedence

    // cout << boolalpha;
    // cout << (true && true) << endl; // 논리연산자가 <<(insertion)연산자 보다 연산 우선 순위가 낮아서 괄호
    // cout << (true || false) << endl;


    // 비교

    // cout << boolalpha;
    // cout << (1 > 3) << endl;
    // cout << (3 == 3) << endl;
    // cout << ( i >= 3) << endl;
    // cout << ('a' != 'c') << endl;
    // cout << ('a' != 'a') << endl;


    // 영역

    i = 123; // 더 넓은 영역

    {
        i = 345;
        // int i = 345; // <- 더 좁은 영역의 다른 변수
        cout << i << endl; // 추측해보세요.
    }
    
    cout << i << endl; // 추측해보세요.

    return 0;
}