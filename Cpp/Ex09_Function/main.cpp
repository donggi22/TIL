#include <iostream>

using namespace std;

// 함수 (선언과 정의 분리 가능하다고 안내)
int Add(int a, int b);
// {
//     return a + b; // 반환값 안내
// }

// 반환 자료형이 지정되지 않았음
void Add(int a, int b, int *c, int *d) // 반환값을 여러개 받는 거처럼 사용하고 싶다면 매개변수로 포인터를 여러개 넣어주는 것이 사용되는 기법!
{
    *c = a + b; // c의 역참조
    *d = a - b;
}

int main()
{
    cout << Add(1, 2) << endl;

    int sum;
    int sub;
    Add(4, 5, &sum, &sub);

    cout << sum << " " << sub << endl;

    return 0;
}

int Add(int a, int b) // 위에서 선언 먼저함!
{
    return a + b; // 반환값 안내
}