#include <iostream>
#include <random> // 난수 생성

using namespace std;

int main()
{
    // 난수 생성
    // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 99); // [1, 99]

    int number = distrib(gen);

    while (1) // true 대신 숫자 1로 무한 반복도 많이 사용합니다.
    {   
        int guess;
        cout << "입력: ";
        cin >> guess;

        if (number > guess)
        {
            cout << "너무 작아요!" << endl;
        }
        else if (number < guess)
        {
            cout << "너무 커요!" << endl;
        }
        else
        {
            cout << "정답!" << endl;
            break;
        }
    }
    return 0;
}