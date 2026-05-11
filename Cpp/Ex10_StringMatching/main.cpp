#include <iostream>

using namespace std;

const int kMaxStr = 100; // 전역 변수 소개

// 문자열을 매개변수로 넣기
// 여기서는 모든 문자열 배열의 길이가 동일하다고 가정


int Len(const char str0[])
{
    int i = 0;

    while (1)
    {
        if (str0[i] != '\0')
        {
            i++;
        }
        else
            return i;
    }
}

bool IsEqual(const char str1[], const char str2[])
{
    // 크기 출력 확인 (배열 크기가 아님 주의!) - 문자열의 길이를 별도로 저장해야 합니다!
    // cout << sizeof(str1) << " " << sizeof(str2) << " " << endl;
    // cout << sizeof(int(str1)) << " " << sizeof(int(str2)) << " " << endl;
    // str1과 str2는 포인터
    // exit(-1);
    
    
    // 힌트: ==, != 같지 않다 비교 연산자
    // 힌트: 문자열 종료 조건
    // 디버깅 힌트: 문자를 정수로 바꿔서 출력해보기
    
    int i = 0;

    if (Len(str1) != Len(str2))
    {
        return false;
    }

    int bigger_len = 0;

    if (Len(str1) >= Len(str2))
    {
        bigger_len = Len(str1);
    }
    else
    {
        bigger_len = Len(str2);
    }
    while (i < bigger_len)
    {
        if (str1[i] == str2[i])
        {
            i++;
        }
        else
        {
            return false;
        }
    }
    return true;
}

int main()
{
    // 영어 사용이 디버깅에 유리합니다.
    const char str1[kMaxStr] = "stop";

    while (1)
    {
        // TODO:
        // char str2[kMaxStr] = "hello";
        char str2[kMaxStr];
        cin >> str2;
        
        // cout << IsEqual(str1, str2) << endl;

        if (IsEqual(str1, str2))
        {
            cout << "종료합니다." << endl;
            break;
        }
        else
        {
            cout << "계속합니다." << endl;
        }
    }
    return 0;
}