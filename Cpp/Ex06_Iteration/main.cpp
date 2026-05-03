#include <iostream>

using namespace std;

int main()
{
    // For 기본 예제
    /*for (int i = 5; i <= 10; i++)
    {
        cout << i << " ";
    }
    cout << endl;*/


    // 배열 데이터 출력 연습 문제로 제공
    // 힌트: sizeof(my_array) / sizeof(int)
    /*int my_array[] = {1, 2, 3, 4, 5, 4, 3, 2, 1};
    // cout << sizeof(my_array) / sizeof(int);
    for (int i=0; i < sizeof(my_array) / sizeof(int); i++)
    {
        cout << my_array[i] << " ";
    }
    cout << endl;*/


    // 문자열 출력
    char my_string[] = "Hello\0, World!";

    // 문자열을 한 글자씩 출력하기 (indexing을 이용하여 '\0'을 만날 때까지)
    // cout << my_string << endl; 사용 X
    // 힌트: sizeof(), '\0', break,

    // cout << sizeof(my_string) / sizeof(char);

    // cout << sizeof(char);

    /*for (int i = 0; i < sizeof(my_string); i++)
    {
        if (my_string[i] == '\0')
        {
            break;
        }
        cout << my_string[i];
    }
    cout << endl;

    for (int i = 0; my_string[i] != '\0'; i++)
    {
        cout << my_string[i];
    }
    cout << endl;

    for (int i = 0; my_string[i] != '\n'; i++)
    {
        cout << i << " " << my_string[i] << endl;
    }
    cout << endl;*/

    /*int j;
    for (int i = 0; i < sizeof(my_string) / sizeof(char); i++)
    {         
        if (my_string[i] == '\0')
        {
            j = i;
            // cout << j << endl; // 5
            break;
        }
    }

    // cout << j << endl; // 5
    // cout << sizeof(j) << endl; // 4

    for (int i = 0; i < j; i++)
    {
        cout << my_string[i];
    }*/

    /*int j = 0;
    for (int i = 0; i < sizeof(my_string) / sizeof(char); i++)
    {
        switch (my_string[i])
        {
            case '\0':
                j = i;
                break;
            default:
                cout << my_string[i];
        }
        if (j != 0)
        {
            break;
        }
    }

    // cout << '\n' << j;*/

    // while 기본 예제
    /*
    int i = 0;
    while (i < 10)
    {
        cout << i << " ";
        i++; // 무한 반복 주의 안내
    }
    cout << endl;
    */

    // 실습 문제

    /*
    int i = 0;
    while(true) // for(;true;) // 초기화도 비워두고 변화부분도 비워둬서 while문과 동일하게 사용 가능
    {
        // 이 구조에서 똑같이 정수 출력하도록 만들게 하기 (break)
        // i++;
        cout << i << " ";
        i++;

        if (i >= 10)
        {
            break;
        }
    }
    */

    // 런타임 오류 주의
    // while문으로 문자열 한글자씩 출력하기
    // 힌트 && logical and

    /*
    int i = 0;
    while (i < sizeof(my_string))
    {   
        if (my_string[i] == '\0')
        {
            break;
        }
        cout << my_string[i];
        i++;
    }
    cout << endl;
    */

    char my_string2[] = "Hello, World!";
    int i = 0;
    while (i < sizeof(my_string2) && my_string2[i] != '\0')
    {
        cout << my_string2[i];
        i ++;
    }
    cout << endl;

    return 0;
}