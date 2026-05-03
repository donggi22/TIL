#include <iostream>

using namespace std;

int main()
{
    // cin은 데이터를 흘려넣어 보내는 스트림이고
    // 그 데이터를 해석하는 것은 자료형
    // 자료형에 따라서 알아서 처리해주기 때문에 scanf()보다 편리

    char user_input[100];

    // cin과 getline의 차이

    // cout << "원하는 문장을 입력해주세요." << endl;
    // cout << "입력: ";

    // cin >> user_input; // 공백 줄바꿈은 받아 오지 않음
    
    
    // cin.getline(user_input, sizeof(user_input)); // 공백 줄바꿈 받아옴

    // cout << "메아리: " << user_input << endl;
    
    int number = -1; // -1로 초기화
    int number2 = -1;

    cin >> user_input;
    // cin.getline(user_input, sizeof(user_input));
    cin.ignore(100, '\n'); // 최대 100글자까지만 입력 받아서 무시 또는 '\n'이 있으면 그때부터 무시
    
    cin >> number;
    // cin >> number2;

    // cin >> number >> number2;

    // cout << number << " " << number2 << endl;
    cout << user_input << " " << number << endl;

    // 참고: cin.ignore(numeric_limits<streamsize>::max(), '\n')

    return 0;
}