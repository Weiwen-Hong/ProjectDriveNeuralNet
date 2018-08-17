#include <iostream>
using namespace std;

int main()
{
  char b;

  for (;;)
  {
    for (;;)
    {
      b = std::cin.get();
      if ((b == '\0') || (b == '\n'))
      {
        //new line or end of file was found,
        //skip handling that character
        break;
      }
      //do something with char b
      cout << "You typed: " << b << endl;
    }

    if (b == '\0') break; //exit outer loop if end of file detected

  }

  return 0;
}
