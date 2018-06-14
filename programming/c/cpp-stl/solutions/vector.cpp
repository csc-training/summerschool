#include <iostream>
#include <vector>
#include <algorithm>


// Asks user a number using std::cin while simultaneously 
// making sure that it is really a number.
//
// NOTE: std::cin << X returns false if it cannot cast given 
// input value into X. Therefore, we loop until correct value 
// is given.
//
int ask_number() {
  int input;

  std::cout << "  Type a number: ";
  while( !(std::cin >> input) ) {
    std::cin.clear();
    std::cin.ignore(10000, '\n');
    std::cout << "Invalid input, try again: ";
  }

  return input;
}


int main()
{
  int N = 10; // number of integers we ask from user
  int number;

  std::vector<int> vec;


  std::cout << "Give me " << N << " numbers :\n";
  for(int i=0; i<N; i++) {
    number = ask_number();

    // appending elements into vector
    vec.push_back(number);
  }


  // printing vector
  std::cout << "You gave: ";
  for(auto i : vec) {
    std::cout << i << ", ";
  }
  std::cout << '\n';


  // sorting
  std::sort(vec.begin(), vec.end());


  // printing vector again
  std::cout << "In sorted form it is:";
  for(auto i : vec) {
    std::cout << i << ", ";
  }
  std::cout << '\n';


  int smallest, largest; 

  // std::min/max_element returns iterator; so we need to take the value
  // that they point into with * operator
  smallest = *std::min_element(vec.begin(), vec.end());
  largest  = *std::max_element(vec.begin(), vec.end());

  std::cout << "Smallest number is: " << smallest << '\n';
  std::cout << "Largest number is : " << largest << '\n';

}
