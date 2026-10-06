#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer
{
  public:
    Timer() { startTimer(); }

    void startTimer() { start = std::chrono::high_resolution_clock::now(); }

    void endTimer(const std::string& message)
    {
        end = std::chrono::high_resolution_clock::now();

        std::chrono::microseconds elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // std::cout << message << ": " << formatWithDots(elapsed_time.count()) << " microseconds\n";
    }

  private:
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::time_point();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::time_point();

    // helper to format number with dots every 3 digits
    std::string formatWithDots(long long value)
    {
        std::string str = std::to_string(value);
        int insertPosition = str.length() - 3;

        while (insertPosition > 0)
        {
            str.insert(insertPosition, ".");
            insertPosition -= 3;
        }

        return str;
    }
};
