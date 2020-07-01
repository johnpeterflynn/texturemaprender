#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <vector>
#include <string>

class Timer {
public:
    static Timer& get() {
      static Timer instance;
      return instance;
    }

    void restart();
    void restart(std::string name);
    void checkpoint(std::string name);
    void end();

private:
    Timer();

    std::vector<std::chrono::high_resolution_clock::time_point> m_time;
    std::vector<std::string> m_checkpoint_name;

    std::chrono::high_resolution_clock::time_point m_last_clock;
};

#endif // TIMER_H
