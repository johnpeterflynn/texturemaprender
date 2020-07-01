#include "timer.h"

#include <iostream>
#include <iomanip>

Timer::Timer()  {

}

void Timer::restart() {
    m_time.clear();
    m_checkpoint_name.clear();
}

void Timer::restart(std::string name) {
    restart();
    checkpoint(name);
}

void Timer::checkpoint(std::string name) {
    auto now = std::chrono::high_resolution_clock::now();

    m_time.push_back(now);
    m_checkpoint_name.push_back(name);

    m_last_clock = now;
}

void Timer::end() {
    checkpoint("");

    std::cout << "-----Benchmarking-----\n";
    for (size_t i = 0; i < m_time.size() - 1; ++i) {
        std::chrono::duration<float> duration = m_time[i+1] - m_time[i];
        std::cout << "T" << std::setfill('0') << std::setw(2) << i << ": "
                << std::fixed << duration.count() << ", "
                  << m_checkpoint_name[i] << "\n";
    }
    std::chrono::duration<float> total_duration = m_time[m_time.size() - 1] - m_time[0];
    std::cout << "Total Time: " << std::fixed << total_duration.count() << "\n";
    std::cout << "----------------------\n";

    restart();
}
