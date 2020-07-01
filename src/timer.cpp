#include "timer.h"

#include <iostream>

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

    std::cout << "Benchmarks: Name, Duration\n";
    for (size_t i = 0; i < m_time.size() - 1; ++i) {
        auto duration = m_time[i+1] - m_time[i];
        std::cout << "Time: " << std::chrono::duration_cast<std::chrono::seconds>(duration).count() << "\n";
        std::cout << m_checkpoint_name[i] << ": "
                  << std::chrono::duration_cast<std::chrono::seconds>(duration).count() << "\n";
    }

    restart();
}