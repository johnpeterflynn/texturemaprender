#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>

namespace dnr
{
namespace time
{

std::string getTimeAsString(std::string prepend_str = "") {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << prepend_str << std::put_time(std::localtime(&in_time_t), "%m%d_%H%M%S");

    return ss.str();
}

}
}

#endif // UTILS_H
