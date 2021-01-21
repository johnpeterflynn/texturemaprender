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

static std::string getTimeAsString(std::string prepend_str = "") {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << prepend_str << std::put_time(std::localtime(&in_time_t), "%m%d_%H%M%S");

    return ss.str();
}
}

namespace format
{
static std::string intToString(int n, int length) {
   std::stringstream ss;

   // Pad remaining length with zeros
   ss << std::setw(length) << std::setfill('0') << n;
   std::string s = ss.str();

   return s;
}
}
}

#endif // UTILS_H
