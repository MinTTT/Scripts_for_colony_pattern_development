cmake_minimum_required(VERSION 3.23)
project(colony_agent_based_SSA)
find_package(OpenMP)
set(CMAKE_CXX_STANDARD 17)

file(GLOB SRC_FILES *.cpp)
add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})