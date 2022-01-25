cmake_minimum_required(VERSION 3.17)
project(mineracao-de-dados C CXX)

find_package(mlpack 3.4.2)
find_package(Boost 1.58 COMPONENTS serialization)
# find_package(armadillo)

add_subdirectory(include)
add_subdirectory(src)


