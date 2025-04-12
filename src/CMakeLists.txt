add_executable(randomforest randomforest.cpp)
add_executable(svm svm.cpp)
# add_executable(adaboost adaboost.cpp)
add_executable(mlpack_test sample_program.cpp)

target_link_libraries(randomforest mlpack armadillo Boost::serialization)
target_link_libraries(svm mlpack armadillo Boost::serialization)
# target_link_libraries(adaboost mlpack armadillo Boost::serialization)
target_link_libraries(mlpack_test mlpack armadillo)

# install(TARGETS mlpack_test
	# DESTINATION ${CMAKE_BINARY_DIR})
