link_libraries(testutil MPI_mat)

#
# mat 测试
#
add_executable(test+MPI_mat+mat mat.cpp)

target_compile_definitions(test+MPI_mat+mat
                           PRIVATE BOOST_TEST_MODULE=MPI_mat+mat)

add_test(NAME MPI_mat+mat COMMAND test+MPI_mat+mat)

target_code_coverage(test+MPI_mat+mat AUTO ALL)

#
# mpi 测试
#
add_executable(test+MPI_mat+mpi mpi.cpp)

target_compile_definitions(test+MPI_mat+mpi
                           PRIVATE BOOST_TEST_MODULE=MPI_mat+mpi)

add_test(NAME MPI_mat+mpi COMMAND test+MPI_mat+mpi)

target_code_coverage(test+MPI_mat+mpi AUTO ALL)
