#
# Copyright (c) 2015, Richard Townsend and Contributors. All rights reserved. 
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

enable_testing()

function(add_test_binary name)
    add_executable(test_${name} ${name}.cpp)
    add_executable(test_${name}_s ${name}.cpp)
    target_link_libraries(test_${name} ${UNET_LIBRARIES} unet)
    target_link_libraries(test_${name}_s ${UNET_LIBRARIES} unets)
endfunction(add_test_binary)

# Credit: http://stackoverflow.com/questions/9303711 
find_program (MEMORYCHECK_COMMAND valgrind)
set (MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full")
function(add_memcheck_test name)
    add_test_binary(${name})
    set(memcheck_command ${CMAKE_MEMORYCHECK_COMMAND} ${CMAKE_MEMORYCHECK_COMMAND_OPTIONS})
    separate_arguments(memcheck_command)
    add_test(test_${name} test_${name} ${ARGN})
    add_test(memcheck_${name} ${memcheck_command} ./test_${name} ${ARGN})
endfunction(add_memcheck_test)

function(set_memcheck_test_properties name)
    set_tests_properties(${name} ${ARGN})
    set_tests_properties(memcheck_${name} ${ARGN})
endfunction(set_memcheck_test_properties)

function (add_standard_test name)
    add_test_binary(${name})
    add_test(test_${name} test_${name})
    add_test(test_static_${name}_s test_${name}_s)
endfunction(add_standard_test)

add_standard_test(BuildCheck)
add_memcheck_test(BuildCheckMem)
add_memcheck_test(BuildCheckVersion)
add_memcheck_test(LogCheckReplacement)
add_memcheck_test(FunctionCheckOpenCL)
add_memcheck_test(OpenCLNetworkFunctionCheck)
add_memcheck_test(OpenCLNetworkFunctionExecuteCheck)
add_standard_test(OpenCLDeviceCheck)
add_standard_test(OpenCLDeviceMemCheck)
