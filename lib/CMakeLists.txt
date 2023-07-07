

add_llvm_pass_plugin(golly MODULE PscopDetection.cpp)
target_link_libraries(golly PUBLIC ${ISL_TARGET})