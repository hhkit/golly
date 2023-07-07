

add_llvm_pass_plugin(golly MODULE 
    golly.cpp
    PscopDetection.cpp
    DimensionDetection.cpp)
target_link_libraries(golly PUBLIC ${ISL_TARGET})