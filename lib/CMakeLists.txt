

add_llvm_pass_plugin(golly MODULE 
    golly.cpp
    Analysis/SyncBlockDetection.cpp
    Analysis/PscopDetection.cpp
    Analysis/DimensionDetection.cpp
    Support/SCEVValidator.cpp)
target_compile_features(golly PRIVATE cxx_std_20)
target_link_libraries(golly PUBLIC ${ISL_TARGET})