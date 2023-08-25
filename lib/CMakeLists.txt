

add_llvm_pass_plugin(golly MODULE 
    golly.cpp
    Analysis/CudaParameterDetection.cpp
    Analysis/PscopBuilder.cpp
    Analysis/RaceDetection.cpp
    Analysis/StatementDetection.cpp
    Analysis/Statements.cpp
    Support/isl.cpp
    Support/isl_llvm.cpp)
target_compile_features(golly PRIVATE cxx_std_20)
target_link_libraries(golly PUBLIC ${ISL_TARGET} fmt::fmt)