

add_llvm_pass_plugin(golly MODULE 
    golly.cpp
    Analysis/ConditionalDominanceAnalysis.cpp
    Analysis/CudaParameterDetection.cpp
    Analysis/PolyhedralBuilder.cpp
    Analysis/PscopDetector.cpp
    Analysis/PscopBuilder.cpp
    Analysis/RaceDetection.cpp
    Analysis/SccOrdering.cpp
    Analysis/StatementDetection.cpp
    Analysis/Statements.cpp
    Support/ConditionalAtomizer.cpp
    Support/isl.cpp
    Support/isl_llvm.cpp)
target_compile_features(golly PRIVATE cxx_std_20)
target_link_libraries(golly PUBLIC ${ISL_TARGET} fmt::fmt)