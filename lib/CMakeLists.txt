

add_llvm_pass_plugin(golly MODULE 
    golly.cpp
    Analysis/ConditionalDominanceAnalysis.cpp
    Analysis/CudaParameterDetection.cpp
    Analysis/PolyhedralBuilder.cpp
    Analysis/Pscop.cpp
    Analysis/PscopDetector.cpp
    Analysis/RaceDetection.cpp
    Analysis/SccOrdering.cpp
    Analysis/StatementDetection.cpp
    Analysis/Statements.cpp
    ErrorHandling/Error.cpp
    ErrorHandling/YamlDumper.cpp
    Support/ConditionalAtomizer.cpp
    Support/GollyOptions.cpp
    Support/isl.cpp
    Support/isl_llvm.cpp)
target_compile_features(golly PRIVATE cxx_std_20)
target_link_libraries(golly PRIVATE ${ISL_TARGET} ryml::ryml)