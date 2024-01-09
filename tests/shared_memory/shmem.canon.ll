; ModuleID = '<stdin>'
source_filename = "/home/hoi/repos/golly/tests/basic/shared_memory/shmem.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_threadIdx_t = type { i8 }

@_ZZ4testPiE3arr = internal addrspace(3) global [256 x i32] undef, align 4
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@p = external dso_local addrspace(3) global [0 x i32], align 4

; Function Attrs: convergent mustprogress norecurse nounwind
define dso_local void @_Z4testPi(i32* noundef %a) #0 !dbg !11 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !dbg !15
  %idxprom = zext i32 %0 to i64, !dbg !19
  %arrayidx1 = getelementptr inbounds [256 x i32], [256 x i32] addrspace(3)* @_ZZ4testPiE3arr, i64 0, i64 %idxprom, !dbg !19
  %arrayidx = addrspacecast i32 addrspace(3)* %arrayidx1 to i32*, !dbg !19
  store i32 0, i32* %arrayidx, align 4, !dbg !20, !tbaa !21
  %arrayidx32 = getelementptr inbounds [0 x i32], [0 x i32] addrspace(3)* @p, i64 0, i64 %idxprom, !dbg !25
  %arrayidx3 = addrspacecast i32 addrspace(3)* %arrayidx32 to i32*, !dbg !25
  store i32 1, i32* %arrayidx3, align 4, !dbg !26, !tbaa !21
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx75,+sm_60" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.dbg.cu = !{!6}
!nvvm.annotations = !{!8}
!llvm.ident = !{!9, !10}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 5]}
!1 = !{i32 7, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !7, producer: "Ubuntu clang version 14.0.0-1ubuntu1.1", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly, splitDebugInlining: false, nameTableKind: None)
!7 = !DIFile(filename: "/home/hoi/repos/golly/tests/basic/shared_memory/shmem.cu", directory: "/tmp/golly/04665be8-5636-48e6-a01b-8752c9a0e7ca")
!8 = !{void (i32*)* @_Z4testPi, !"kernel", i32 1}
!9 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!10 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!11 = distinct !DISubprogram(name: "test", scope: !12, file: !12, line: 4, type: !13, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !6, retainedNodes: !14)
!12 = !DIFile(filename: "/home/hoi/repos/golly/tests/basic/shared_memory/shmem.cu", directory: "")
!13 = !DISubroutineType(types: !14)
!14 = !{}
!15 = !DILocation(line: 53, column: 3, scope: !16, inlinedAt: !18)
!16 = distinct !DISubprogram(name: "__fetch_builtin_x", scope: !17, file: !17, line: 53, type: !13, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !6, retainedNodes: !14)
!17 = !DIFile(filename: "/usr/lib/llvm-14/lib/clang/14.0.0/include/__clang_cuda_builtin_vars.h", directory: "")
!18 = distinct !DILocation(line: 8, column: 7, scope: !11)
!19 = !DILocation(line: 8, column: 3, scope: !11)
!20 = !DILocation(line: 8, column: 20, scope: !11)
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !DILocation(line: 9, column: 3, scope: !11)
!26 = !DILocation(line: 9, column: 18, scope: !11)
!27 = !DILocation(line: 11, column: 1, scope: !11)
