; ModuleID = '<stdin>'
source_filename = "/home/hoi/repos/golly/tests/basic/fn_inline.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }

@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1

; Function Attrs: convergent mustprogress nounwind
define dso_local void @_Z8inlinemePii(i32* noundef %arr, i32 noundef %i) #0 !dbg !11 {
entry:
  %idxprom = sext i32 %i to i64, !dbg !15
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom, !dbg !15
  store i32 2, i32* %arrayidx, align 4, !dbg !16, !tbaa !17
  ret void, !dbg !21
}

; Function Attrs: convergent mustprogress norecurse nounwind
define dso_local void @_Z7fn_testPi(i32* noundef %arr) #1 !dbg !22 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !23
  %idxprom.i = sext i32 %0 to i64, !dbg !27
  %arrayidx.i = getelementptr inbounds i32, i32* %arr, i64 %idxprom.i, !dbg !27
  store i32 2, i32* %arrayidx.i, align 4, !dbg !29, !tbaa !17
  ret void, !dbg !30
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

attributes #0 = { convergent mustprogress nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx75,+sm_60" }
attributes #1 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx75,+sm_60" }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

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
!7 = !DIFile(filename: "/home/hoi/repos/golly/tests/basic/fn_inline.cu", directory: "/tmp/golly/6b0a65bb-4b60-4ebc-a551-b58f9df71f3b")
!8 = !{void (i32*)* @_Z7fn_testPi, !"kernel", i32 1}
!9 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!10 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!11 = distinct !DISubprogram(name: "inlineme", scope: !12, file: !12, line: 6, type: !13, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !6, retainedNodes: !14)
!12 = !DIFile(filename: "/home/hoi/repos/golly/tests/basic/fn_inline.cu", directory: "")
!13 = !DISubroutineType(types: !14)
!14 = !{}
!15 = !DILocation(line: 6, column: 45, scope: !11)
!16 = !DILocation(line: 6, column: 52, scope: !11)
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C++ TBAA"}
!21 = !DILocation(line: 6, column: 57, scope: !11)
!22 = distinct !DISubprogram(name: "fn_test", scope: !12, file: !12, line: 8, type: !13, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !6, retainedNodes: !14)
!23 = !DILocation(line: 66, column: 3, scope: !24, inlinedAt: !26)
!24 = distinct !DISubprogram(name: "__fetch_builtin_x", scope: !25, file: !25, line: 66, type: !13, scopeLine: 66, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !6, retainedNodes: !14)
!25 = !DIFile(filename: "/usr/lib/llvm-14/lib/clang/14.0.0/include/__clang_cuda_builtin_vars.h", directory: "")
!26 = distinct !DILocation(line: 10, column: 13, scope: !22)
!27 = !DILocation(line: 6, column: 45, scope: !11, inlinedAt: !28)
!28 = distinct !DILocation(line: 11, column: 3, scope: !22)
!29 = !DILocation(line: 6, column: 52, scope: !11, inlinedAt: !28)
!30 = !DILocation(line: 12, column: 1, scope: !22)
