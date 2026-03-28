#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// Command-line options.
static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input MLIR file>"), llvm::cl::Required);

static llvm::cl::opt<std::string> opChainStr(
    "op-chain",
    llvm::cl::desc("Comma separated list of operations to fuse in the form of \"[opA, opB, opC]\""),
    llvm::cl::Required);

static llvm::cl::opt<std::string> kernelName(
    "kernel-name",
    llvm::cl::desc("Final hardware kernel name after fusion (e.g., \"opA_opB_opC_fused_kernel\")"),
    llvm::cl::Required);

// New option: last op tile sizes as a comma-separated string.
static llvm::cl::opt<std::string> lastTileSizesStr(
    "last-tile-sizes",
    llvm::cl::desc("Tile sizes for last op tiling as a comma-separated string (e.g., \"4,4,4\")"),
    llvm::cl::init("8,32"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Op Fusion Transform\n");

  // Create an MLIR context and load the necessary dialects.
  MLIRContext context;
  context.loadDialect<func::FuncDialect, transform::TransformDialect>();

  // Parse the input MLIR file.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << "Error opening file: " << errorMessage << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  OwningModuleRef module = parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error: Failed to parse the input MLIR file\n";
    return 1;
  }

  // Create the transform module with a named sequence.
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module->getBody());
  auto transformModule = builder.create<transform::TransformModuleOp>(
      builder.getUnknownLoc(), "transform.with_named_sequence");

  builder.setInsertionPointToStart(transformModule.getBody());
  auto namedSequence = builder.create<transform::NamedSequenceOp>(
      builder.getUnknownLoc(), "__fuse_transform_main",
      builder.getType<transform::AnyOpType>());
  Block *entryBlock = namedSequence.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  Value anyOp = entryBlock->getArgument(0);

  // Parse the op chain.
  llvm::StringRef opChainRef(opChainStr);
  opChainRef = opChainRef.trim();
  if (opChainRef.startswith("[")) opChainRef = opChainRef.drop_front(1);
  if (opChainRef.endswith("]")) opChainRef = opChainRef.drop_back(1);
  SmallVector<llvm::StringRef, 4> opNames;
  opChainRef.split(opNames, ",", /*KeepEmpty=*/false);
  SmallVector<Attribute, 4> opNameAttrs;
  for (auto name : opNames)
    opNameAttrs.push_back(builder.getStringAttr(name.trim()));

  if (opNameAttrs.empty()) {
    llvm::errs() << "Error: No operations provided in op chain.\n";
    return 1;
  }

  // Parse lastTileSizesStr into a vector of int64_t.
  SmallVector<int64_t, 4> lastTileSizes;
  SmallVector<llvm::StringRef, 4> tileTokens;
  llvm::split(tileTokens, lastTileSizesStr, ',');
  for (auto token : tileTokens) {
    token = token.trim();
    int64_t num;
    if (token.getAsInteger(10, num)) {
      llvm::errs() << "Error parsing tile size token: " << token << "\n";
      return 1;
    }
    lastTileSizes.push_back(num);
  }

  // In the fusion pipeline, we start from the last op in the chain (to be tiled)
  // and then iterate in reverse order to fuse preceding ops.
  // Match the last op in the chain.
  auto currentHandle = builder.create<transform::MatchOp>(
      builder.getUnknownLoc(), anyOp, opNameAttrs.back());

  // Iterate in reverse order, starting from secondLast to first.
  for (int i = opNameAttrs.size() - 2; i >= 0; --i) {
    // For the current fusion, retrieve the operand of the current handle.
    // This simulates obtaining the producer handle for fusion.
    // Here we assume index 0; in a real scenario, an index selection may be needed.
    auto operandHandle = builder.create<transform::GetOperandOp>(
         builder.getUnknownLoc(), currentHandle.getResult(), builder.getI64IntegerAttr(0));
    
    // Match the preceding op from the chain.
    auto prevMatch = builder.create<transform::MatchOp>(
         builder.getUnknownLoc(), anyOp, opNameAttrs[i]);
    
    // Check if the operand handle "contains" the matched op.
    // (In this simplified example, we assume it does. In practice, a verification
    // step may be added.)
    
    // If this is the first fusion (i.e. fusing into the tiled op),
    // tile the current op handle using the provided tile sizes.
    // We only tile once for the last op.
    if (i == opNameAttrs.size() - 2) {
      auto tiledOp = builder.create<transform::TileUsingForallOp>(
          builder.getUnknownLoc(), currentHandle.getResult(), builder.getI64ArrayAttr(lastTileSizes));
      // Assume tileOp returns two results, second is loop handle.
      currentHandle = tiledOp.getResult(1); // update current handle to loop handle.
    }
    
    // Fuse the preceding matched op into the current loop handle.
    auto fuseOp = builder.create<transform::FuseIntoContainingOp>(
         builder.getUnknownLoc(), prevMatch.getResult(), currentHandle,
         builder.getStringAttr(kernelName));
    
    // fuseOp returns two results; update currentHandle to its second result.
    currentHandle = fuseOp.getResult(1);
  }

  // Finally, yield the fused handle.
  builder.create<transform::YieldOp>(builder.getUnknownLoc(), currentHandle);
  
  // Dump the updated module for verification.
  module->dump();
  return 0;
}
