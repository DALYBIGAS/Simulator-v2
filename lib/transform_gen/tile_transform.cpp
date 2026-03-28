#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// Command-line options
static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input MLIR file>"), llvm::cl::Required);

static llvm::cl::opt<std::string> matchOpName(
    "match-op", llvm::cl::desc("Operation name to match (e.g., linalg.matmul)"),
    llvm::cl::init("linalg.matmul"));

static llvm::cl::list<int64_t> tileSizes(
    "tile-sizes", llvm::cl::desc("Tile sizes for tiling (e.g., 4 4 4)"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);

int main(int argc, char **argv) {
  // Initialize LLVM and command-line options
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Transform Code Generator\n");

  // Create an MLIR context and load the necessary dialects
  MLIRContext context;
  context.loadDialect<StandardOpsDialect, linalg::LinalgDialect, transform::TransformDialect>();

  // Parse the input MLIR file
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << "Error: " << errorMessage << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  OwningModuleRef module = parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error: Failed to parse the input MLIR file\n";
    return 1;
  }

  // Create a transform module with the named sequence
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module->getBody());

  // Create the transform module with the named sequence
  auto transformModule = builder.create<transform::TransformModuleOp>(
      builder.getUnknownLoc(), "transform.with_named_sequence");

  // Create the named sequence within the transform module
  builder.setInsertionPointToStart(transformModule.getBody());
  auto namedSequence = builder.create<transform::NamedSequenceOp>(
      builder.getUnknownLoc(), "__transform_main",
      builder.getType<transform::AnyOpType>());

  // Add the entry block to the named sequence
  // FIXME: namedSequenceOp has no addEntryBlock() function
  Block *entryBlock = namedSequence.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Create the match operation for the specified operation name
  Value anyOp = entryBlock->getArgument(0);
  auto matchOp = builder.create<transform::MatchOp>(
      builder.getUnknownLoc(), anyOp, builder.getArrayAttr({builder.getStringAttr(matchOpName)}));

  // Create the tile operation using for loops with the specified tile sizes
  SmallVector<int64_t> tileSizesVec(tileSizes.begin(), tileSizes.end());
  auto tileOp = builder.create<transform::TileUsingForOp>(
      builder.getUnknownLoc(), matchOp.getResult(), tileSizesVec);

  // Yield the results
  builder.create<transform::YieldOp>(builder.getUnknownLoc(), tileOp.getResults());

  // Print the updated module to verify the output
  module->dump();

  return 0;
}