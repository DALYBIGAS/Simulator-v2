#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
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

static llvm::cl::opt<std::string> opName(
    "op-name", llvm::cl::desc("Name of the operation to replace (e.g., linalg.matmul)"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> functionName(
    "function-name", llvm::cl::desc("Name of the function to generate (e.g., my_matmul)"),
    llvm::cl::Required);

int main(int argc, char **argv) {
  // Initialize LLVM and command-line options
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Transform Program\n");

  // Create an MLIR context and load the necessary dialects
  MLIRContext context;
  context.loadDialect<func::FuncDialect, transform::TransformDialect>();

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

  // Find the operation to replace and extract its input and result types
  Operation *targetOp = nullptr;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == opName) {
      targetOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!targetOp) {
    llvm::errs() << "Error: Operation '" << opName << "' not found in the input MLIR file\n";
    return 1;
  }

  // Extract input and result types from the target operation
  SmallVector<Type, 4> inputTypes;
  SmallVector<Type, 4> resultTypes;
  for (Value operand : targetOp->getOperands()) {
    inputTypes.push_back(operand.getType());
  }
  for (Value result : targetOp->getResults()) {
    resultTypes.push_back(result.getType());
  }

  // Create a function declaration with the same input and result types as the target operation
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module->getBody());

  // Create the function type
  auto funcType = builder.getFunctionType(inputTypes, resultTypes);

  // Create the function declaration
  auto funcOp = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), functionName, funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Public);

  // Create a transform module to replace the specified operation with a function call
  auto transformModule = builder.create<transform::TransformModuleOp>(
      builder.getUnknownLoc(), "transform.with_named_sequence");

  // Create the named sequence within the transform module
  builder.setInsertionPointToStart(transformModule.getBody());
  auto namedSequence = builder.create<transform::NamedSequenceOp>(
      builder.getUnknownLoc(), "__transform_main",
      builder.getType<transform::AnyOpType>());

  // Add the entry block to the named sequence
  Block *entryBlock = namedSequence.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Match the specified operation
  Value anyOp = entryBlock->getArgument(0);
  auto matchOp = builder.create<transform::MatchOp>(
      builder.getUnknownLoc(), anyOp, builder.getArrayAttr({builder.getStringAttr(opName)}));

  // Replace the matched operation with a function call
  auto replaceOp = builder.create<transform::ReplaceOp>(
      builder.getUnknownLoc(), matchOp.getResult(), functionName);

  // Yield the results
  builder.create<transform::YieldOp>(builder.getUnknownLoc(), replaceOp.getResults());

  // Print the updated module to verify the output
  module->dump();

  return 0;
}