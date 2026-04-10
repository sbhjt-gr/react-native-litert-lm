const { getDefaultConfig } = require("expo/metro-config");
const path = require("path");

const projectRoot = __dirname;
const libraryRoot = path.resolve(projectRoot, "..");

const config = getDefaultConfig(projectRoot);

// Watch the local library folder for changes
config.watchFolders = [libraryRoot];

const libraryNodeModules = path.resolve(libraryRoot, "node_modules");

// Helper to create regex for blocking
// We block the library's react and react-native to ensure we use the example app's versions
const modulesToBlock = ["react", "react-native"];
const blockRegexes = modulesToBlock.map(
  (moduleName) =>
    new RegExp(
      `^${escapeRegExp(path.join(libraryNodeModules, moduleName))}\\/.*`,
    ),
);

if (!config.resolver.blockList) {
  config.resolver.blockList = [];
}

if (Array.isArray(config.resolver.blockList)) {
  config.resolver.blockList.push(...blockRegexes);
} else {
  // Fallback for older Metro versions or if default config structure is different
  config.resolver.blockList = [
    ...(config.resolver.blockList || []),
    ...blockRegexes,
  ];
}

// Resolve the library's source files and force React resolution
config.resolver.extraNodeModules = {
  "@inferrlm/react-native-litert-lm": libraryRoot,
  react: path.resolve(projectRoot, "node_modules/react"),
  "react-native": path.resolve(projectRoot, "node_modules/react-native"),
};

config.resolver.nodeModulesPaths = [path.resolve(projectRoot, "node_modules")];

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

module.exports = config;
