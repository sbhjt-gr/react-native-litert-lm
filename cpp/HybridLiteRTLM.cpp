//
// HybridLiteRTLM.cpp
// react-native-litert-lm
//
// High-performance LLM inference using LiteRT-LM C API.
//
// NOTE: This C++ implementation is used for iOS ONLY.
// Android uses the Kotlin implementation in `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/HybridLiteRTLM.kt`.
// Do not assume changes here will affect Android.
//

#include "HybridLiteRTLM.hpp"




#include <NitroModules/Promise.hpp>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <sys/stat.h>
#include <cstdio>

#ifdef __APPLE__
#include "IOSDownloadHelper.h"
#include <dlfcn.h>
#include <os/proc.h>
#endif
#include <fstream>
#include <thread>
#include <regex>
#include <pthread.h>
#include <functional>

namespace margelo::nitro::litertlm {

// =============================================================================
// Thread Helper — LiteRT engine operations need >512KB stack (XNNPack, Metal)
// =============================================================================

static void runOnLargeStack(std::function<void()> work, size_t stackSize = 8 * 1024 * 1024) {
  struct Context {
    std::function<void()> fn;
    std::exception_ptr exception;
  };
  Context ctx{std::move(work), nullptr};

  pthread_t thread;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, stackSize);

  int rc = pthread_create(&thread, &attr, [](void* arg) -> void* {
    auto* c = static_cast<Context*>(arg);
    try {
      c->fn();
    } catch (...) {
      c->exception = std::current_exception();
    }
    return nullptr;
  }, &ctx);
  pthread_attr_destroy(&attr);
  if (rc != 0) {
    throw std::runtime_error("Failed to create large-stack thread (errno: " + std::to_string(rc) + ")");
  }
  pthread_join(thread, nullptr);

  if (ctx.exception) {
    std::rethrow_exception(ctx.exception);
  }
}

#ifdef __APPLE__
struct LiteRtApi {
  using SetMinLogLevelFn = void (*)(int);
  using EngineSettingsCreateFn = LiteRtLmEngineSettings* (*)(const char*, const char*, const char*, const char*);
  using EngineSettingsDeleteFn = void (*)(LiteRtLmEngineSettings*);
  using EngineSettingsSetMaxNumTokensFn = void (*)(LiteRtLmEngineSettings*, int);
  using EngineSettingsEnableBenchmarkFn = void (*)(LiteRtLmEngineSettings*);
  using EngineSettingsSetCacheDirFn = void (*)(LiteRtLmEngineSettings*, const char*);
  using EngineCreateFn = LiteRtLmEngine* (*)(LiteRtLmEngineSettings*);
  using EngineDeleteFn = void (*)(LiteRtLmEngine*);
  using GetLastErrorFn = const char* (*)();
  using JsonResponseGetStringFn = const char* (*)(const LiteRtLmJsonResponse*);
  using JsonResponseDeleteFn = void (*)(LiteRtLmJsonResponse*);
  using ConversationDeleteFn = void (*)(LiteRtLmConversation*);
  using SessionConfigDeleteFn = void (*)(LiteRtLmSessionConfig*);
  using NewSessionConfigCreateFn = LiteRtLmSessionConfig* (*)();
  using OldSessionConfigCreateFn = LiteRtLmSessionConfig* (*)(const LiteRtLmSamplerParams*);
  using SessionConfigSetMaxOutputTokensFn = void (*)(LiteRtLmSessionConfig*, int);
  using SessionConfigSetSamplerParamsFn = void (*)(LiteRtLmSessionConfig*, const LiteRtLmSamplerParams*);
  using ConversationConfigCreateFn = LiteRtLmConversationConfig* (*)(LiteRtLmEngine*, const LiteRtLmSessionConfig*, const char*, const char*, const char*, bool);
  using ConversationConfigDeleteFn = void (*)(LiteRtLmConversationConfig*);
  using NewConversationCreateFn = LiteRtLmConversation* (*)(LiteRtLmEngine*, LiteRtLmConversationConfig*);
  using OldConversationCreateFn = LiteRtLmConversation* (*)(LiteRtLmEngine*);
  using NewSendMessageFn = LiteRtLmJsonResponse* (*)(LiteRtLmConversation*, const char*, const char*);
  using OldSendMessageFn = LiteRtLmJsonResponse* (*)(LiteRtLmConversation*, const char*);
  using NewSendMessageStreamFn = int (*)(LiteRtLmConversation*, const char*, const char*, LiteRtLmStreamCallback, void*);
  using OldSendMessageStreamFn = int (*)(LiteRtLmConversation*, const char*, LiteRtLmStreamCallback, void*);
  using GetBenchmarkInfoFn = LiteRtLmBenchmarkInfo* (*)(LiteRtLmConversation*);
  using GetNumDecodeTurnsFn = int (*)(const LiteRtLmBenchmarkInfo*);
  using GetDecodeTokensPerSecAtFn = double (*)(const LiteRtLmBenchmarkInfo*, int);
  using GetDecodeTokenCountAtFn = int (*)(const LiteRtLmBenchmarkInfo*, int);
  using GetTimeToFirstTokenFn = double (*)(const LiteRtLmBenchmarkInfo*);
  using DeleteBenchmarkInfoFn = void (*)(LiteRtLmBenchmarkInfo*);

  SetMinLogLevelFn setMinLogLevel;
  EngineSettingsCreateFn createEngineSettings;
  EngineSettingsDeleteFn deleteEngineSettings;
  EngineSettingsSetMaxNumTokensFn setEngineMaxTokens;
  EngineSettingsEnableBenchmarkFn enableEngineBenchmark;
  EngineSettingsSetCacheDirFn setEngineCacheDir;
  EngineCreateFn createEngine;
  EngineDeleteFn deleteEngine;
  GetLastErrorFn getLastError;
  JsonResponseGetStringFn getJsonResponseString;
  JsonResponseDeleteFn deleteJsonResponse;
  ConversationDeleteFn deleteConversation;
  SessionConfigDeleteFn deleteSessionConfig;
  NewSessionConfigCreateFn createSessionConfigNew;
  OldSessionConfigCreateFn createSessionConfigOld;
  SessionConfigSetMaxOutputTokensFn setMaxOutputTokens;
  SessionConfigSetSamplerParamsFn setSamplerParams;
  ConversationConfigCreateFn createConversationConfig;
  ConversationConfigDeleteFn deleteConversationConfig;
  NewConversationCreateFn createConversationNew;
  OldConversationCreateFn createConversationOld;
  NewSendMessageFn sendMessageNew;
  OldSendMessageFn sendMessageOld;
  NewSendMessageStreamFn sendMessageStreamNew;
  OldSendMessageStreamFn sendMessageStreamOld;
  GetBenchmarkInfoFn getBenchmarkInfo;
  GetNumDecodeTurnsFn getNumDecodeTurns;
  GetDecodeTokensPerSecAtFn getDecodeTokensPerSecAt;
  GetDecodeTokenCountAtFn getDecodeTokenCountAt;
  GetTimeToFirstTokenFn getTimeToFirstToken;
  DeleteBenchmarkInfoFn deleteBenchmarkInfo;

  bool hasModernSessionApi() const {
    return setMaxOutputTokens != nullptr && setSamplerParams != nullptr;
  }

  bool hasModernConversationApi() const {
    return createConversationConfig != nullptr &&
           deleteConversationConfig != nullptr &&
           createConversationNew != nullptr;
  }

  bool hasBenchmarkApi() const {
    return getBenchmarkInfo != nullptr &&
           getNumDecodeTurns != nullptr &&
           getDecodeTokensPerSecAt != nullptr &&
           getDecodeTokenCountAt != nullptr &&
           getTimeToFirstToken != nullptr &&
           deleteBenchmarkInfo != nullptr;
  }
};

static const LiteRtApi& getLiteRtApi() {
  static const LiteRtApi api{
    reinterpret_cast<LiteRtApi::SetMinLogLevelFn>(dlsym(RTLD_DEFAULT, "litert_lm_set_min_log_level")),
    reinterpret_cast<LiteRtApi::EngineSettingsCreateFn>(dlsym(RTLD_DEFAULT, "litert_lm_engine_settings_create")),
    reinterpret_cast<LiteRtApi::EngineSettingsDeleteFn>(dlsym(RTLD_DEFAULT, "litert_lm_engine_settings_delete")),
    reinterpret_cast<LiteRtApi::EngineSettingsSetMaxNumTokensFn>(dlsym(RTLD_DEFAULT, "litert_lm_engine_settings_set_max_num_tokens")),
    reinterpret_cast<LiteRtApi::EngineSettingsEnableBenchmarkFn>(dlsym(RTLD_DEFAULT, "litert_lm_engine_settings_enable_benchmark")),
    reinterpret_cast<LiteRtApi::EngineSettingsSetCacheDirFn>(dlsym(RTLD_DEFAULT, "litert_lm_engine_settings_set_cache_dir")),
    reinterpret_cast<LiteRtApi::EngineCreateFn>(dlsym(RTLD_DEFAULT, "litert_lm_engine_create")),
    reinterpret_cast<LiteRtApi::EngineDeleteFn>(dlsym(RTLD_DEFAULT, "litert_lm_engine_delete")),
    reinterpret_cast<LiteRtApi::GetLastErrorFn>(dlsym(RTLD_DEFAULT, "litert_lm_get_last_error")),
    reinterpret_cast<LiteRtApi::JsonResponseGetStringFn>(dlsym(RTLD_DEFAULT, "litert_lm_json_response_get_string")),
    reinterpret_cast<LiteRtApi::JsonResponseDeleteFn>(dlsym(RTLD_DEFAULT, "litert_lm_json_response_delete")),
    reinterpret_cast<LiteRtApi::ConversationDeleteFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_delete")),
    reinterpret_cast<LiteRtApi::SessionConfigDeleteFn>(dlsym(RTLD_DEFAULT, "litert_lm_session_config_delete")),
    reinterpret_cast<LiteRtApi::NewSessionConfigCreateFn>(dlsym(RTLD_DEFAULT, "litert_lm_session_config_create")),
    reinterpret_cast<LiteRtApi::OldSessionConfigCreateFn>(dlsym(RTLD_DEFAULT, "litert_lm_session_config_create")),
    reinterpret_cast<LiteRtApi::SessionConfigSetMaxOutputTokensFn>(dlsym(RTLD_DEFAULT, "litert_lm_session_config_set_max_output_tokens")),
    reinterpret_cast<LiteRtApi::SessionConfigSetSamplerParamsFn>(dlsym(RTLD_DEFAULT, "litert_lm_session_config_set_sampler_params")),
    reinterpret_cast<LiteRtApi::ConversationConfigCreateFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_config_create")),
    reinterpret_cast<LiteRtApi::ConversationConfigDeleteFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_config_delete")),
    reinterpret_cast<LiteRtApi::NewConversationCreateFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_create")),
    reinterpret_cast<LiteRtApi::OldConversationCreateFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_create")),
    reinterpret_cast<LiteRtApi::NewSendMessageFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_send_message")),
    reinterpret_cast<LiteRtApi::OldSendMessageFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_send_message")),
    reinterpret_cast<LiteRtApi::NewSendMessageStreamFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_send_message_stream")),
    reinterpret_cast<LiteRtApi::OldSendMessageStreamFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_send_message_stream")),
    reinterpret_cast<LiteRtApi::GetBenchmarkInfoFn>(dlsym(RTLD_DEFAULT, "litert_lm_conversation_get_benchmark_info")),
    reinterpret_cast<LiteRtApi::GetNumDecodeTurnsFn>(dlsym(RTLD_DEFAULT, "litert_lm_benchmark_info_get_num_decode_turns")),
    reinterpret_cast<LiteRtApi::GetDecodeTokensPerSecAtFn>(dlsym(RTLD_DEFAULT, "litert_lm_benchmark_info_get_decode_tokens_per_sec_at")),
    reinterpret_cast<LiteRtApi::GetDecodeTokenCountAtFn>(dlsym(RTLD_DEFAULT, "litert_lm_benchmark_info_get_decode_token_count_at")),
    reinterpret_cast<LiteRtApi::GetTimeToFirstTokenFn>(dlsym(RTLD_DEFAULT, "litert_lm_benchmark_info_get_time_to_first_token")),
    reinterpret_cast<LiteRtApi::DeleteBenchmarkInfoFn>(dlsym(RTLD_DEFAULT, "litert_lm_benchmark_info_delete")),
  };
  return api;
}

static void setMinLogLevelCompat(int level) {
  const auto& api = getLiteRtApi();
  if (api.setMinLogLevel) {
    api.setMinLogLevel(level);
  }
}

static LiteRtLmEngineSettings* createEngineSettingsCompat(
    const char* modelPath,
    const char* backend,
    const char* visionBackend,
    const char* audioBackend) {
  const auto& api = getLiteRtApi();
  return api.createEngineSettings
    ? api.createEngineSettings(modelPath, backend, visionBackend, audioBackend)
    : nullptr;
}

static void deleteEngineSettingsCompat(LiteRtLmEngineSettings* settings) {
  const auto& api = getLiteRtApi();
  if (settings && api.deleteEngineSettings) {
    api.deleteEngineSettings(settings);
  }
}

static void setEngineMaxTokensCompat(LiteRtLmEngineSettings* settings, int maxTokens) {
  const auto& api = getLiteRtApi();
  if (settings && api.setEngineMaxTokens) {
    api.setEngineMaxTokens(settings, maxTokens);
  }
}

static void enableEngineBenchmarkCompat(LiteRtLmEngineSettings* settings) {
  const auto& api = getLiteRtApi();
  if (settings && api.enableEngineBenchmark) {
    api.enableEngineBenchmark(settings);
  }
}

static void setEngineCacheDirCompat(LiteRtLmEngineSettings* settings, const char* cacheDir) {
  const auto& api = getLiteRtApi();
  if (settings && cacheDir && api.setEngineCacheDir) {
    api.setEngineCacheDir(settings, cacheDir);
  }
}

static LiteRtLmEngine* createEngineCompat(LiteRtLmEngineSettings* settings) {
  const auto& api = getLiteRtApi();
  return api.createEngine ? api.createEngine(settings) : nullptr;
}

static void deleteEngineCompat(LiteRtLmEngine* engine) {
  const auto& api = getLiteRtApi();
  if (engine && api.deleteEngine) {
    api.deleteEngine(engine);
  }
}

static const char* getLastErrorCompat() {
  const auto& api = getLiteRtApi();
  return api.getLastError ? api.getLastError() : nullptr;
}

static const char* getJsonResponseStringCompat(const LiteRtLmJsonResponse* response) {
  const auto& api = getLiteRtApi();
  return api.getJsonResponseString ? api.getJsonResponseString(response) : nullptr;
}

static void deleteJsonResponseCompat(LiteRtLmJsonResponse* response) {
  const auto& api = getLiteRtApi();
  if (response && api.deleteJsonResponse) {
    api.deleteJsonResponse(response);
  }
}

static void deleteConversationCompat(LiteRtLmConversation* conversation) {
  const auto& api = getLiteRtApi();
  if (conversation && api.deleteConversation) {
    api.deleteConversation(conversation);
  }
}

static LiteRtLmSessionConfig* createSessionConfigCompat(const LiteRtLmSamplerParams& sampler, int maxTokens) {
  const auto& api = getLiteRtApi();
  if (api.hasModernSessionApi()) {
    auto* config = api.createSessionConfigNew ? api.createSessionConfigNew() : nullptr;
    if (!config) {
      return nullptr;
    }
    api.setMaxOutputTokens(config, maxTokens);
    api.setSamplerParams(config, &sampler);
    return config;
  }
  return api.createSessionConfigOld ? api.createSessionConfigOld(&sampler) : nullptr;
}

static void deleteConversationConfigCompat(LiteRtLmConversationConfig* config) {
  if (!config) {
    return;
  }
  const auto& api = getLiteRtApi();
  if (api.deleteConversationConfig) {
    api.deleteConversationConfig(config);
  }
}

static void deleteSessionConfigCompat(LiteRtLmSessionConfig* config) {
  const auto& api = getLiteRtApi();
  if (config && api.deleteSessionConfig) {
    api.deleteSessionConfig(config);
  }
}

static LiteRtLmConversation* createConversationCompat(
    LiteRtLmEngine* engine,
    LiteRtLmSessionConfig* sessionConfig,
    const char* systemMessageJson,
    LiteRtLmConversationConfig** outConfig) {
  if (outConfig) {
    *outConfig = nullptr;
  }

  const auto& api = getLiteRtApi();
  if (api.hasModernConversationApi()) {
    auto* config = api.createConversationConfig(
      engine,
      sessionConfig,
      systemMessageJson,
      nullptr,
      nullptr,
      false
    );
    if (!config) {
      return nullptr;
    }

    auto* conversation = api.createConversationNew(engine, config);
    if (!conversation) {
      api.deleteConversationConfig(config);
      return nullptr;
    }

    if (outConfig) {
      *outConfig = config;
    }
    return conversation;
  }

  return api.createConversationOld ? api.createConversationOld(engine) : nullptr;
}

static LiteRtLmJsonResponse* sendConversationMessageCompat(
    LiteRtLmConversation* conversation,
    const std::string& messageJson) {
  const auto& api = getLiteRtApi();
  if (api.hasModernConversationApi()) {
    return api.sendMessageNew ? api.sendMessageNew(conversation, messageJson.c_str(), nullptr) : nullptr;
  }
  return api.sendMessageOld ? api.sendMessageOld(conversation, messageJson.c_str()) : nullptr;
}

static int sendConversationMessageStreamCompat(
    LiteRtLmConversation* conversation,
    const std::string& messageJson,
    LiteRtLmStreamCallback callback,
    void* callbackData) {
  const auto& api = getLiteRtApi();
  if (api.hasModernConversationApi()) {
    return api.sendMessageStreamNew
      ? api.sendMessageStreamNew(conversation, messageJson.c_str(), nullptr, callback, callbackData)
      : -1;
  }
  return api.sendMessageStreamOld
    ? api.sendMessageStreamOld(conversation, messageJson.c_str(), callback, callbackData)
    : -1;
}
#endif

// =============================================================================
// JSON Helpers
// =============================================================================

std::string HybridLiteRTLM::escapeJson(const std::string& input) {
  std::string output;
  output.reserve(input.size() + 16);
  for (char c : input) {
    switch (c) {
      case '"':  output += "\\\""; break;
      case '\\': output += "\\\\"; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      case '\b': output += "\\b"; break;
      case '\f': output += "\\f"; break;
      default:   output += c; break;
    }
  }
  return output;
}

std::string HybridLiteRTLM::buildTextMessageJson(const std::string& text) {
  return "{\"role\":\"user\",\"content\":\"" + escapeJson(text) + "\"}";
}

std::string HybridLiteRTLM::buildImageMessageJson(const std::string& text, const std::string& imagePath) {
  return "{\"role\":\"user\",\"content\":["
         "{\"type\":\"text\",\"text\":\"" + escapeJson(text) + "\"},"
         "{\"type\":\"image\",\"path\":\"" + escapeJson(imagePath) + "\"}"
         "]}";
}

std::string HybridLiteRTLM::buildAudioMessageJson(const std::string& text, const std::string& audioPath) {
  return "{\"role\":\"user\",\"content\":["
         "{\"type\":\"text\",\"text\":\"" + escapeJson(text) + "\"},"
         "{\"type\":\"audio\",\"path\":\"" + escapeJson(audioPath) + "\"}"
         "]}";
}

static const char* kControlTokens[] = {
  "<end_of_turn>",
  "<start_of_turn>model",
  "<start_of_turn>user",
  "<start_of_turn>",
  "<eos>",
};

/**
 * Strip control tokens from model output, preserving whitespace.
 * Streaming chunks can include meaningful leading spaces that should not be
 * trimmed until the full response is complete.
 */
static std::string stripControlTokens(const std::string& text) {
  std::string result = text;
  for (auto* tok : kControlTokens) {
    std::string t(tok);
    size_t pos;
    while ((pos = result.find(t)) != std::string::npos) {
      result.erase(pos, t.length());
    }
  }
  return result;
}

static size_t safeEmitLength(const std::string& text) {
  size_t lastAngle = text.rfind('<');
  if (lastAngle == std::string::npos) {
    return text.length();
  }

  std::string suffix = text.substr(lastAngle);
  for (auto* tok : kControlTokens) {
    std::string token(tok);
    if (suffix.length() < token.length() && token.compare(0, suffix.length(), suffix) == 0) {
      return lastAngle;
    }
  }

  return text.length();
}

static std::string trimWhitespace(const std::string& text) {
  size_t start = text.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = text.find_last_not_of(" \t\n\r");
  return text.substr(start, end - start + 1);
}

std::string HybridLiteRTLM::extractTextFromResponse(const std::string& jsonResponse) {
  // The C API response JSON is structured as:
  //   {"role":"model","content":[{"type":"text","text":"..."}]}
  // or:
  //   {"role":"model","content":"..."}
  //
  // We use simple string extraction to avoid a JSON library dependency.
  
  // Try array format first: find "text":"..." after "type":"text"
  std::string textMarker = "\"text\":\"";
  size_t pos = jsonResponse.find("\"type\":\"text\"");
  if (pos != std::string::npos) {
    pos = jsonResponse.find(textMarker, pos);
    if (pos != std::string::npos) {
      pos += textMarker.length();
      std::string result;
      result.reserve(jsonResponse.size() - pos);
      for (size_t i = pos; i < jsonResponse.size(); i++) {
        if (jsonResponse[i] == '\\' && i + 1 < jsonResponse.size()) {
          char next = jsonResponse[i + 1];
          if (next == '"') { result += '"'; i++; }
          else if (next == '\\') { result += '\\'; i++; }
          else if (next == 'n') { result += '\n'; i++; }
          else if (next == 'r') { result += '\r'; i++; }
          else if (next == 't') { result += '\t'; i++; }
          else { result += jsonResponse[i]; }
        } else if (jsonResponse[i] == '"') {
          break;  // End of the text value
        } else {
          result += jsonResponse[i];
        }
      }
      return stripControlTokens(result);
    }
  }
  
  // Try simple string format: "content":"..."
  std::string contentMarker = "\"content\":\"";
  pos = jsonResponse.find(contentMarker);
  if (pos != std::string::npos) {
    pos += contentMarker.length();
    std::string result;
    for (size_t i = pos; i < jsonResponse.size(); i++) {
      if (jsonResponse[i] == '\\' && i + 1 < jsonResponse.size()) {
        char next = jsonResponse[i + 1];
        if (next == '"') { result += '"'; i++; }
        else if (next == '\\') { result += '\\'; i++; }
        else if (next == 'n') { result += '\n'; i++; }
        else { result += jsonResponse[i]; }
      } else if (jsonResponse[i] == '"') {
        break;
      } else {
        result += jsonResponse[i];
      }
    }
    return stripControlTokens(result);
  }
  
  // Fallback: return full response (still strip control tokens)
  return stripControlTokens(jsonResponse);
}

// =============================================================================
// Conversation Management
// =============================================================================

void HybridLiteRTLM::createNewConversation() {
#ifdef __APPLE__
  if (!engine_) {
    throw std::runtime_error("Cannot create conversation: engine not initialized");
  }
  
  // Clean up previous conversation
  if (conversation_) {
    deleteConversationCompat(conversation_);
    conversation_ = nullptr;
  }
  if (conv_config_) {
    deleteConversationConfigCompat(conv_config_);
    conv_config_ = nullptr;
  }
  
  // Build system message JSON if provided
  std::string systemMsgJson;
  const char* systemMsgPtr = nullptr;
  if (!systemPrompt_.empty()) {
    systemMsgJson = "{\"role\":\"system\",\"content\":\"" + escapeJson(systemPrompt_) + "\"}";
    systemMsgPtr = systemMsgJson.c_str();
  }
  
  // Create conversation config with session config
  conversation_ = createConversationCompat(
    engine_,
    session_config_,
    systemMsgPtr,
    &conv_config_
  );
  if (!conversation_) {
    throw std::runtime_error("Failed to create conversation");
  }
#endif
}

// =============================================================================
// loadModel
// =============================================================================

std::shared_ptr<Promise<void>> HybridLiteRTLM::loadModel(
    const std::string& modelPath,
    const std::optional<LLMConfig>& config) {
  return Promise<void>::async([this, modelPath, config]() {
    runOnLargeStack([&]() {
      loadModelInternal(modelPath, config);
    });
  });
}

void HybridLiteRTLM::loadModelInternal(
    const std::string& modelPath,
    const std::optional<LLMConfig>& config) {
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (isLoaded_) {
    close();
  }
  
  if (config.has_value()) {
    if (config->backend.has_value()) {
      backend_ = config->backend.value();
    }
    if (config->temperature.has_value()) {
      temperature_ = config->temperature.value();
    }
    if (config->topK.has_value()) {
      topK_ = config->topK.value();
    }
    if (config->topP.has_value()) {
      topP_ = config->topP.value();
    }
    if (config->maxTokens.has_value()) {
      maxTokens_ = config->maxTokens.value();
    }
    if (config->systemPrompt.has_value()) {
      systemPrompt_ = config->systemPrompt.value();
    }
  }
  
#ifdef __APPLE__
  // Set log verbosity: 2=WARNING (production), 0=INFO (debug)
  setMinLogLevelCompat(2);

  auto backendStr = [](Backend b) -> const char* {
    switch (b) {
      case Backend::GPU: return "gpu";
      case Backend::NPU: return "gpu"; // NPU not available on iOS, use GPU
      default: return "cpu";
    }
  };
  
  auto tryCreateEngine = [&](const char* backend, const char* visionBackend) -> bool {
    auto* settings = createEngineSettingsCompat(
      modelPath.c_str(),
      backend,
      visionBackend,
      "cpu" // audio executor: iOS XCFramework lacks compiled audio ops (INTERNAL ERROR at Invoke)
    );
    if (!settings) {
      return false;
    }
    
    setEngineMaxTokensCompat(settings, static_cast<int>(maxTokens_));
    enableEngineBenchmarkCompat(settings);
    
    // Set cache directory to the same directory as the model file
    std::string cacheDir = modelPath.substr(0, modelPath.find_last_of('/'));
    setEngineCacheDirCompat(settings, cacheDir.c_str());
    
    engine_ = createEngineCompat(settings);
    deleteEngineSettingsCompat(settings);
    
    return engine_ != nullptr;
  };
  
  // Try requested backend first (e.g. gpu/gpu)
  const char* primaryBackend = backendStr(backend_);
  if (!tryCreateEngine(primaryBackend, primaryBackend)) {
    // Fallback chain for when the primary backend fails:
    bool fallbackOk = false;
    if (backend_ != Backend::CPU) {
      // 1) Try CPU main + GPU vision (model's vision encoder often requires GPU)
      fallbackOk = tryCreateEngine("cpu", "gpu");
      // 2) Try CPU main + CPU vision
      if (!fallbackOk) fallbackOk = tryCreateEngine("cpu", "cpu");
    }
    // 3) Try CPU main + no vision (nullptr skips vision executor entirely)
    if (!fallbackOk) fallbackOk = tryCreateEngine("cpu", nullptr);
    if (fallbackOk) {
      backend_ = Backend::CPU;
    }
  }
  
  if (!engine_) {
    // Collect diagnostic info
    std::string diag = " | Diagnostics: ";
    struct stat st;
    if (stat(modelPath.c_str(), &st) == 0) {
      diag += "File size: " + std::to_string(st.st_size) + " bytes";
    } else {
      diag += "Failed to stat file (errno: " + std::to_string(errno) + ")";
    }
    
    FILE* f = fopen(modelPath.c_str(), "rb");
    if (f) {
      diag += ", Readable: YES";
      fclose(f);
    } else {
      diag += ", Readable: NO (errno: " + std::to_string(errno) + ")";
    }
    
    // Get the native error from the C API
    const char* nativeErr = getLastErrorCompat();
    if (nativeErr && nativeErr[0] != '\0') {
      diag += " | Native error: " + std::string(nativeErr);
    }

    throw std::runtime_error(
      "Failed to create LiteRT-LM engine. Tried backend '" +
      std::string(primaryBackend) + "' and CPU fallback. Model path: " + modelPath + diag);
  }
  
  LiteRtLmSamplerParams sampler{};
  sampler.type = kTopP;
  sampler.top_k = static_cast<int32_t>(topK_);
  sampler.top_p = static_cast<float>(topP_);
  sampler.temperature = static_cast<float>(temperature_);
  sampler.seed = 0;

  session_config_ = createSessionConfigCompat(sampler, static_cast<int>(maxTokens_));
  
  createNewConversation();
#endif
  
  isLoaded_ = true;
  history_.clear();
  lastStats_ = GenerationStats{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

// =============================================================================
// sendMessage — Blocking text inference
// =============================================================================

std::shared_ptr<Promise<std::string>> HybridLiteRTLM::sendMessage(const std::string& message) {
  return Promise<std::string>::async([this, message]() -> std::string {
    std::string result;
    runOnLargeStack([&]() {
      result = sendMessageInternal(message);
    });
    return result;
  });
}

std::string HybridLiteRTLM::sendMessageInternal(const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensureLoaded();
  
  auto startTime = std::chrono::steady_clock::now();
  std::string result;
  
#ifdef __APPLE__
  std::string promptText = message;
  if (!getLiteRtApi().hasModernConversationApi() && !systemPrompt_.empty() && history_.empty()) {
    promptText = systemPrompt_ + "\n\n" + message;
  }
  std::string msgJson = buildTextMessageJson(promptText);
  
  auto* response = sendConversationMessageCompat(conversation_, msgJson);
  
  if (!response) {
    throw std::runtime_error("LiteRT-LM: sendMessage failed");
  }
  
  const char* responseStr = getJsonResponseStringCompat(response);
  if (responseStr) {
    result = trimWhitespace(extractTextFromResponse(std::string(responseStr)));
  }
  deleteJsonResponseCompat(response);
  lastStats_.completionTokens = std::max(static_cast<double>(result.size()) / 4.0, 1.0);
  lastStats_.timeToFirstToken = 0.0;
  
  const auto& api = getLiteRtApi();
  if (api.hasBenchmarkApi()) {
    auto* benchInfo = api.getBenchmarkInfo(conversation_);
    if (benchInfo) {
      int numDecodeTurns = api.getNumDecodeTurns(benchInfo);
      if (numDecodeTurns > 0) {
        int lastIdx = numDecodeTurns - 1;
        lastStats_.tokensPerSecond = api.getDecodeTokensPerSecAt(benchInfo, lastIdx);
        lastStats_.completionTokens = static_cast<double>(
          api.getDecodeTokenCountAt(benchInfo, lastIdx));
      }
      lastStats_.timeToFirstToken = api.getTimeToFirstToken(benchInfo);
      api.deleteBenchmarkInfo(benchInfo);
    }
  }
#else
  // Non-Apple stub
  result = "[iOS only] LiteRT-LM inference not available on this platform.";
#endif
  
  auto endTime = std::chrono::steady_clock::now();
  double latencyMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  lastStats_.promptTokens = std::max(static_cast<double>(promptText.size()) / 4.0, 1.0);
  lastStats_.totalTokens = lastStats_.promptTokens + lastStats_.completionTokens;
  lastStats_.totalTime = latencyMs;
  
  // Update history
  history_.push_back(Message{Role::USER, message});
  history_.push_back(Message{Role::MODEL, result});
  
  return result;
}

// =============================================================================
// sendMessageAsync — Streaming text inference
// =============================================================================

void HybridLiteRTLM::streamCallbackFn(void* callback_data, const char* chunk,
                                        bool is_final, const char* error_msg) {
  auto* ctx = static_cast<StreamContext*>(callback_data);
  
  if (error_msg) {
    // Error occurred — notify JS and clean up
    ctx->onToken(std::string("Error: ") + error_msg, true);
    delete ctx;
    return;
  }
  
  if (is_final) {
    // Calculate stats
    auto endTime = std::chrono::steady_clock::now();
    double durationMs = std::chrono::duration<double, std::milli>(endTime - ctx->startTime).count();
    
    if (ctx->lastStats && ctx->tokenCount > 0) {
      ctx->lastStats->promptTokens = std::max(static_cast<double>(ctx->userMessage.size()) / 4.0, 1.0);
      ctx->lastStats->completionTokens = std::max(static_cast<double>(ctx->fullResponse.size()) / 4.0, 1.0);
      ctx->lastStats->totalTokens = ctx->lastStats->promptTokens + ctx->lastStats->completionTokens;
      ctx->lastStats->totalTime = durationMs;
      ctx->lastStats->tokensPerSecond = (ctx->lastStats->completionTokens / durationMs) * 1000.0;
      ctx->lastStats->timeToFirstToken = 0.0;
    }

    std::string cleaned = stripControlTokens(ctx->rawResponse);
    size_t start = cleaned.find_first_not_of(" \t\n\r");
    if (start != std::string::npos) {
      cleaned = cleaned.substr(start);
      if (!ctx->userMessage.empty() && cleaned.find(ctx->userMessage) == 0) {
        cleaned = cleaned.substr(ctx->userMessage.length());
        size_t nextStart = cleaned.find_first_not_of(" \t\n\r");
        cleaned = (nextStart != std::string::npos) ? cleaned.substr(nextStart) : "";
      }
      if (cleaned.length() > ctx->lastEmittedLength) {
        std::string remaining = cleaned.substr(ctx->lastEmittedLength);
        ctx->onToken(remaining, false);
      }
      ctx->fullResponse = trimWhitespace(cleaned);
    }
    
    // Update history (thread-safe)
    {
      std::lock_guard<std::mutex> lock(*ctx->historyMutex);
      ctx->history->push_back(Message{Role::USER, ctx->userMessage});
      ctx->history->push_back(Message{Role::MODEL, ctx->fullResponse});
    }
    
    ctx->onToken("", true);
    delete ctx;
    return;
  }
  
  if (chunk) {
    std::string token(chunk);

    std::string raw;
    if (token.size() > 2 && token[0] == '{' && token.find("\"role\"") != std::string::npos) {
      raw = HybridLiteRTLM::extractTextFromResponse(token);
    } else {
      raw = token;
    }

    ctx->rawResponse += raw;
    std::string cleaned = stripControlTokens(ctx->rawResponse);

    size_t start = cleaned.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
      return;
    }
    cleaned = cleaned.substr(start);

    if (!ctx->userMessage.empty() && cleaned.find(ctx->userMessage) == 0) {
      cleaned = cleaned.substr(ctx->userMessage.length());
      size_t nextStart = cleaned.find_first_not_of(" \t\n\r");
      if (nextStart == std::string::npos) {
        return;
      }
      cleaned = cleaned.substr(nextStart);
    }

    size_t safe = safeEmitLength(cleaned);
    if (safe > ctx->lastEmittedLength) {
      std::string newText = cleaned.substr(ctx->lastEmittedLength, safe - ctx->lastEmittedLength);
      ctx->fullResponse = cleaned.substr(0, safe);
      ctx->lastEmittedLength = safe;
      ctx->tokenCount++;
      ctx->onToken(newText, false);
    }
  }
}

void HybridLiteRTLM::sendMessageAsync(
    const std::string& message,
    const std::function<void(const std::string&, bool)>& onToken) {
  
  // Copy values for the background thread (avoid use-after-free)
  auto onTokenCopy = onToken;
  auto messageCopy = message;
  
  // Capture shared state safely — use unique_ptr to prevent leaks
  auto ctxOwner = std::make_unique<StreamContext>();
  ctxOwner->onToken = std::move(onTokenCopy);
  ctxOwner->rawResponse = "";
  ctxOwner->fullResponse = "";
  ctxOwner->lastEmittedLength = 0;
  ctxOwner->history = &history_;
  ctxOwner->historyMutex = &mutex_;
  ctxOwner->userMessage = messageCopy;
  ctxOwner->lastStats = &lastStats_;
  ctxOwner->startTime = std::chrono::steady_clock::now();
  ctxOwner->tokenCount = 0;
  
#ifdef __APPLE__
  ensureLoaded();
  
  std::string promptText = messageCopy;
  if (!getLiteRtApi().hasModernConversationApi() && !systemPrompt_.empty() && history_.empty()) {
    promptText = systemPrompt_ + "\n\n" + messageCopy;
  }
  std::string msgJson = buildTextMessageJson(promptText);
  
  // Release ownership — the C callback now owns the context via raw pointer.
  // streamCallbackFn will delete it when done or on error.
  StreamContext* ctx = ctxOwner.release();
  
  // Wrap the initial engine call in runOnLargeStack for consistency
  // with all other engine entry points (XNNPack needs >512KB stack).
  runOnLargeStack([&]() {
    int result = sendConversationMessageStreamCompat(
      conversation_, msgJson, streamCallbackFn, ctx);
    
    if (result != 0) {
      delete ctx;
      throw std::runtime_error("LiteRT-LM: Failed to start streaming inference");
    }
  });
#else
  // Non-Apple stub
  ctxOwner->onToken("[iOS only] Streaming not available on this platform.", true);
  // ctxOwner auto-deleted by unique_ptr
#endif
}

// =============================================================================
// runBenchmark — Dedicated benchmark turns with fresh conversation state
// =============================================================================

std::shared_ptr<Promise<std::vector<GenerationStats>>> HybridLiteRTLM::runBenchmark(
    const std::string& prompt,
    double warmupRuns,
    double benchmarkRuns) {
  return Promise<std::vector<GenerationStats>>::async([this, prompt, warmupRuns, benchmarkRuns]() {
    std::vector<GenerationStats> samples;

    runOnLargeStack([&]() {
      const int warmups = std::max(0, static_cast<int>(warmupRuns));
      const int measuredRuns = std::max(0, static_cast<int>(benchmarkRuns));

      if (prompt.empty()) {
        throw std::runtime_error("LiteRT-LM: Benchmark prompt cannot be empty.");
      }
      if (measuredRuns == 0) {
        throw std::runtime_error("LiteRT-LM: benchmarkRuns must be greater than zero.");
      }

      try {
        resetConversation();
        for (int index = 0; index < warmups; index += 1) {
          sendMessageInternal(prompt);
          resetConversation();
        }

        samples.reserve(static_cast<size_t>(measuredRuns));
        for (int index = 0; index < measuredRuns; index += 1) {
          sendMessageInternal(prompt);
          samples.push_back(getStats());
          resetConversation();
        }
      } catch (...) {
        try {
          resetConversation();
        } catch (...) {
        }
        throw;
      }
    });

    return samples;
  });
}

// =============================================================================
// sendMessageWithImage — Multimodal (vision)
// =============================================================================

std::shared_ptr<Promise<std::string>> HybridLiteRTLM::sendMessageWithImage(
    const std::string& message,
    const std::string& imagePath) {
  return Promise<std::string>::async([this, message, imagePath]() -> std::string {
    std::string result;
    runOnLargeStack([&]() {
      result = sendMessageWithImageInternal(message, imagePath);
    });
    return result;
  });
}

std::string HybridLiteRTLM::sendMessageWithImageInternal(
    const std::string& message,
    const std::string& imagePath) {
  
  std::lock_guard<std::mutex> lock(mutex_);
  ensureLoaded();
  
  auto startTime = std::chrono::steady_clock::now();
  std::string result;
  
#ifdef __APPLE__
  // Verify image exists
  std::ifstream imageFile(imagePath);
  if (!imageFile.good()) {
    throw std::runtime_error("Image file not found: " + imagePath);
  }
  imageFile.close();
  
  // Build multimodal message JSON — the C API handles image preprocessing
  std::string promptText = message;
  if (!getLiteRtApi().hasModernConversationApi() && !systemPrompt_.empty() && history_.empty()) {
    promptText = systemPrompt_ + "\n\n" + message;
  }
  std::string msgJson = buildImageMessageJson(promptText, imagePath);
  
  auto* response = sendConversationMessageCompat(conversation_, msgJson);
  
  if (!response) {
    std::string errMsg = "LiteRT-LM: sendMessageWithImage failed";
    const char* nativeErr = getLastErrorCompat();
    if (nativeErr && nativeErr[0] != '\0') {
      errMsg += ": " + std::string(nativeErr);
    }
    throw std::runtime_error(errMsg);
  }
  
  const char* responseStr = getJsonResponseStringCompat(response);
  if (responseStr) {
    result = trimWhitespace(extractTextFromResponse(std::string(responseStr)));
  }
  deleteJsonResponseCompat(response);
#else
  result = "[iOS only] Vision inference not available on this platform.";
#endif
  
  auto endTime = std::chrono::steady_clock::now();
  lastStats_.totalTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  
  history_.push_back(Message{Role::USER, message + " [image: " + imagePath + "]"});
  history_.push_back(Message{Role::MODEL, result});
  
  return result;
}

// =============================================================================
// sendMessageWithAudio — Multimodal (audio)
// =============================================================================

std::shared_ptr<Promise<std::string>> HybridLiteRTLM::sendMessageWithAudio(
    const std::string& message,
    const std::string& audioPath) {
  return Promise<std::string>::async([this, message, audioPath]() -> std::string {
    std::string result;
    runOnLargeStack([&]() {
      result = sendMessageWithAudioInternal(message, audioPath);
    });
    return result;
  });
}

std::string HybridLiteRTLM::sendMessageWithAudioInternal(
    const std::string& message,
    const std::string& audioPath) {
  
  std::lock_guard<std::mutex> lock(mutex_);
  ensureLoaded();
  
  auto startTime = std::chrono::steady_clock::now();
  std::string result;
  
#ifdef __APPLE__
  std::ifstream audioFile(audioPath);
  if (!audioFile.good()) {
    throw std::runtime_error("Audio file not found: " + audioPath);
  }
  audioFile.close();
  
  std::string promptText = message;
  if (!getLiteRtApi().hasModernConversationApi() && !systemPrompt_.empty() && history_.empty()) {
    promptText = systemPrompt_ + "\n\n" + message;
  }
  std::string msgJson = buildAudioMessageJson(promptText, audioPath);
  
  auto* response = sendConversationMessageCompat(conversation_, msgJson);
  
  if (!response) {
    std::string errMsg = "LiteRT-LM: sendMessageWithAudio failed";
    const char* nativeErr = getLastErrorCompat();
    if (nativeErr && nativeErr[0] != '\0') {
      errMsg += ": " + std::string(nativeErr);
    }
    throw std::runtime_error(errMsg);
  }
  
  const char* responseStr = getJsonResponseStringCompat(response);
  if (responseStr) {
    result = trimWhitespace(extractTextFromResponse(std::string(responseStr)));
  }
  deleteJsonResponseCompat(response);
#else
  result = "[iOS only] Audio inference not available on this platform.";
#endif
  
  auto endTime = std::chrono::steady_clock::now();
  lastStats_.totalTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  
  history_.push_back(Message{Role::USER, message + " [audio: " + audioPath + "]"});
  history_.push_back(Message{Role::MODEL, result});
  
  return result;
}

// =============================================================================
// downloadModel — Download model from URL
// =============================================================================

std::shared_ptr<Promise<std::string>> HybridLiteRTLM::downloadModel(
    const std::string& url,
    const std::string& fileName,
    const std::optional<std::function<void(double)>>& onProgress) {
  return Promise<std::string>::async([url, fileName, onProgress]() -> std::string {
#ifdef __APPLE__
    return litert_lm::downloadModelFile(url, fileName, onProgress);
#else
    // Non-Apple platforms: not supported from C++ (Android uses Kotlin)
    throw std::runtime_error("Download not available on this platform. Use the Kotlin implementation.");
#endif
  });
}

std::shared_ptr<Promise<void>> HybridLiteRTLM::deleteModel(const std::string& fileName) {
  return Promise<void>::async([fileName]() {
    std::string path;
#ifdef __APPLE__
    // Match the path used by IOSDownloadHelper: ~/Library/Caches/litert_models/
    const char* home = getenv("HOME");
    if (home) {
      path = std::string(home) + "/Library/Caches/litert_models/" + fileName;
    }
#else
    path = "/tmp/" + fileName;
#endif
    if (!path.empty()) {
      std::remove(path.c_str());
    }
  });
}

// =============================================================================
// getHistory
// =============================================================================

std::vector<Message> HybridLiteRTLM::getHistory() {
  std::lock_guard<std::mutex> lock(mutex_);
  return history_;
}

// =============================================================================
// resetConversation
// =============================================================================

void HybridLiteRTLM::resetConversation() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  history_.clear();
  lastStats_ = GenerationStats{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
#ifdef __APPLE__
  if (isLoaded_ && engine_) {
    createNewConversation();
  }
#endif
}

// =============================================================================
// isReady
// =============================================================================

bool HybridLiteRTLM::isReady() {
  std::lock_guard<std::mutex> lock(mutex_);
  return isLoaded_;
}

// =============================================================================
// getStats
// =============================================================================

GenerationStats HybridLiteRTLM::getStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  return lastStats_;
}

// =============================================================================
// getMemoryUsage — Uses Mach APIs for iOS process memory
// =============================================================================

MemoryUsage HybridLiteRTLM::getMemoryUsage() {
  double nativeHeapBytes = 0;
  double residentBytes = 0;
  double availableBytes = 0;
  bool isLowMemory = false;
  
#ifdef __APPLE__
  // Get app process memory (resident set size)
  struct mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  
  kern_return_t kr = task_info(mach_task_self(),
                               MACH_TASK_BASIC_INFO,
                               (task_info_t)&info,
                               &count);
  
  if (kr == KERN_SUCCESS) {
    residentBytes = static_cast<double>(info.resident_size);
    // On iOS, mach_task_basic_info doesn't separate heap from RSS.
    // Use resident_size_max as a proxy for peak native allocation.
    nativeHeapBytes = static_cast<double>(info.resident_size);
  }
  
  // Use os_proc_available_memory() (iOS 13+) for accurate Jetsam headroom.
  // This reports how much memory the process can still allocate before
  // the system kills it — far more accurate than total_physical - process_rss.
  availableBytes = static_cast<double>(os_proc_available_memory());
  
  // Low memory threshold (~200MB available)
  isLowMemory = availableBytes < 200.0 * 1024.0 * 1024.0;
#endif
  
  return MemoryUsage{
    nativeHeapBytes,            // nativeHeapBytes (RSS as proxy on iOS)
    residentBytes,              // residentBytes  
    availableBytes,             // availableMemoryBytes
    isLowMemory                 // isLowMemory
  };
}

// =============================================================================
// close — Clean up all LiteRT-LM resources
// =============================================================================

void HybridLiteRTLM::close() {
  // Note: Don't lock here if called from destructor (mutex may be destroyed)
  // The caller (loadModel, destructor) should handle locking.
  
  isLoaded_ = false;
  history_.clear();
  
#ifdef __APPLE__
  if (conversation_) {
    deleteConversationCompat(conversation_);
    conversation_ = nullptr;
  }
  if (conv_config_) {
    deleteConversationConfigCompat(conv_config_);
    conv_config_ = nullptr;
  }
  if (session_config_) {
    deleteSessionConfigCompat(session_config_);
    session_config_ = nullptr;
  }
  if (engine_) {
    deleteEngineCompat(engine_);
    engine_ = nullptr;
  }
#endif
  
  lastStats_ = GenerationStats{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

} // namespace margelo::nitro::litertlm
