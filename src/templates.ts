/**
 * Prompt template utilities for different LLM families.
 *
 * LiteRT-LM's Conversation API may handle templates internally for some models,
 * but these utilities give developers explicit control for custom workflows
 * or when using models with different template formats.
 *
 * @example
 * ```typescript
 * import { applyGemmaTemplate, ChatMessage } from '@inferrlm/react-native-litert-lm';
 *
 * const history: ChatMessage[] = [
 *   { role: 'user', content: 'What is React Native?' },
 *   { role: 'model', content: 'React Native is a framework for building...' },
 *   { role: 'user', content: 'How do I use hooks?' }
 * ];
 *
 * const prompt = applyGemmaTemplate(history, 'You are a helpful coding assistant.');
 * ```
 */

/**
 * A message in a conversation.
 */
export type ChatMessage = {
  role: "user" | "model" | "system";
  content: string;
};

/**
 * Apply Gemma chat template (Gemma 2, Gemma 3, Gemma 3n).
 *
 * @param history Array of previous messages
 * @param systemPrompt Optional system prompt
 * @returns Formatted prompt string
 */
export function applyGemmaTemplate(
  history: ChatMessage[],
  systemPrompt?: string,
): string {
  let result = "";

  if (systemPrompt) {
    result += `<start_of_turn>system\n${systemPrompt}<end_of_turn>\n`;
  }

  for (const m of history) {
    result += `<start_of_turn>${m.role}\n${m.content}<end_of_turn>\n`;
  }

  result += "<start_of_turn>model\n";
  return result;
}

/**
 * Apply Phi chat template (Phi-3, Phi-4).
 *
 * @param history Array of previous messages
 * @param systemPrompt Optional system prompt
 * @returns Formatted prompt string
 */
export function applyPhiTemplate(
  history: ChatMessage[],
  systemPrompt?: string,
): string {
  let result = "";

  if (systemPrompt) {
    result += `<|system|>\n${systemPrompt}<|end|>\n`;
  }

  for (const m of history) {
    const role = m.role === "model" ? "assistant" : m.role;
    result += `<|${role}|>\n${m.content}<|end|>\n`;
  }

  result += "<|assistant|>\n";
  return result;
}

/**
 * Apply Llama 3 chat template.
 *
 * @param history Array of previous messages
 * @param systemPrompt Optional system prompt
 * @returns Formatted prompt string
 */
export function applyLlamaTemplate(
  history: ChatMessage[],
  systemPrompt?: string,
): string {
  let result = "<|begin_of_text|>";

  if (systemPrompt) {
    result += `<|start_header_id|>system<|end_header_id|>\n\n${systemPrompt}<|eot_id|>`;
  }

  for (const m of history) {
    const role = m.role === "model" ? "assistant" : m.role;
    result += `<|start_header_id|>${role}<|end_header_id|>\n\n${m.content}<|eot_id|>`;
  }

  result += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return result;
}
