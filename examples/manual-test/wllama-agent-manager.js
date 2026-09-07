/**
 * WllamaAgentManager — Main agent orchestrator for wllama (llama.cpp) inference.
 *
 * Architecture:
 *   User message → formatChat → createChatCompletion → detectToolCall
 *     → if tool call: PyodideToolBridge.executeTool → synthesis pass
 *     → if no tool call: return response directly
 *
 * Tool calling uses a native-first approach with a grammar fallback:
 *   1. Native (wllama v3): tools/tool_choice are passed straight to
 *      createChatCompletion. The WASM side renders them through the model's
 *      own chat template and parses structured message.tool_calls from the
 *      output — no prompt injection or grammar needed.
 *   2. Grammar-Constrained (GBNF) fallback: when the chat template is not
 *      tool-aware or the native pass fails, a GBNF grammar forces the model
 *      to output valid JSON matching the tool call schema. Works reliably
 *      with any instruction-following model because it operates at the token
 *      sampling level.
 *
 * Threading: Single-thread WASM only. Multi-threading (SharedArrayBuffer) requires
 * COOP/COEP headers unavailable for standalone HTML files from file://.
 *
 * Supports:
 *   - Native OAI-compatible tool calling (tools / tool_choice / tool_calls)
 *   - Grammar-constrained tool calling (GBNF) as fallback
 *   - Streaming token output
 *   - Conversation memory with configurable history window
 *   - OPFS model caching via wllama
 */

class WllamaAgentManager {
  // Guardrail constants — small local models need tighter budgets than
  // frontier cloud models (mirrors max_iterations=5 / _MAX_TOOL_RESULT_CHARS
  // in the cloud loop at apps/agents/templates/mustache/unified_agent.mustache).
  static MAX_TOOL_SCHEMAS = 8; // GBNF grammar grows with tool count; cap to keep it manageable
  static MAX_TOKENS_DECISION_PASS = 256; // Grammar-constrained pass outputs only: null | JSON tool call
  static MAX_TOKENS_SYNTHESIS_PASS = 1024; // Unconstrained synthesis / final-answer pass

  constructor(config = {}) {
    this.wllama = null;
    this.toolBridge = null;
    this.toolSchemas = [];
    this.systemPrompt =
      config.systemPrompt || 'You are a helpful AI assistant.';
    this.modelConfig = null;
    this.functionCallingConfig = null;
    this.isLoaded = false;

    // Conversation memory
    this.chatHistory = [];
    this.memoryEnabled = config.memoryEnabled ?? window.MEMORY_ENABLED ?? true;
    this.maxHistoryTurns =
      config.maxHistoryTurns ?? window.MAX_HISTORY_TURNS ?? 5;
    // Cap user message length stored in history to avoid context overflow when
    // the initial user message is very long (e.g. full contract text).
    // Characters, not tokens; ~1500 chars ≈ 1125 tokens leaves ample room.
    this.maxHistoryUserMsgLength =
      config.maxHistoryUserMsgLength ??
      window.MAX_HISTORY_USER_MSG_LENGTH ??
      1500;

    // Callbacks
    this.onToken = config.onToken || null;
    this.onStatusChange = config.onStatusChange || null;

    // Set to true when grammar-constrained sampling fails (e.g. null function
    // in WebGPU WASM builds). Inference stays on GPU; only the GBNF pass is
    // skipped and replaced by unconstrained sampling + heuristic tool detection.
    this._grammarUnsupported = false;

    // Set to true when the native tools path fails for this session (e.g.
    // the model's chat template is not tool-aware). The GBNF grammar layer
    // is used as fallback instead.
    this._nativeToolsUnsupported = false;

    console.log('[WllamaAgent] Manager created', {
      memoryEnabled: this.memoryEnabled,
      maxHistoryTurns: this.maxHistoryTurns,
    });
  }

  /**
   * Set the Pyodide tool bridge for executing Python tools.
   * @param {PyodideToolBridge} bridge
   */
  setToolBridge(bridge) {
    this.toolBridge = bridge;
    console.log('[WllamaAgent] Tool bridge connected');
  }

  /**
   * Set tool schemas (OpenAI format).
   * @param {Array} schemas
   */
  setToolSchemas(schemas) {
    const all = schemas || [];
    if (all.length > WllamaAgentManager.MAX_TOOL_SCHEMAS) {
      console.warn(
        `[WllamaAgent] ${all.length} tool schemas supplied; ` +
          `capping to ${WllamaAgentManager.MAX_TOOL_SCHEMAS} to keep the local model coherent.`
      );
      this.toolSchemas = all.slice(0, WllamaAgentManager.MAX_TOOL_SCHEMAS);
    } else {
      this.toolSchemas = all;
    }
    console.log(
      `[WllamaAgent] ${this.toolSchemas.length} tool schema(s) registered`
    );
  }

  /**
   * Set function calling config (from model registry).
   * Stored and used in processMessage() to verify the model supports
   * grammar-constrained tool calling before applying the GBNF pass.
   * @param {Object} config - {method, tag_format, architecture}
   */
  setFunctionCallingConfig(config) {
    this.functionCallingConfig = config;
    console.log('[WllamaAgent] Function calling config:', config);
  }

  /**
   * Initialize wllama and load a GGUF model.
   *
   * Uses WllamaCacheManager (Cache API) instead of wllama's built-in OPFS
   * cache so standalone file:// agents work correctly.  OPFS requires a real
   * origin and throws SecurityError on file://; Cache API keys on the
   * cross-origin HuggingFace URL and is allowed from any page origin.
   *
   * Flow:
   *   1. Resolve the canonical model URL (hfRepo+hfFile → full HF URL, or modelUrl directly)
   *   2. Check Cache API for existing blob → loadModel([blob]) directly
   *   1. Resolve the canonical model URL (hfRepo+hfFile → full HF URL, or modelUrl directly)
   *   2. Check Cache API for existing blob → loadModel([blob]) directly
   *   3. Cache miss: fetch with progress → store in Cache API → loadModel([blob])
   *   4. No ObjectURL needed — blobs are transferred to the Worker via postMessage
   *
   * @param {Object} opts
   * @param {string} [opts.hfRepo] - HuggingFace repo ID (e.g. 'NousResearch/Hermes-2-Pro-Mistral-7B-GGUF')
   * @param {string} [opts.hfFile] - GGUF filename (e.g. 'Hermes-2-Pro-Mistral-7B.Q5_K_M.gguf')
   * @param {string} [opts.modelUrl] - Direct URL (alternative to hfRepo/hfFile)
   * @param {number} [opts.n_ctx=4096] - Context length
   * @param {number} [opts.n_gpu_layers=0] - GPU layers (WebGPU)
   * @param {Function} [opts.progressCallback] - Download progress callback({loaded, total})
   * @returns {Promise<void>}
   *   4. No ObjectURL needed — blobs are transferred to the Worker via postMessage
   *
   * @param {Object} opts
   * @param {string} [opts.hfRepo] - HuggingFace repo ID (e.g. 'NousResearch/Hermes-2-Pro-Mistral-7B-GGUF')
   * @param {string} [opts.hfFile] - GGUF filename (e.g. 'Hermes-2-Pro-Mistral-7B.Q5_K_M.gguf')
   * @param {string} [opts.modelUrl] - Direct URL (alternative to hfRepo/hfFile)
   * @param {number} [opts.n_ctx=4096] - Context length
   * @param {number} [opts.n_gpu_layers=0] - GPU layers (WebGPU)
   * @param {Function} [opts.progressCallback] - Download progress callback({loaded, total})
   * @returns {Promise<void>}
   */
  async initializeModel(opts = {}) {
    this._setStatus('loading');

    const wasmUrl =
      window.WLLAMA_WASM_URL || '/static/wasm/single-thread/wllama.wasm';

    console.log('[WllamaAgent] Creating wllama instance...');
    console.log('[WllamaAgent] WASM URL:', wasmUrl);

    const WllamaClass = window.Wllama;
    if (!WllamaClass) {
      throw new Error(
        'Wllama not loaded. Ensure wllama-bundle.js is imported.'
      );
    }

    const isFileProtocol = location.protocol === 'file:';

    // On file://, the blob: Worker created by wllama cannot fetch() a blob:
    // URL that was created in the main thread — blob URLs are scoped to their
    // creator's browsing context, and a Worker is a separate context.
    // Use a data: URL instead: it is self-contained (the bytes are encoded
    // directly into the URL string) so any Worker can read it without an
    // origin/context check.
    let resolvedWasmUrl = wasmUrl;
    if (isFileProtocol) {
      console.log('[WllamaAgent] file:// — encoding WASM as data: URL...');
      const wasmResponse = await fetch(wasmUrl);
      if (!wasmResponse.ok) {
        throw new Error(`Failed to fetch WASM: ${wasmResponse.status}`);
      }
      const wasmBuffer = await wasmResponse.arrayBuffer();
      // Chunked base64 — avoids stack overflow on the ~4 MB WASM file
      const bytes = new Uint8Array(wasmBuffer);
      let binary = '';
      const chunk = 8192;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      resolvedWasmUrl = `data:application/wasm;base64,${btoa(binary)}`;
      // No blob URL to track — data: URLs are self-contained
      console.log('[WllamaAgent] ✅ WASM encoded as data: URL');
    }

    // wllama >= 2.4 uses { default: url } pathConfig (previously 'single-thread/wllama.wasm')
    this.wllama = new WllamaClass({ default: resolvedWasmUrl });

    // wllama v3.x sets setCompat("default") in its constructor, which overrides
    // the provided WASM URL with CDN compat resources whenever needCompat() is
    // true (i.e. JSPI/Mem64 unsupported — the common case).  The single-thread
    // WASM we host works on all browsers without JSPI, so disable compat mode
    // to ensure our WASM is used and no CDN dependency is introduced.
    if (typeof this.wllama.setCompat === 'function') {
      this.wllama.setCompat(null);
    }

    const hasWebGPU = !!navigator.gpu;
    const nGpuLayers = opts.n_gpu_layers ?? (hasWebGPU ? 99 : 0);
    console.log(
      `[WllamaAgent] WebGPU: ${hasWebGPU ? 'available' : 'not available'}, GPU layers: ${nGpuLayers}`
    );

    // ── Resolve the canonical model URL ─────────────────────────────
    let modelUrl = opts.modelUrl;
    if (!modelUrl) {
      if (opts.hfRepo && opts.hfFile) {
        // Construct the same URL wllama uses internally so Cache API hits work
        modelUrl = `https://huggingface.co/${opts.hfRepo}/resolve/main/${opts.hfFile}`;
      } else {
        throw new Error('Either modelUrl or hfRepo+hfFile must be provided');
      }
    }

    const loadConfig = {
      n_ctx: opts.n_ctx || 4096,
      n_gpu_layers: nGpuLayers,
    };

    // ── Cache API lookup ─────────────────────────────────────────────
    // Blobs are passed to loadModel([blob]) via postMessage (transferable),
    // so the Worker never needs to access a URL — no cross-origin issues.
    const cache = window.WllamaCacheManager;

    if (cache) {
      let blob = await cache.get(modelUrl);
      if (blob) {
        console.log('[WllamaAgent] ✅ Cache hit — loading from Cache API');
      } else {
        console.log('[WllamaAgent] Cache miss — downloading model...');
        blob = await this._fetchAndCache(
          modelUrl,
          opts.progressCallback,
          cache
        );
      }
      await this._loadModelWithGpuFallback(
        (fallbackConfig) => this.wllama.loadModel([blob], fallbackConfig),
        loadConfig
      );
    } else {
      // WllamaCacheManager absent — fall back to direct load.
      // OPFS will fail on file:// but this path only runs when the
      // cache manager script was not loaded.
      console.warn(
        '[WllamaAgent] No cache manager — falling back to direct HF load'
      );
      if (!opts.hfRepo || !opts.hfFile) {
        throw new Error(
          'No cache manager available and no hfRepo+hfFile provided'
        );
      }
      const progressCb =
        opts.progressCallback || this._defaultProgressCallback.bind(this);
      await this._loadModelWithGpuFallback(
        (fallbackConfig) =>
          this.wllama.loadModelFromHF(opts.hfRepo, opts.hfFile, {
            ...fallbackConfig,
            progressCallback: progressCb,
          }),
        loadConfig
      );
    }

    this.modelConfig = { ...opts, resolvedUrl: modelUrl };
    this.isLoaded = true;
    this._setStatus('ready');
    console.log('[WllamaAgent] ✅ Model loaded successfully');
  }

  async _loadModelWithGpuFallback(loadFn, loadConfig) {
    // No CPU fallback — GPU is required.  Any WebGPU error is propagated
    // immediately so the user sees it rather than silently degrading to CPU.
    await loadFn(loadConfig);
  }

  /**
   * Fetch a model with progress reporting, store in Cache API, return the Blob.
   *
   * The write to cache is fire-and-forget (non-blocking) so a write failure
   * never prevents the model from loading.
   *
   * @param {string} url - Canonical model URL (https://huggingface.co/…)
   * @param {Function|undefined} progressCallback - ({loaded, total}) callback
   * @param {Object} cache - WllamaCacheManager instance
   * @returns {Promise<Blob>} The downloaded model blob (already written to cache)
   * @private
   */
  async _fetchAndCache(url, progressCallback, cache) {
    const progressCb =
      progressCallback || this._defaultProgressCallback.bind(this);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch model: ${response.status} ${response.statusText}`
      );
    }

    const contentLength = response.headers.get('Content-Length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      progressCb({ loaded, total });
    }

    const blob = new Blob(chunks, { type: 'application/octet-stream' });

    // Write to cache asynchronously — don't block model loading on write errors
    cache.put(url, blob).catch((err) => {
      if (err && err.name === 'QuotaExceededError') {
        // Surface quota failures: silently swallowing them means the
        // multi-GB model re-downloads on every page load.
        const msg =
          'Model too large to cache — it will re-download next time (browser storage quota exceeded)';
        console.error('[WllamaAgent] ❌ ' + msg, err);
        if (this.onStatusChange) {
          try {
            this.onStatusChange('cache_quota_exceeded', msg);
          } catch (_) {
            /* status UI is best-effort */
          }
        }
      } else {
        console.warn('[WllamaAgent] Cache write failed (non-fatal):', err);
      }
    });

    // Return the blob directly; caller passes it to loadModel([blob])
    return blob;
  }

  /**
   * Process a user message through the full agent loop.
   *
   * @param {string} userMessage
   * @param {Object} [opts]
   * @param {number} [opts.maxTokens=1024] - Max tokens to generate
   * @param {number} [opts.temperature=0.7]
   * @param {boolean} [opts.stream=false] - Whether to stream tokens
   * @param {Function} [opts.onToken] - Per-token callback for streaming
   * @returns {Promise<string>} Final response text
   */
  async processMessage(userMessage, opts = {}) {
    if (!this.isLoaded || !this.wllama) {
      throw new Error('Model not loaded. Call initializeModel() first.');
    }

    this._setStatus('thinking');

    // Build messages array
    const messages = this._buildMessages(userMessage);

    // Tool calling setup: only use grammar constraints when the model's
    // functionCallingConfig confirms grammar-wllama architecture (or when
    // no config has been set yet, which keeps backward compatibility).
    const grammarSupported =
      !this.functionCallingConfig ||
      this.functionCallingConfig.architecture === 'grammar-wllama';
    const hasTools =
      this.toolSchemas.length > 0 && this.toolBridge && grammarSupported;

    let sampling = {
      temp: opts.temperature ?? 0.7,
      top_p: 0.9,
      top_k: 40,
    };

    if (hasTools && window.WllamaToolCaller) {
      // ── Native tool calling (wllama v3) ──────────────────────────
      // Preferred path: tools/tool_choice go straight to the WASM, the
      // model's own chat template formats the tool definitions, and
      // llama.cpp parses structured tool_calls from the output. Falls
      // back to the GBNF grammar layer below when the template is not
      // tool-aware or the native pass fails.
      if (!this._nativeToolsUnsupported) {
        try {
          const response = await this._processWithNativeTools(
            userMessage,
            messages,
            sampling,
            opts
          );
          this._addToHistory(userMessage, response);
          this._setStatus('ready');
          return response;
        } catch (nativeErr) {
          const errMsg = nativeErr?.message || String(nativeErr);
          this._nativeToolsUnsupported = true;
          if (errMsg.includes('null function')) {
            // The native tool-call grammar uses the same sampler
            // machinery as the GBNF layer — a constrained retry
            // would trap identically, so disable grammar too.
            this._grammarUnsupported = true;
          }
          console.warn(
            '[WllamaAgent] Native tool calling unavailable — falling back to GBNF grammar layer:',
            errMsg
          );
        }
      }

      // ── GBNF grammar fallback ─────────────────────────────────────
      // Inject tool descriptions into system message
      const toolPrompt = window.WllamaToolCaller.buildToolSystemPrompt(
        this.toolSchemas
      );
      if (toolPrompt && messages.length > 0 && messages[0].role === 'system') {
        messages[0].content += '\n\n' + toolPrompt;
      }

      // ── Pass 1: tool-decision pass ────────────────────────────────
      // Prefer grammar-constrained sampling (GBNF forces valid JSON or
      // "null").  On some WebGPU/WASM builds grammar function pointers are
      // null — catch that error, disable grammar for the session, and
      // retry with unconstrained sampling (GPU remains fully active).
      const grammar = !this._grammarUnsupported
        ? window.WllamaToolCaller.generateToolCallGrammar(this.toolSchemas)
        : '';
      const constrainedSampling = grammar ? { ...sampling, grammar } : sampling;

      console.log(
        `[WllamaAgent] Pass 1: ${grammar ? 'constrained' : 'unconstrained'} tool-decision pass`
      );

      let pass1Response;
      try {
        pass1Response = this._extractContent(
          await this.wllama.createChatCompletion({
            messages,
            // Decision pass outputs only "null" or a JSON tool call — 256 tokens is ample.
            max_tokens: WllamaAgentManager.MAX_TOKENS_DECISION_PASS,
            ...constrainedSampling,
          })
        );
      } catch (grammarErr) {
        const errMsg = grammarErr?.message || String(grammarErr);
        if (errMsg.includes('null function') && grammar) {
          // Grammar sampler has a null function pointer in this WASM
          // build (common when WebGPU offloads all layers to GPU).
          // Disable grammar for the rest of this session and retry
          // with unconstrained sampling — GPU stays fully active.
          console.warn(
            '[WllamaAgent] Grammar sampling failed (null function in WASM). ' +
              'Disabling grammar for this session and retrying unconstrained ' +
              '(GPU remains active).',
            grammarErr
          );
          this._grammarUnsupported = true;
          pass1Response = this._extractContent(
            await this.wllama.createChatCompletion({
              messages,
              max_tokens: WllamaAgentManager.MAX_TOKENS_DECISION_PASS,
              ...sampling,
            })
          );
        } else {
          throw grammarErr;
        }
      }

      console.log('[WllamaAgent] Pass 1 response:', pass1Response.trim());

      const toolCall = window.WllamaToolCaller.detectToolCall(
        pass1Response,
        this.toolSchemas
      );

      if (toolCall) {
        // Tool call path: execute tool then synthesize
        console.log('[WllamaAgent] Tool call detected:', toolCall.name);
        this._setStatus('executing_tool');

        const toolResult = await this._executeTool(toolCall);

        // Synthesis pass: let the model summarize the tool result
        this._setStatus('synthesizing');
        const synthesisResponse = await this._synthesizeToolResult(
          userMessage,
          toolCall,
          toolResult,
          {
            sampling: { temp: sampling.temp },
            onToken: opts.onToken || this.onToken,
          }
        );

        this._addToHistory(userMessage, synthesisResponse);
        this._setStatus('ready');
        return synthesisResponse;
      }

      // ── Pass 2: unconstrained — model said "null" (no tool needed) ─
      // When grammar was active the model must have output exactly "null".
      // When grammar was inactive (this._grammarUnsupported) the model may
      // have already produced the full conversational answer instead of
      // the literal sentinel — return it directly to avoid a redundant call.
      if (
        this._grammarUnsupported &&
        pass1Response.trim().toLowerCase() !== 'null'
      ) {
        console.log(
          '[WllamaAgent] Unconstrained pass returned direct answer — skipping Pass 2'
        );
        this._addToHistory(userMessage, pass1Response);
        this._setStatus('ready');
        return pass1Response;
      }

      // Run a second, unconstrained completion to get the real answer.
      console.log('[WllamaAgent] Pass 2: unconstrained response pass');
      let response;
      if (opts.onToken) {
        response = await this._streamCompletion(messages, {
          max_tokens:
            opts.maxTokens || WllamaAgentManager.MAX_TOKENS_SYNTHESIS_PASS,
          ...sampling,
          onNewToken: opts.onToken,
        });
      } else {
        response = this._extractContent(
          await this.wllama.createChatCompletion({
            messages,
            max_tokens:
              opts.maxTokens || WllamaAgentManager.MAX_TOKENS_SYNTHESIS_PASS,
            ...sampling,
          })
        );
      }

      console.log('[WllamaAgent] Pass 2 response length:', response.length);
      this._addToHistory(userMessage, response);
      this._setStatus('ready');
      return response;
    }

    // No tools configured — single unconstrained pass
    let response;
    if (opts.onToken) {
      response = await this._streamCompletion(messages, {
        max_tokens:
          opts.maxTokens || WllamaAgentManager.MAX_TOKENS_SYNTHESIS_PASS,
        ...sampling,
        onNewToken: opts.onToken,
      });
    } else {
      response = this._extractContent(
        await this.wllama.createChatCompletion({
          messages,
          max_tokens:
            opts.maxTokens || WllamaAgentManager.MAX_TOKENS_SYNTHESIS_PASS,
          ...sampling,
        })
      );
    }

    console.log('[WllamaAgent] Response length:', response.length);
    this._addToHistory(userMessage, response);
    this._setStatus('ready');
    return response;
  }

  /**
   * Destroy the wllama instance and free memory.
   */
  async destroy() {
    if (this.wllama) {
      await this.wllama.exit();
      this.wllama = null;
    }
    // No blob URL to revoke — WASM is encoded as a data: URL on file://
    this.isLoaded = false;
    this.chatHistory = [];
    console.log('[WllamaAgent] Destroyed');
  }

  // ── Private methods ──────────────────────────────────────────────

  _buildMessages(userMessage) {
    const messages = [];

    // System prompt
    messages.push({ role: 'system', content: this.systemPrompt });

    // Conversation history (windowed)
    if (this.memoryEnabled && this.chatHistory.length > 0) {
      const historyWindow = this.chatHistory.slice(-this.maxHistoryTurns * 2);
      messages.push(...historyWindow);
    }

    // Current user message
    messages.push({ role: 'user', content: userMessage });

    return messages;
  }

  _addToHistory(userMessage, assistantResponse) {
    if (!this.memoryEnabled) return;
    // Truncate very long user messages so they don't overflow the model's
    // context window in subsequent follow-up turns.
    const storedUserMsg =
      this.maxHistoryUserMsgLength > 0 &&
      userMessage.length > this.maxHistoryUserMsgLength
        ? userMessage.slice(0, this.maxHistoryUserMsgLength) +
          '\n[...truncated for context window...]'
        : userMessage;
    this.chatHistory.push({ role: 'user', content: storedUserMsg });
    this.chatHistory.push({ role: 'assistant', content: assistantResponse });

    // Trim history to max turns
    const maxMessages = this.maxHistoryTurns * 2;
    if (this.chatHistory.length > maxMessages) {
      this.chatHistory = this.chatHistory.slice(-maxMessages);
    }
  }

  async _streamCompletion(messages, options) {
    let fullText = '';
    const { onNewToken, ...samplingOpts } = options;
    const iter = await this.wllama.createChatCompletion({
      messages,
      ...samplingOpts,
      stream: true,
    });
    for await (const chunk of iter) {
      const delta = chunk.choices?.[0]?.delta?.content || '';
      if (delta) {
        fullText += delta;
        if (onNewToken) {
          onNewToken(null, delta, fullText, {});
        }
      }
    }
    return fullText;
  }

  /**
   * Native tool-calling loop using wllama v3's OAI-compatible API.
   *
   * tools/tool_choice are passed straight to createChatCompletion — the
   * WASM side renders them through the model's own chat template and parses
   * the output into structured message.tool_calls. No prompt injection or
   * GBNF grammar involved.
   *
   * Loop: completion → tool_calls? → execute via bridge → append
   * assistant(tool_calls) + tool(result) messages → completion again for
   * the final answer (up to MAX_TOOL_ITERATIONS rounds).
   *
   * Error contract: throws only while no tool has executed yet, so the
   * caller can safely fall back to the GBNF layer. Once a tool has run,
   * failures are salvaged with the plain-text synthesis pass instead —
   * falling back would re-execute the tool.
   *
   * @param {string} userMessage - Original user message (for synthesis salvage)
   * @param {Array} messages - Chat messages without any tool prompt injected
   * @param {Object} sampling - Sampling params {temp, top_p, top_k}
   * @param {Object} opts - processMessage options (maxTokens, onToken)
   * @returns {Promise<string>} Final response text
   */
  async _processWithNativeTools(userMessage, messages, sampling, opts) {
    // Normalise schemas to the OpenAI wrapper format the chat template
    // expects; setToolSchemas accepts both wrapped and bare schemas.
    const tools = this.toolSchemas.map((schema) => {
      const fn = schema.function || schema;
      return {
        type: 'function',
        function: {
          name: fn.name,
          description: fn.description || '',
          parameters: fn.parameters || {},
        },
      };
    });

    const nativeMessages = [...messages];
    const MAX_TOOL_ITERATIONS = 3;
    let toolsExecuted = 0;
    let lastToolCall = null;
    let lastToolResult = null;
    const synthesisOpts = {
      sampling: { temp: sampling.temp },
      onToken: opts.onToken || this.onToken,
    };

    for (let round = 1; round <= MAX_TOOL_ITERATIONS; round++) {
      console.log(`[WllamaAgent] Native tools round ${round}`);

      let completion;
      try {
        completion = await this._nativeToolCompletion(
          nativeMessages,
          tools,
          sampling,
          opts
        );
      } catch (err) {
        if (toolsExecuted === 0) throw err;
        console.warn(
          '[WllamaAgent] Native follow-up failed after tool execution — synthesizing from tool result:',
          err
        );
        return await this._synthesizeToolResult(
          userMessage,
          lastToolCall,
          lastToolResult,
          synthesisOpts
        );
      }
      const { fullContent, toolCalls } = completion;

      if (toolCalls.length === 0) {
        // Direct answer — no tool needed (or tool results consumed).
        if (fullContent) return fullContent;
        if (toolsExecuted > 0) {
          // Model went silent after the tool ran — synthesize instead.
          return await this._synthesizeToolResult(
            userMessage,
            lastToolCall,
            lastToolResult,
            synthesisOpts
          );
        }
        throw new Error('Native tools pass produced no output');
      }

      // Append the assistant turn exactly as the template expects it,
      // then execute each call and append its result as a tool message.
      const assistantMsg = {
        role: 'assistant',
        content: fullContent || null,
        tool_calls: toolCalls.map((tc, i) => ({
          id: tc.id || `call_${round}_${i}`,
          type: 'function',
          function: { name: tc.name, arguments: tc.arguments },
        })),
      };
      nativeMessages.push(assistantMsg);

      this._setStatus('executing_tool');
      for (let i = 0; i < toolCalls.length; i++) {
        const tc = toolCalls[i];
        let args;
        try {
          args = JSON.parse(tc.arguments || '{}');
        } catch (_) {
          args = {};
        }
        if (!args || typeof args !== 'object' || Array.isArray(args)) {
          args = {};
        }
        const toolCall = { name: tc.name, parameters: args };
        console.log(
          '[WllamaAgent] Native tool call:',
          toolCall.name,
          toolCall.parameters
        );

        const toolResult = await this._executeTool(toolCall);
        toolsExecuted++;
        lastToolCall = toolCall;
        lastToolResult = toolResult;

        // Same context-window guard as _synthesizeToolResult: a long
        // tool result on the follow-up pass can trigger a GPU TDR crash.
        const resultStr =
          typeof toolResult === 'string'
            ? toolResult
            : JSON.stringify(toolResult);
        const NATIVE_MAX_RESULT_CHARS = 2000;
        nativeMessages.push({
          role: 'tool',
          tool_call_id: assistantMsg.tool_calls[i].id,
          content:
            resultStr.length > NATIVE_MAX_RESULT_CHARS
              ? resultStr.slice(0, NATIVE_MAX_RESULT_CHARS) +
                '\n[...truncated for context window...]'
              : resultStr,
        });
      }
      this._setStatus('synthesizing');
      // Loop back so the model can turn the tool results into an answer.
    }

    console.warn(
      '[WllamaAgent] Native tools hit iteration limit — synthesizing from last tool result'
    );
    return await this._synthesizeToolResult(
      userMessage,
      lastToolCall,
      lastToolResult,
      synthesisOpts
    );
  }

  /**
   * One streamed native-tools completion. Content deltas are forwarded to
   * the token callback; delta.tool_calls fragments are accumulated by index
   * (id / function.name / function.arguments arrive in pieces).
   *
   * @param {Array} messages
   * @param {Array} tools - OpenAI-format tool definitions
   * @param {Object} sampling
   * @param {Object} opts - {maxTokens, onToken}
   * @returns {Promise<{fullContent: string, toolCalls: Array<{id, name, arguments}>, finishReason: string|null}>}
   */
  async _nativeToolCompletion(messages, tools, sampling, opts) {
    const onToken = opts.onToken || this.onToken;
    let fullContent = '';
    let finishReason = null;
    const collected = {}; // index → {id, name, arguments}

    const iter = await this.wllama.createChatCompletion({
      messages,
      tools,
      tool_choice: 'auto',
      max_tokens:
        opts.maxTokens || WllamaAgentManager.MAX_TOKENS_SYNTHESIS_PASS,
      ...sampling,
      stream: true,
    });

    for await (const chunk of iter) {
      const choice = chunk.choices?.[0];
      if (!choice) continue;
      finishReason = choice.finish_reason ?? finishReason;
      const delta = choice.delta || {};
      if (delta.content) {
        fullContent += delta.content;
        if (onToken) {
          onToken(null, delta.content, fullContent, {});
        }
      }
      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? 0;
          if (!collected[idx]) {
            collected[idx] = { id: '', name: '', arguments: '' };
          }
          if (tc.id) collected[idx].id = tc.id;
          if (tc.function?.name) collected[idx].name += tc.function.name;
          if (tc.function?.arguments)
            collected[idx].arguments += tc.function.arguments;
        }
      }
    }

    return { fullContent, toolCalls: Object.values(collected), finishReason };
  }

  async _executeTool(toolCall) {
    if (!this.toolBridge) {
      return { error: 'No tool bridge available' };
    }

    try {
      const result = await this.toolBridge.executeTool(
        toolCall.name,
        toolCall.parameters
      );
      console.log(`[WllamaAgent] Tool "${toolCall.name}" result:`, result);
      return result;
    } catch (error) {
      const msg = error.message || String(error);
      // Model sometimes hallucinates arguments for zero-parameter tools.
      // When Python rejects them with a TypeError, retry with no args so
      // the tool can still succeed on the second attempt.
      if (
        msg.includes('unexpected keyword argument') ||
        msg.includes('takes 0 positional argument')
      ) {
        console.warn(
          `[WllamaAgent] Tool "${toolCall.name}" rejected hallucinated args — retrying with no arguments`
        );
        try {
          const result = await this.toolBridge.executeTool(toolCall.name, {});
          console.log(
            `[WllamaAgent] Tool "${toolCall.name}" retry result:`,
            result
          );
          return result;
        } catch (retryErr) {
          console.error(
            `[WllamaAgent] Tool "${toolCall.name}" retry also failed:`,
            retryErr
          );
          return { error: retryErr.message || String(retryErr) };
        }
      }
      console.error(`[WllamaAgent] Tool "${toolCall.name}" error:`, error);
      return { error: msg };
    }
  }

  async _synthesizeToolResult(userMessage, toolCall, toolResult, opts = {}) {
    const toolFailed =
      typeof toolResult === 'object' &&
      toolResult !== null &&
      'error' in toolResult;
    const resultStr =
      typeof toolResult === 'string'
        ? toolResult
        : JSON.stringify(toolResult, null, 2);

    // Truncate long tool results before synthesis. A statistics table with
    // hundreds of decimal-precision numbers can push the synthesis prompt
    // past 1000 tokens, triggering a GPU TDR crash on the second inference
    // pass. 2000 chars covers the full overview + data types + start of
    // stats — more than enough for the LLM to write a good summary.
    const MAX_RESULT_CHARS = 2000;
    const truncatedResult =
      resultStr.length > MAX_RESULT_CHARS
        ? resultStr.slice(0, MAX_RESULT_CHARS) +
          '\n[...truncated for context window...]'
        : resultStr;

    const synthesisMessages = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: userMessage },
      {
        role: 'assistant',
        content: toolFailed
          ? `I tried to retrieve that information but encountered an error: ${toolResult.error}`
          : `I retrieved the following data:\n${truncatedResult}`,
      },
      {
        role: 'user',
        content:
          'Please provide a clear, helpful answer in plain natural language. Do not output JSON or tool calls — respond with text only.',
      },
    ];

    const synthSampling = opts.sampling || { temp: 0.7 };
    let response;
    if (opts.onToken) {
      response = await this._streamCompletion(synthesisMessages, {
        max_tokens: WllamaAgentManager.MAX_TOKENS_SYNTHESIS_PASS,
        ...synthSampling,
        onNewToken: opts.onToken,
      });
    } else {
      response = this._extractContent(
        await this.wllama.createChatCompletion({
          messages: synthesisMessages,
          max_tokens: WllamaAgentManager.MAX_TOKENS_SYNTHESIS_PASS,
          ...synthSampling,
        })
      );
    }

    // Guard: if synthesis echoed a tool-call JSON instead of natural text,
    // replace it with a structured fallback so the user never sees raw JSON.
    const trimmed = response ? response.trim() : '';
    if (trimmed.startsWith('{') && trimmed.includes('"name"')) {
      console.warn(
        '[WllamaAgent] Synthesis produced tool-call JSON — using fallback message'
      );
      return toolFailed
        ? 'I encountered an error while processing your request. Please try rephrasing your question.'
        : `Here are the results:\n${resultStr}`;
    }

    return (
      response ||
      (toolFailed
        ? 'I encountered an error while processing your request. Please try again.'
        : `Here are the results:\n${resultStr}`)
    );
  }

  /**
   * Extract text content from a wllama 2.4 ChatCompletionResponse.
   * Falls back to empty string on missing data.
   * @param {Object|string} response
   * @returns {string}
   */
  _extractContent(response) {
    if (!response) return '';
    if (typeof response === 'string') return response;
    return response.choices?.[0]?.message?.content ?? '';
  }

  _defaultProgressCallback({ loaded, total }) {
    if (total > 0) {
      const pct = Math.round((loaded / total) * 100);
      console.log(`[WllamaAgent] Download: ${pct}%`);
    }
  }

  _setStatus(status) {
    if (this.onStatusChange) {
      this.onStatusChange(status);
    }
  }
}

// Export for CommonJS environments (testing) or make global in browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { WllamaAgentManager };
} else {
  window.WllamaAgentManager = WllamaAgentManager;
}
