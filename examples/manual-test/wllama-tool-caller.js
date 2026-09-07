/**
 * Wllama Tool Caller
 *
 * Grammar-constrained tool calling for wllama inference.
 * Uses GBNF grammar to force the model to output valid JSON matching
 * the tool call schema. Works reliably with any instruction-following model.
 */

/**
 * Encode a JSON primitive value as a GBNF literal terminal.
 * Used for `enum` constraints so the grammar only admits the declared values.
 *
 * @param {*} v - string | number | boolean | null
 * @returns {string|null} GBNF literal, or null if the value can't be encoded
 *                        as a simple terminal (objects/arrays).
 */
function _jsonLiteralToGbnf(v) {
  if (typeof v === 'string') {
    const esc = v.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
    return `"\\"${esc}\\""`;
  }
  if (typeof v === 'number') return `"${v}"`;
  if (typeof v === 'boolean') return v ? '"true"' : '"false"';
  if (v === null) return '"null"';
  return null;
}

/**
 * Resolve a JSON-Schema property to the GBNF rule reference that should match
 * its value. Emits any needed named sub-rules (enum/array) into `extraRules`.
 *
 * Supported: enum (any primitive), string, integer, number, boolean, typed
 * arrays of primitives. Anything else falls back to the generic `value` rule.
 *
 * @param {Object} schema - JSON Schema for a single property
 * @param {string} ruleBase - unique prefix for generated sub-rules (e.g. "a0-1")
 * @param {Array<string>} extraRules - accumulator for generated rule lines
 * @returns {string} GBNF rule reference (terminal or named rule)
 */
function _propValueRule(schema, ruleBase, extraRules) {
  if (!schema || typeof schema !== 'object') return 'value';

  // enum — constrain to the exact declared literals
  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    const alts = schema.enum.map(_jsonLiteralToGbnf);
    if (alts.every(Boolean)) {
      const ruleName = `${ruleBase}-enum`;
      extraRules.push(`${ruleName} ::= ${alts.join(' | ')}`);
      return ruleName;
    }
    // Non-primitive enum values — fall through to type-based handling.
  }

  switch (schema.type) {
    case 'string':
      return 'string';
    case 'integer':
      return 'integer';
    case 'number':
      return 'number';
    case 'boolean':
      return 'boolean';
    case 'array': {
      const itemRule = schema.items
        ? _propValueRule(schema.items, `${ruleBase}-i`, extraRules)
        : 'value';
      const ruleName = `${ruleBase}-arr`;
      extraRules.push(
        `${ruleName} ::= "[" ws "]" | "[" ws ${itemRule} (ws "," ws ${itemRule})* ws "]"`
      );
      return ruleName;
    }
    default:
      // object / unknown / untyped — generic JSON value
      return 'value';
  }
}

/**
 * Build the GBNF rule body that matches a single tool's `arguments` object,
 * constrained to that tool's JSON Schema. Required keys are emitted in a fixed
 * order; optional keys follow as `( "," key : val )?` groups.
 *
 * Falls back to the permissive generic `object` rule (returning usedFallback)
 * when the schema is absent, has no declared properties, or uses a shape we
 * don't strictly model (e.g. all-optional properties) — so the tool stays
 * callable rather than becoming unreachable.
 *
 * @param {Object} parameters - JSON Schema (type:object) for the tool
 * @param {number} toolIndex - position index, for unique sub-rule names
 * @param {Array<string>} extraRules - accumulator for generated rule lines
 * @returns {{body: string, usedFallback: boolean}}
 */
function _buildArgsRule(parameters, toolIndex, extraRules) {
  const hasProps =
    parameters &&
    parameters.properties &&
    typeof parameters.properties === 'object' &&
    Object.keys(parameters.properties).length > 0;

  if (!hasProps) {
    // Explicit empty-object schema → match exactly {}. Unknown/missing
    // schema → permissive object so the tool is still callable.
    if (parameters && parameters.type === 'object') {
      return { body: '"{" ws "}"', usedFallback: false };
    }
    return { body: 'object', usedFallback: true };
  }

  const props = parameters.properties;
  const required = Array.isArray(parameters.required)
    ? parameters.required
    : [];
  const keys = Object.keys(props);
  const requiredKeys = keys.filter((k) => required.includes(k));
  const optionalKeys = keys.filter((k) => !required.includes(k));

  // All-optional objects make leading-comma handling ambiguous in GBNF;
  // keep them permissive rather than risk an unsatisfiable grammar.
  if (requiredKeys.length === 0) {
    return { body: 'object', usedFallback: true };
  }

  const keyLiteral = (k) => {
    const esc = k.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
    return `"\\"${esc}\\""`;
  };

  const pairFor = (key) => {
    const j = keys.indexOf(key);
    const valRule = _propValueRule(
      props[key],
      `a${toolIndex}-${j}`,
      extraRules
    );
    return `${keyLiteral(key)} ws ":" ws ${valRule}`;
  };

  // Required keys: comma-joined head, fixed order.
  const head = requiredKeys.map(pairFor).join(' ws "," ws ');
  // Optional keys: each an independent ( "," pair )? after the head.
  const tail = optionalKeys.map((k) => ` (ws "," ws ${pairFor(k)})?`).join('');

  return { body: `"{" ws ${head}${tail} ws "}"`, usedFallback: false };
}

/**
 * Generate a GBNF grammar that constrains the model output to either a valid
 * tool call JSON object or the literal string "null" (no-tool sentinel).
 *
 * Each tool gets its own root alternative that binds the literal tool name to a
 * schema-faithful `arguments` rule — so the model cannot emit arguments that
 * don't match the selected tool's declared parameters (the #1 local tool-call
 * failure mode). Tools whose schema we can't strictly model fall back to a
 * permissive object so they stay callable.
 *
 * Two-pass design:
 *   - Pass 1 (constrained): model outputs a tool-call JSON OR "null".
 *   - If "null": the caller runs a second, unconstrained pass for the real answer.
 *   - If tool-call: arguments are parsed and the tool is executed.
 *
 * Output formats:
 *   Tool call:  {"name": "<tool_name>", "arguments": {<args>}}
 *   No tool:    null
 *
 * @param {Array} toolSchemas - OpenAI-format tool schemas
 * @returns {string} GBNF grammar string, or empty string if no tools
 */
function generateToolCallGrammar(toolSchemas) {
  if (!toolSchemas || toolSchemas.length === 0) {
    return '';
  }

  const tools = toolSchemas
    .map((s) => (s.function ? s.function : s))
    .filter((t) => t && t.name);

  if (tools.length === 0) return '';

  const rootAlts = [];
  const callRules = [];
  const argsRules = [];
  const extraRules = [];

  tools.forEach((tool, i) => {
    const callRule = `tc-${i}`;
    const argsRule = `args-${i}`;
    const nameEsc = String(tool.name)
      .replace(/\\/g, '\\\\')
      .replace(/"/g, '\\"');

    rootAlts.push(callRule);
    callRules.push(
      `${callRule} ::= "{" ws "\\"name\\"" ws ":" ws "\\"${nameEsc}\\"" ws "," ` +
        `ws "\\"arguments\\"" ws ":" ws ${argsRule} ws "}"`
    );

    const built = _buildArgsRule(tool.parameters, i, extraRules);
    argsRules.push(`${argsRule} ::= ${built.body}`);
  });

  rootAlts.push('"null"');

  // Generic fallback + shared primitives. `object`/`value`/`array` back the
  // permissive fallback and any untyped nested values.
  const shared = [
    'object ::= "{" ws "}" | "{" ws pair (ws "," ws pair)* ws "}"',
    'pair ::= string ws ":" ws value',
    'array ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"',
    'value ::= string | number | object | array | "true" | "false" | "null"',
    'string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""',
    'integer ::= "-"? [0-9]+',
    'number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?',
    'boolean ::= "true" | "false"',
    'ws ::= ([ \\t\\n\\r])*',
  ];

  return [
    `root ::= ${rootAlts.join(' | ')}`,
    ...callRules,
    ...argsRules,
    ...extraRules,
    ...shared,
  ].join('\n');
}

/**
 * Build a system prompt section describing available tools.
 * Injected into the system message so the model knows what tools exist.
 *
 * @param {Array} toolSchemas - OpenAI-format tool schemas
 * @returns {string} System prompt tool description
 */
function buildToolSystemPrompt(toolSchemas) {
  if (!toolSchemas || toolSchemas.length === 0) return '';

  const toolDescriptions = toolSchemas
    .map((schema) => {
      const fn = schema.function || schema;
      return JSON.stringify(
        {
          name: fn.name,
          description: fn.description || '',
          parameters: fn.parameters || {},
        },
        null,
        2
      );
    })
    .join('\n');

  return `You have access to the following tools. This is the first (constrained) pass: you must output EITHER a JSON tool call object OR the literal word null.

Available tools:
${toolDescriptions}

Rules:
- If a tool is needed, respond ONLY with the JSON object: {"name": "<tool_name>", "arguments": {<args>}}
- If no tool is needed, respond with exactly: null
- Do NOT include any other text, explanation, or formatting.`;
}

/**
 * Detect whether a response contains a tool call.
 * Parses grammar-constrained JSON output. Also handles edge cases where
 * the model wraps the JSON in XML-like tags.
 *
 * @param {string} response - Model output text
 * @param {Array} toolSchemas - Tool schemas (for validation)
 * @returns {Object|null} Parsed tool call {name, parameters} or null
 */
function detectToolCall(response, toolSchemas = []) {
  if (!response || typeof response !== 'string') return null;

  // Build a fast-lookup set of valid tool names when schemas are provided.
  // An empty toolSchemas array means "no validation" (e.g. during testing).
  const validNames =
    toolSchemas.length > 0
      ? new Set(
          toolSchemas
            .map((s) => (s.function ? s.function.name : s.name))
            .filter(Boolean)
        )
      : null;

  /** Normalise raw args to a plain object, never an array or primitive. */
  function normalizeArgs(raw) {
    if (raw && typeof raw === 'object' && !Array.isArray(raw)) return raw;
    return {};
  }

  /**
   * Validate a parsed JSON object against the known schema names, then build
   * the normalised tool-call result.  Returns null if invalid or unknown.
   */
  function validateAndBuild(parsed) {
    if (!parsed || !parsed.name || typeof parsed.name !== 'string') return null;
    if (validNames && !validNames.has(parsed.name)) return null;
    return {
      name: parsed.name,
      parameters: normalizeArgs(parsed.arguments || parsed.parameters),
    };
  }

  const trimmed = response.trim();

  // 1. Pure JSON (grammar-constrained output — primary path)
  if (trimmed.startsWith('{')) {
    try {
      const result = validateAndBuild(JSON.parse(trimmed));
      if (result) return result;
    } catch (_) {
      /* fall through */
    }
  }

  // 2. JSON wrapped in <tool_call> tags (some models emit these naturally)
  const tagMatch = trimmed.match(/<tool_call>([\s\S]*?)<\/tool_call>/i);
  if (tagMatch) {
    try {
      return validateAndBuild(JSON.parse(tagMatch[1].trim()));
    } catch (_) {
      /* ignore */
    }
  }

  return null;
}

// Export for CommonJS environments (testing) or make global in browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    generateToolCallGrammar,
    buildToolSystemPrompt,
    detectToolCall,
  };
} else {
  window.WllamaToolCaller = {
    generateToolCallGrammar,
    buildToolSystemPrompt,
    detectToolCall,
  };
}
